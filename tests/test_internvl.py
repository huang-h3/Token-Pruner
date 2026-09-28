"""InternVL Instruct: complete pixel-shuffle cells, feature counts and prompt runs."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers.models.internvl.modeling_internvl")
from transformers import Qwen2Config
from transformers.models.internvl.configuration_internvl import (
    InternVLConfig,
    InternVLVisionConfig,
)
from transformers.models.internvl.modeling_internvl import (
    InternVLForConditionalGeneration,
    InternVLModel,
)

from token_pruner.cells import (
    cell_members,
    cell_view_cls,
    gather_cells,
    lay_down_cells,
    predict_cells,
    reduce_cells,
    split_group_features,
)
from token_pruner.layer_boundary import EncoderPruningContext
from token_pruner.run_internvl import (
    CELL_SIDE,
    InternVlBackend,
    install_feature_shim,
    install_pruning,
    merge_kept_cells,
    shrink_image_placeholders,
    tower_blocks,
)
from token_pruner.selection import LayerPruningPlan, resolve_pruning_budget
from token_pruner.tokens import (
    PruningConfig,
    PruningScope,
    PruningSignal,
    SignalSource,
    TokenReducer,
)

IMAGE, PATCH, BOUNDARY = 64, 16, 2
CELL_TOKENS = CELL_SIDE**2
GRID = IMAGE // PATCH
CELLS = (GRID // 2) ** 2
GROUPS = 2
IMAGE_TOKEN = 150


class _FirstKFolder:
    """Keep the first ``keep`` tokens in place of the external FOLDER merge."""

    def reduce(self, _cls_token, frame_tokens, keep):
        return frame_tokens[:, :keep]


@pytest.fixture
def first_k_folder(monkeypatch):
    monkeypatch.setattr("token_pruner.engine.folder_wrapper", _FirstKFolder)


def _hf_model():
    vision = InternVLVisionConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=IMAGE,
        patch_size=PATCH,
    )
    text = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=200,
    )
    config = InternVLConfig(
        vision_config=vision,
        text_config=text,
        image_token_id=IMAGE_TOKEN,
        downsample_ratio=0.5,
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
    )
    return InternVLForConditionalGeneration(config).eval()


class _Vision(torch.nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.tower = tower
        self.encoder = tower.encoder
        # The checkpoint's InternViT names its blocks ``encoder.layers``.
        self.encoder.layers = self.encoder.layer

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        return self.tower(pixel_values=pixel_values)


class _ChatModel(torch.nn.Module):
    """The checkpoint's ``InternVLChatModel`` feature path, at toy size."""

    def __init__(self):
        super().__init__()
        hf = _hf_model()
        self.config = hf.config
        self.vision_model = _Vision(hf.model.vision_tower)
        self.mlp1 = hf.model.multi_modal_projector
        self.select_layer = -1

    def extract_feature(self, pixel_values):
        outputs = self.vision_model(
            pixel_values, output_hidden_states=self.select_layer != -1
        )
        features = (
            outputs.last_hidden_state
            if self.select_layer == -1
            else outputs.hidden_states[self.select_layer]
        )
        patches = features[:, 1:]
        batch, _, hidden = patches.shape
        merged = InternVLModel.pixel_shuffle(
            None, patches.reshape(batch, GRID, GRID, hidden), scale_factor=0.5
        )
        return self.mlp1(merged.reshape(batch, CELLS, -1))


def _model():
    return _ChatModel().eval()


def merge_cells(tokens, *, grid_side, cells=None):
    members = gather_cells(
        tokens, grid_side=grid_side, cell_side=CELL_SIDE, cells=cells
    )
    return members.flatten(2)


def _config(scope, i_mode, p_mode, keep):
    return PruningConfig(
        scope=PruningScope(scope),
        reducer=TokenReducer.FOLDER if p_mode == "folder" else TokenReducer.GATHER,
        anchor_reducer=(
            TokenReducer.FOLDER if i_mode == "folder" else TokenReducer.PRESERVE
        ),
        signal_source=SignalSource.NONE if p_mode == "folder" else SignalSource.HEVC,
        keep_per_partition=keep,
        keep_total=keep * GROUPS,
        min_per_partition=1,
        uniform_partitions=True,
    )


def _plan(keep):
    """Every partition keeps its first ``keep`` cells; no anchor group."""

    config = PruningConfig(
        scope=PruningScope.LOCAL,
        reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE,
        keep_per_partition=keep,
        keep_total=keep * GROUPS,
        signal_source=SignalSource.HEVC,
    )
    signal = PruningSignal(
        visible_indices=[
            torch.tensor(
                [
                    group * CELLS + cell
                    for group in range(GROUPS)
                    for cell in range(keep)
                ]
            )
        ],
        anchor_partitions=torch.tensor([[False] * GROUPS]),
    )
    return LayerPruningPlan(BOUNDARY, config, signal)


def _signal(keep, anchor=0):
    """Every non-anchor partition keeps its first ``keep`` cells."""

    return PruningSignal(
        visible_indices=[
            torch.tensor(
                [
                    g * CELLS + c
                    for g in range(GROUPS)
                    if g != anchor
                    for c in range(keep)
                ]
            )
        ],
        anchor_partitions=torch.tensor([[g == anchor for g in range(GROUPS)]]),
    )


def _ragged_plan(keep_total):
    """A preserved anchor takes its whole frame; the rest share what is left."""

    config = PruningConfig(
        scope=PruningScope.LOCAL,
        reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE,
        signal_source=SignalSource.HEVC,
        keep_per_partition=keep_total // GROUPS,
        keep_total=keep_total,
        min_per_partition=1,
    )
    budget = resolve_pruning_budget(
        config, partition_sizes={g: CELLS for g in range(GROUPS)}, anchor_partitions=[0]
    )
    visible = [
        torch.arange(budget.other_allocations[g]) + g * CELLS
        for g in range(GROUPS)
        if g not in budget.anchor_partitions
    ]
    signal = PruningSignal(
        visible_indices=[torch.cat(visible)],
        anchor_partitions=torch.tensor([[g == 0 for g in range(GROUPS)]]),
    )
    widths = [
        CELLS if g in budget.anchor_partitions else budget.other_allocations[g]
        for g in range(GROUPS)
    ]
    return LayerPruningPlan(BOUNDARY, config, signal), widths


def _run(model, context, pixels, state, keep):
    context.plan = _plan(keep)
    context.enabled = True
    with torch.no_grad():
        model.vision_model(pixel_values=pixels)
    state.update(cells=context.kept_cells, grid_side=GRID, num_groups=GROUPS)
    with torch.no_grad():
        return model.extract_feature(pixels)


def _reduce(plan, *, expected=None):
    torch.manual_seed(0)
    patches = torch.randn(GROUPS, GRID * GRID, 8)
    cls = torch.randn(GROUPS, 1, 8)
    reduced = reduce_cells(
        patches,
        plan,
        num_groups=GROUPS,
        grid_side=GRID,
        cell_side=CELL_SIDE,
        expected=expected,
        cls_tokens=cell_view_cls(cls, num_groups=GROUPS, cell_side=CELL_SIDE),
    )
    return patches, reduced


def test_merge_cells_reproduces_upstream_pixel_shuffle():
    """The cell-aware merge matches InternVL's ``pixel_shuffle`` exactly."""

    for side, dim, batch in ((4, 3, 1), (8, 5, 2), (32, 64, 2)):
        tokens = torch.randn(batch, side * side, dim)
        upstream = InternVLModel.pixel_shuffle(
            None, tokens.reshape(batch, side, side, dim), scale_factor=0.5
        )
        upstream = upstream.reshape(batch, -1, upstream.shape[-1])
        assert torch.equal(
            upstream, merge_cells(tokens, grid_side=side)
        ), f"the cell-aware merge diverged from pixel_shuffle at side={side}"


def test_merging_after_a_selection_equals_selecting_after_a_merge():
    torch.manual_seed(0)
    side, dim, batch = 8, 5, 2
    tokens = torch.randn(batch, side * side, dim)
    dense = merge_cells(tokens, grid_side=side)
    total = (side // 2) ** 2
    for keep in (1, total // 2, total):
        cells = torch.randperm(total)[:keep].sort().values
        members = cell_members(side, CELL_SIDE).index_select(0, cells)
        pruned = tokens.index_select(1, torch.sort(members.reshape(-1)).values)
        assert torch.equal(
            merge_kept_cells(pruned, grid_side=side, cells=cells),
            dense.index_select(1, cells),
        ), f"prune-then-merge diverged from merge-then-select at keep={keep}"


def test_merge_rejects_a_token_count_that_is_not_the_cells_it_was_given():
    side = 8
    tokens = torch.randn(1, side * side, 4)
    with pytest.raises(AssertionError, match="token count"):
        merge_kept_cells(tokens, grid_side=side, cells=torch.tensor([0, 1]))


def test_the_whole_model_prunes_cells_and_an_untouched_run_is_unchanged():
    torch.manual_seed(0)
    model = _model()
    pixels = torch.randn(GROUPS, 3, IMAGE, IMAGE)
    with torch.no_grad():
        reference = model.extract_feature(pixels)
    assert reference.shape[1] == CELLS

    state = {}
    restore = install_feature_shim(model, state)
    context = EncoderPruningContext(num_groups=GROUPS)
    cleanup = install_pruning(model.vision_model, context)
    try:
        with torch.no_grad():
            inert = model.extract_feature(pixels)
        assert torch.equal(reference, inert), "an unarmed shim changed the stock path"

        # Keeping every cell exercises the hook and the shim while removing nothing.
        assert torch.equal(
            reference, _run(model, context, pixels, state, CELLS)
        ), "a no-op selection did not reproduce the unpruned features"

        pruned = _run(model, context, pixels, state, CELLS // 2)
        assert pruned.shape == (GROUPS, CELLS // 2, reference.shape[-1])
    finally:
        cleanup()
        restore()


def test_official_chat_feature_path_rebuilds_the_pruned_cell_merge():
    """The Instruct checkpoint exposes ``extract_feature``, not the HF feature API."""

    chat = _model()
    pixels = torch.randn(GROUPS, 3, IMAGE, IMAGE)
    with torch.no_grad():
        reference = chat.extract_feature(pixels)
    state = {}
    restore = install_feature_shim(chat, state)
    context = EncoderPruningContext(num_groups=GROUPS)
    state["context"] = context
    cleanup = install_pruning(chat.vision_model, context)
    try:
        context.plan = _plan(CELLS // 2)
        context.enabled = True
        with torch.no_grad():
            chat.vision_model(pixel_values=pixels)
        state.update(
            cells=context.kept_cells,
            grid_side=GRID,
            num_groups=GROUPS,
        )
        with torch.no_grad():
            output = chat.extract_feature(pixels)
        assert output.shape[:2] == (GROUPS, CELLS // 2)
        context.enabled = False
        with torch.no_grad():
            assert torch.equal(chat.extract_feature(pixels), reference)
    finally:
        cleanup()
        restore()


def test_placeholders_shrink_to_the_surviving_cells():
    kept = [[torch.tensor([0, 1]), torch.tensor([2, 3])]]
    inputs = {
        "input_ids": torch.tensor(
            [[1, 2, *([IMAGE_TOKEN] * CELLS), 3, *([IMAGE_TOKEN] * CELLS), 4]]
        ),
        "attention_mask": torch.ones(1, 2 * CELLS + 4, dtype=torch.long),
    }
    out = shrink_image_placeholders(
        inputs, kept, image_token_id=IMAGE_TOKEN, pad_token_id=0
    )
    ids = out["input_ids"][0]
    assert int((ids == IMAGE_TOKEN).sum()) == 4, "wrong number of placeholders left"
    assert ids.tolist() == [
        1,
        2,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        3,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        4,
    ]
    assert int(out["attention_mask"].sum()) == ids.numel()


def test_geometry_reads_a_list_valued_image_size():
    """Checkpoints may store ``image_size`` and ``patch_size`` as ``[h, w]`` pairs."""

    backend = InternVlBackend.__new__(InternVlBackend)
    assert backend._square([448, 448], "image_size") == 448
    assert backend._square(14, "patch_size") == 14
    with pytest.raises(ValueError, match="square grid"):
        backend._square([448, 336], "image_size")


def test_backend_rejects_unsupported_pruning_contracts_up_front():
    backend = InternVlBackend.__new__(InternVlBackend)
    hf_model = _model()
    backend.model = SimpleNamespace(
        config=hf_model.config,
        downsample_ratio=0.5,
        num_image_token=CELLS,
        select_layer=-1,
    )
    backend.validate_pruning_contract()

    backend.model.downsample_ratio = 0.25
    with pytest.raises(ValueError, match="downsample_ratio=0.5"):
        backend.validate_pruning_contract()
    backend.model.downsample_ratio = 0.5

    backend.model.num_image_token = CELLS + 1
    with pytest.raises(ValueError, match="image-token count"):
        backend.validate_pruning_contract()
    backend.model.num_image_token = CELLS

    backend.model.select_layer = [-1]
    with pytest.raises(ValueError, match="one integer"):
        backend.validate_pruning_contract()


def test_backend_builds_the_checkpoint_chat_prompt():
    class Template:
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        system_message = "template"

        def __init__(self):
            self.messages = []

        def copy(self):
            return Template()

        def append_message(self, role, message):
            self.messages.append((role, message))

        def get_prompt(self):
            return self

    backend = InternVlBackend.__new__(InternVlBackend)
    backend.model = SimpleNamespace(conv_template=Template(), system_message="checkpoint")
    prompt = backend._prompt_from_messages(
        [{"role": "user", "content": [{"type": "text", "text": "question"}]}]
    )

    assert prompt.system_message == "checkpoint"
    assert prompt.messages == [
        (Template.roles[0], "<video>\nquestion"),
        (Template.roles[1], None),
    ]
    assert backend.model.conv_template.messages == []
    assert backend.decoding_parameters() == {
        "checkpoint_variant": "instruct",
        "enable_thinking": False,
    }


@pytest.mark.usefixtures("first_k_folder")
@pytest.mark.parametrize("i_mode,p_mode", [("folder", "hevc"), ("folder", "folder")])
def test_a_merged_anchor_round_trips_through_the_cell_merge(i_mode, p_mode):
    """Merged cells survive being laid into member slots and merged again."""

    keep = CELLS // 2
    plan = LayerPruningPlan(
        BOUNDARY,
        _config("local", i_mode, p_mode, keep),
        None if p_mode == "folder" else _signal(keep),
    )
    patches, reduced = _reduce(plan)
    assert {int(cells.shape[0]) for row in reduced for cells, _ in row} == {
        keep
    }, "partitions came out ragged; a hook cannot carry that"

    laid = lay_down_cells(
        patches,
        reduced,
        num_groups=GROUPS,
        grid_side=GRID,
        cell_side=CELL_SIDE,
        has_cls=False,
    )
    for group in range(GROUPS):
        cells, slots = reduced[0][group]
        remerged = merge_kept_cells(
            laid[group : group + 1], grid_side=GRID, cells=slots
        )
        assert torch.equal(
            remerged[0], cells
        ), f"partition {group} did not survive lay-down and re-merge"


@pytest.mark.usefixtures("first_k_folder")
def test_a_merged_partition_takes_the_first_slots():
    """Merged cells have no original position, so they take the first slots."""

    keep = CELLS // 2
    plan = LayerPruningPlan(
        BOUNDARY, _config("local", "folder", "hevc", keep), _signal(keep)
    )
    _, reduced = _reduce(plan)
    predicted = predict_cells(
        plan, batch_size=1, num_groups=GROUPS, grid_side=GRID, cell_side=CELL_SIDE
    )

    anchor_slots = reduced[0][0][1]
    assert torch.equal(
        anchor_slots, torch.arange(keep)
    ), "the folded anchor should occupy the first k slots"
    other_slots = reduced[0][1][1]
    assert torch.equal(
        other_slots, torch.arange(keep)
    ), "a selected partition keeps its own cells' slots"
    for group in range(GROUPS):
        assert torch.equal(
            torch.as_tensor(predicted[0][group]), reduced[0][group][1]
        ), f"the pre-tower prediction disagreed with the tower at group {group}"


@pytest.mark.usefixtures("first_k_folder")
def test_the_tower_refuses_a_prediction_it_did_not_produce():
    """Slots or topology that differ from the prompt's prediction raise."""

    keep = CELLS // 2
    plan = LayerPruningPlan(
        BOUNDARY, _config("local", "folder", "hevc", keep), _signal(keep)
    )
    wrong = [[torch.arange(keep) + 1 for _ in range(GROUPS)]]
    with pytest.raises(AssertionError, match="different slots"):
        _reduce(plan, expected=wrong)

    missing_group = [[torch.arange(keep)]]
    with pytest.raises(AssertionError, match="topology"):
        _reduce(plan, expected=missing_group)


@pytest.mark.usefixtures("first_k_folder")
@pytest.mark.parametrize("i_mode,p_mode", [("folder", "hevc"), ("folder", "folder")])
def test_the_whole_model_runs_a_merging_mode(i_mode, p_mode):
    """Tower, feature shim and prompt agree when the anchor is merged."""

    torch.manual_seed(0)
    model = _model()
    pixels = torch.randn(GROUPS, 3, IMAGE, IMAGE)
    with torch.no_grad():
        reference = model.extract_feature(pixels)

    keep = CELLS // 2
    state = {}
    restore = install_feature_shim(model, state)
    context = EncoderPruningContext(num_groups=GROUPS)
    cleanup = install_pruning(model.vision_model, context)
    try:
        context.plan = LayerPruningPlan(
            BOUNDARY,
            _config("local", i_mode, p_mode, keep),
            None if p_mode == "folder" else _signal(keep),
        )
        context.enabled = True
        # The prompt is shrunk from the plan before the tower runs and checks it.
        context.expected_group_cells = predict_cells(
            context.plan,
            batch_size=1,
            num_groups=GROUPS,
            grid_side=GRID,
            cell_side=CELL_SIDE,
        )
        with torch.no_grad():
            model.vision_model(pixel_values=pixels)
        state.update(cells=context.kept_cells, grid_side=GRID, num_groups=GROUPS)
        with torch.no_grad():
            pruned = model.extract_feature(pixels)
        assert pruned.shape == (GROUPS, keep, reference.shape[-1])
        assert torch.isfinite(pruned).all(), "merging produced non-finite features"
    finally:
        cleanup()
        restore()


def test_a_preserved_anchor_keeps_its_frame_while_the_others_shrink():
    """Ragged groups run through each block one at a time, never packed."""

    torch.manual_seed(0)
    model = _model()
    pixels = torch.randn(GROUPS, 3, IMAGE, IMAGE)

    plan, widths = _ragged_plan(CELLS + (GROUPS - 1) * (CELLS // 2))
    assert len(set(widths)) > 1, "this plan is not ragged; the test proves nothing"

    state = {}
    restore = install_feature_shim(model, state)
    context = EncoderPruningContext(num_groups=GROUPS)
    cleanup = install_pruning(model.vision_model, context)

    seen = set()
    handles = [
        block.register_forward_pre_hook(
            lambda module, args, kwargs: seen.add(
                (args[0] if args else kwargs["hidden_states"]).shape[1]
            ),
            with_kwargs=True,
        )
        for block in tower_blocks(model.vision_model)
    ]
    try:
        context.plan = plan
        context.enabled = True
        context.expected_group_cells = predict_cells(
            plan, batch_size=1, num_groups=GROUPS, grid_side=GRID, cell_side=CELL_SIDE
        )
        with torch.no_grad():
            model.vision_model(pixel_values=pixels)
        state.update(cells=context.kept_cells, grid_side=GRID, num_groups=GROUPS)
        with torch.no_grad():
            pruned = model.extract_feature(pixels)
    finally:
        for handle in handles:
            handle.remove()
        cleanup()
        restore()

    assert pruned.shape == (
        1,
        sum(widths),
        32,
    ), "a ragged run packs one sequence per sample"
    packed = sum(1 + width * CELL_TOKENS for width in widths)
    assert (
        packed not in seen
    ), f"a block was handed the packed sequence ({packed} tokens): {sorted(seen)}"


def test_a_ragged_no_op_still_reproduces_the_unpruned_features():
    """A ragged plan that removes nothing reproduces the unpruned features."""

    torch.manual_seed(0)
    model = _model()
    pixels = torch.randn(GROUPS, 3, IMAGE, IMAGE)
    with torch.no_grad():
        reference = model.extract_feature(pixels)

    plan, widths = _ragged_plan(CELLS * GROUPS)
    assert widths == [CELLS] * GROUPS, "a full budget keeps every cell"

    state = {}
    restore = install_feature_shim(model, state)
    context = EncoderPruningContext(num_groups=GROUPS)
    cleanup = install_pruning(model.vision_model, context)
    try:
        context.plan = plan
        context.enabled = True
        context.expected_group_cells = predict_cells(
            plan, batch_size=1, num_groups=GROUPS, grid_side=GRID, cell_side=CELL_SIDE
        )
        with torch.no_grad():
            model.vision_model(pixel_values=pixels)
        state.update(cells=context.kept_cells, grid_side=GRID, num_groups=GROUPS)
        with torch.no_grad():
            kept = model.extract_feature(pixels)
    finally:
        cleanup()
        restore()

    assert kept.shape == reference.shape
    assert torch.equal(
        kept, reference
    ), f"max|delta|={float((kept - reference).abs().max()):.3e}"


def test_group_feature_helpers_reject_layout_drift():
    cells = [[torch.arange(2), torch.arange(1)]]
    with pytest.raises(AssertionError, match="rectangular group widths"):
        split_group_features(
            torch.randn(2, 9, 4), cells, num_groups=2, cell_side=2, has_cls=True
        )
    with pytest.raises(AssertionError, match="group topology"):
        split_group_features(
            torch.randn(1, 9, 4),
            [[torch.arange(2)]],
            num_groups=2,
            cell_side=2,
            has_cls=True,
        )


def test_placeholder_shrink_rejects_missing_or_short_runs():
    kept = [[torch.arange(2), torch.arange(2)]]
    with pytest.raises(AssertionError, match="placeholder topology"):
        shrink_image_placeholders(
            {"input_ids": torch.tensor([[1, 2, 3]])},
            kept,
            image_token_id=IMAGE_TOKEN,
            pad_token_id=0,
        )
    with pytest.raises(AssertionError, match="shorter than its cell budget"):
        shrink_image_placeholders(
            {"input_ids": torch.tensor([[IMAGE_TOKEN, 1, IMAGE_TOKEN]])},
            kept,
            image_token_id=IMAGE_TOKEN,
            pad_token_id=0,
        )


def test_backend_rejects_pixel_rows_that_do_not_form_complete_videos():
    backend = InternVlBackend.__new__(InternVlBackend)
    session = SimpleNamespace(
        pruning_active=True,
        context=SimpleNamespace(num_groups=GROUPS, plan=_plan(1)),
    )
    with pytest.raises(ValueError, match="rows divisible by G"):
        backend.adapt_lm_inputs(
            {
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "pixel_values": torch.randn(GROUPS + 1, 3, IMAGE, IMAGE),
            },
            session,
            effective_keep_total=2,
        )


@pytest.mark.parametrize("boundary", [0, BOUNDARY])
def test_global_folder_merges_all_frame_cells_once(boundary, monkeypatch):
    model = _model()
    calls = []

    class Folder:
        def reduce(self, cls, tokens, keep):
            calls.append(tokens.shape)
            return tokens.mean(1, keepdim=True).expand(-1, keep, -1).clone()

    monkeypatch.setattr("token_pruner.engine.folder_wrapper", Folder)
    plan = LayerPruningPlan(boundary, PruningConfig(scope="global", reducer="global_folder",
        keep_per_partition=CELLS // 2, keep_total=GROUPS * CELLS // 2))
    context = EncoderPruningContext(num_groups=GROUPS, plan=plan)
    kept = predict_cells(plan, batch_size=1, num_groups=GROUPS,
        grid_side=GRID, cell_side=CELL_SIDE)
    context.expected_group_cells = kept
    state = dict(context=context, cells=kept, grid_side=GRID, num_groups=GROUPS)
    cleanup = install_pruning(model.vision_model, context)
    restore = install_feature_shim(model, state)
    try:
        with torch.no_grad():
            result = model.extract_feature(torch.randn(GROUPS, 3, IMAGE, IMAGE))
        assert calls == [torch.Size([1, GROUPS * CELLS, 32 * CELL_TOKENS])]
        assert result.shape == (GROUPS, CELLS // 2, 32)
        assert torch.isfinite(result).all()
    finally:
        cleanup()
        restore()


def test_remote_chat_class_is_post_initialised_and_follows_the_requested_dtype(
    tmp_path, monkeypatch
):
    from transformers import PretrainedConfig, PreTrainedModel, Qwen3Config, Qwen3ForCausalLM

    from token_pruner import run_internvl

    class InternVLChatModel(PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.vision_model = torch.nn.Linear(4, 4)
            self.language_model = Qwen3ForCausalLM(config.llm_config)

    monkeypatch.setattr(run_internvl, "AutoConfig", SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(
            auto_map={"AutoModel": "modeling_internvl_chat.InternVLChatModel"}
        )
    ))
    monkeypatch.setattr(
        run_internvl, "get_class_from_dynamic_module",
        lambda *args, **kwargs: InternVLChatModel,
    )
    config = PretrainedConfig(llm_config=Qwen3Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=200, dtype="bfloat16",
    ))
    chat_class = run_internvl.internvl_chat_class("checkpoint", local_files_only=True)
    chat_class(config).save_pretrained(tmp_path)

    with pytest.raises(AttributeError, match="all_tied_weights_keys"):
        InternVLChatModel.from_pretrained(tmp_path, config=config)
    loaded = chat_class.from_pretrained(tmp_path, config=config, dtype=torch.float32)

    assert type(loaded).__name__ == "InternVLChatModel"
    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float32}
