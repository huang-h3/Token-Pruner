"""LLaVA-NeXT-Video pruning: pooling inside a cell survives a selection."""

from token_pruner.run_llava_next_video import LLAVA_NEXT_VIDEO_LEGAL_MODES
import pytest
import torch

pytest.importorskip("transformers.models.llava_next_video.modeling_llava_next_video")
from transformers import CLIPVisionConfig, Qwen2Config
from transformers.models.llava_next_video.configuration_llava_next_video import (
    LlavaNextVideoConfig,
)
from transformers.models.llava_next_video.modeling_llava_next_video import (
    LlavaNextVideoForConditionalGeneration, LlavaNextVideoPooler,
)

from token_pruner.cells import cell_members
from token_pruner.run_llava_next_video import (
    install_feature_shim, install_pruning, pool_cells,
)
from token_pruner.layer_boundary import (
    EncoderPruningContext,
)
from token_pruner.selection import LayerPruningPlan
from token_pruner.tokens import (
    PruningConfig, PruningScope, PruningSignal, SignalSource, TokenReducer,
)

IMAGE, PATCH, STRIDE, BOUNDARY = 64, 16, 2, 2
GRID = IMAGE // PATCH
CELLS = (GRID // STRIDE) ** 2
FRAMES = 2
VIDEO_TOKEN = 150


def _pooler(mode, hidden):
    config = LlavaNextVideoConfig(
        vision_config=CLIPVisionConfig(
            hidden_size=hidden, image_size=IMAGE, patch_size=PATCH,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=16),
        spatial_pool_mode=mode, spatial_pool_stride=STRIDE)
    return LlavaNextVideoPooler(config)


@pytest.mark.parametrize("mode", ("average", "max"))
def test_pool_cells_reproduces_the_upstream_pooler(mode):
    """The integration rests on this being the same bytes."""

    hidden = 8
    pooler = _pooler(mode, hidden)
    for side in (4, 8, 16):
        tokens = torch.randn(3, side * side, hidden)
        assert torch.equal(pooler(tokens), pool_cells(tokens, pooler, grid_side=side)), (
            f"the cell-aware pool diverged from {mode} pooling at side={side}")


def test_pooling_after_a_selection_equals_selecting_after_a_pool():
    hidden, side = 8, 8
    pooler = _pooler("average", hidden)
    tokens = torch.randn(2, side * side, hidden)
    dense = pool_cells(tokens, pooler, grid_side=side)
    total = (side // STRIDE) ** 2
    for keep in (1, total // 2, total):
        cells = torch.randperm(total)[:keep].sort().values
        members = cell_members(side, STRIDE).index_select(0, cells)
        pruned = tokens.index_select(1, torch.sort(members.reshape(-1)).values)
        assert torch.equal(
            pool_cells(pruned, pooler, grid_side=side, cells=cells, kept=True),
            dense.index_select(1, cells),
        ), f"prune-then-pool diverged from pool-then-select at keep={keep}"


def test_a_conv_pooler_is_refused_rather_than_approximated():
    """A learned Conv2d cell reduction is not what mean/amax computes."""

    with pytest.raises(RuntimeError, match="Conv2d"):
        pool_cells(torch.randn(1, 16, 8), _pooler("conv", 8), grid_side=4)


def _model():
    vision = CLIPVisionConfig(
        hidden_size=32, image_size=IMAGE, patch_size=PATCH, num_hidden_layers=4,
        num_attention_heads=4, intermediate_size=64)
    text = Qwen2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=200)
    config = LlavaNextVideoConfig(
        vision_config=vision, text_config=text, video_token_index=VIDEO_TOKEN,
        spatial_pool_mode="average", spatial_pool_stride=STRIDE,
        vision_feature_layer=-2, vision_feature_select_strategy="default")
    return LlavaNextVideoForConditionalGeneration(config).eval()


def _plan(keep):
    config = PruningConfig(
        scope=PruningScope.LOCAL, reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE, keep_per_partition=keep,
        keep_total=keep * FRAMES, signal_source=SignalSource.HEVC)
    signal = PruningSignal(
        visible_indices=[torch.tensor(
            [g * CELLS + c for g in range(FRAMES) for c in range(keep)])],
        anchor_partitions=torch.tensor([[False] * FRAMES]))
    return LayerPruningPlan(BOUNDARY, config, signal)


def test_the_whole_model_prunes_cells_and_an_untouched_run_is_unchanged():
    torch.manual_seed(0)
    model = _model()
    inner = model.model
    pixels = torch.randn(1, FRAMES, 3, IMAGE, IMAGE)
    with torch.no_grad():
        reference = inner.get_video_features(pixels).pooler_output[0]
    assert reference.shape[1] == CELLS

    state = {}
    restore = install_feature_shim(model, state)
    context = EncoderPruningContext(num_groups=FRAMES)
    cleanup = install_pruning(
        inner.vision_tower, context, cell_side=STRIDE)
    try:
        with torch.no_grad():
            assert torch.equal(reference, inner.get_video_features(pixels).pooler_output[0]), (
                "an unarmed shim changed the stock path")

        def run(keep):
            context.plan = _plan(keep)
            context.enabled = True
            with torch.no_grad():
                inner.vision_tower(
                    pixels.reshape(FRAMES, 3, IMAGE, IMAGE), output_hidden_states=True)
            state.update(
                cells=context.kept_cells, grid_side=GRID, num_groups=FRAMES)
            with torch.no_grad():
                return inner.get_video_features(pixels).pooler_output[0]

        assert torch.equal(reference, run(CELLS)), (
            "a no-op selection did not reproduce the unpruned features")
        assert run(CELLS // 2).shape == (FRAMES, CELLS // 2, reference.shape[-1])
    finally:
        cleanup()
        restore()


class _FirstKFolder:
    """Keep the first ``keep`` tokens in place of the external FOLDER merge."""

    def reduce(self, _cls_token, frame_tokens, keep):
        return frame_tokens[:, :keep]


def _merging_config(i_mode, p_mode, keep):
    from token_pruner.selection import build_model_pruning_config

    return build_model_pruning_config(
        model_family="llava_next_video", stage="input", prune_mode="local",
        legal_modes=LLAVA_NEXT_VIDEO_LEGAL_MODES, uniform_partitions=True,
        i_mode=i_mode, p_mode=p_mode, keep_per_partition=keep,
        keep_total=keep * FRAMES, min_per_partition=1)


def _merging_signal(keep, anchor=0):
    return PruningSignal(
        visible_indices=[torch.tensor(
            [g * CELLS + c for g in range(FRAMES) if g != anchor
             for c in range(keep)])],
        anchor_partitions=torch.tensor([[g == anchor for g in range(FRAMES)]]))


def test_a_merged_cell_survives_the_pooling_merge(monkeypatch):
    """The pooled result is the mean of the merged cell's own members."""

    monkeypatch.setattr("token_pruner.engine.folder_wrapper", _FirstKFolder)
    from token_pruner.run_llava_next_video import (
        cell_view_cls, lay_down_cells, reduce_cells,
    )

    torch.manual_seed(0)
    hidden, keep = 8, CELLS // 2
    pooler = _pooler("average", hidden)
    patches = torch.randn(FRAMES, GRID * GRID, hidden)
    cls = torch.randn(FRAMES, 1, hidden)
    plan = LayerPruningPlan(
        BOUNDARY, _merging_config("folder", "hevc", keep), _merging_signal(keep))

    reduced = reduce_cells(
        patches, plan, num_groups=FRAMES, grid_side=GRID, cell_side=STRIDE,
        cls_tokens=cell_view_cls(cls, num_groups=FRAMES, cell_side=STRIDE))
    assert {int(c.shape[0]) for row in reduced for c, _ in row} == {keep}

    laid = lay_down_cells(
        patches, reduced, num_groups=FRAMES, grid_side=GRID,
        cell_side=STRIDE, has_cls=False)
    for group in range(FRAMES):
        cells, slots = reduced[0][group]
        pooled = pool_cells(
            laid[group : group + 1], pooler, grid_side=GRID,
            cells=slots, kept=True)
        expected = cells.reshape(keep, STRIDE * STRIDE, hidden).mean(dim=1)
        assert torch.allclose(pooled[0], expected, atol=1e-6), (
            f"partition {group} pooled a different cell's members")


def test_the_whole_model_runs_a_merging_mode(monkeypatch):
    from token_pruner.run_llava_next_video import predict_cells

    monkeypatch.setattr("token_pruner.engine.folder_wrapper", _FirstKFolder)
    torch.manual_seed(0)
    model = _model()
    inner = model.model
    pixels = torch.randn(1, FRAMES, 3, IMAGE, IMAGE)
    with torch.no_grad():
        reference = inner.get_video_features(pixels).pooler_output[0]

    keep = CELLS // 2
    state = {}
    restore = install_feature_shim(model, state)
    context = EncoderPruningContext(num_groups=FRAMES)
    cleanup = install_pruning(
        inner.vision_tower, context, cell_side=STRIDE)
    try:
        context.plan = LayerPruningPlan(
            BOUNDARY, _merging_config("folder", "hevc", keep), _merging_signal(keep))
        context.enabled = True
        context.expected_group_cells = predict_cells(
            context.plan, batch_size=1, num_groups=FRAMES,
            grid_side=GRID, cell_side=STRIDE)
        with torch.no_grad():
            inner.vision_tower(
                pixels.reshape(FRAMES, 3, IMAGE, IMAGE), output_hidden_states=True)
        state.update(cells=context.kept_cells, grid_side=GRID, num_groups=FRAMES)
        with torch.no_grad():
            pruned = inner.get_video_features(pixels).pooler_output[0]
        assert pruned.shape == (FRAMES, keep, reference.shape[-1])
        assert torch.isfinite(pruned).all()
    finally:
        cleanup()
        restore()


def _ragged_batch_plan(batch_size):
    """Preserve alternating anchors and keep half of every other frame."""

    from token_pruner.selection import build_model_pruning_config

    keep_total = CELLS + CELLS // 2
    config = build_model_pruning_config(
        model_family="llava_next_video", stage="input", prune_mode="local",
        legal_modes=LLAVA_NEXT_VIDEO_LEGAL_MODES, uniform_partitions=False,
        i_mode="preserve", p_mode="hevc",
        keep_per_partition=keep_total // FRAMES, keep_total=keep_total,
        min_per_partition=1,
    )
    anchors, visible = [], []
    for sample in range(batch_size):
        anchor = sample % FRAMES
        anchors.append([group == anchor for group in range(FRAMES)])
        visible.append(torch.cat([
            torch.arange(CELLS // 2) + group * CELLS
            for group in range(FRAMES) if group != anchor
        ]))
    signal = PruningSignal(
        visible_indices=visible, anchor_partitions=torch.tensor(anchors))
    return LayerPruningPlan(BOUNDARY, config, signal)


def test_ragged_batch_returns_one_feature_tensor_per_video_and_runs_the_model():
    """Packed ragged rows must not be split by the frame count along B."""

    from token_pruner.run_llava_next_video import predict_cells

    torch.manual_seed(0)
    batch = 2
    model = _model()
    inner = model.model
    plan = _ragged_batch_plan(batch)
    kept = predict_cells(
        plan, batch_size=batch, num_groups=FRAMES,
        grid_side=GRID, cell_side=STRIDE)
    assert [[len(cells) for cells in row] for row in kept] == [[4, 2], [2, 4]]

    state = dict(cells=kept, grid_side=GRID, num_groups=FRAMES)
    restore = install_feature_shim(model, state)
    context = EncoderPruningContext(
        enabled=True, num_groups=FRAMES, plan=plan,
        expected_group_cells=kept)
    cleanup = install_pruning(
        inner.vision_tower, context, cell_side=STRIDE)
    pixels = torch.randn(batch, FRAMES, 3, IMAGE, IMAGE)
    try:
        with torch.no_grad():
            features = inner.get_video_features(pixels).pooler_output
        assert len(features) == batch
        assert [tuple(feature.shape[:2]) for feature in features] == [
            (1, 6), (1, 6)]

        input_ids = torch.full(
            (batch, 6), VIDEO_TOKEN, dtype=torch.long)
        with torch.no_grad():
            output = inner(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                pixel_values_videos=pixels)
        assert output.last_hidden_state.shape[:2] == (batch, 6)
        assert output.video_hidden_states.shape[0] == batch * 6
    finally:
        cleanup()
        restore()


def test_backend_resizes_one_video_placeholder_run_per_sample():
    """The batch total must never be duplicated into every prompt row."""

    from types import SimpleNamespace

    from token_pruner.run_llava_next_video import (
        LlavaNextVideoBackend,
    )

    backend = LlavaNextVideoBackend.__new__(LlavaNextVideoBackend)
    backend.model = SimpleNamespace(config=SimpleNamespace(
        video_token_index=VIDEO_TOKEN,
        spatial_pool_stride=STRIDE,
        vision_config=SimpleNamespace(image_size=IMAGE, patch_size=PATCH),
    ))
    backend.processor = SimpleNamespace(tokenizer=SimpleNamespace(
        pad_token_id=0, padding_side="left"))
    backend._shim_state = {}

    batch = 2
    plan = _ragged_batch_plan(batch)
    session = SimpleNamespace(
        pruning_active=True,
        context=SimpleNamespace(
            num_groups=FRAMES, plan=plan, expected_group_cells=None),
    )
    inputs = {
        "input_ids": torch.full(
            (batch, CELLS * FRAMES), VIDEO_TOKEN, dtype=torch.long),
        "attention_mask": torch.ones(
            batch, CELLS * FRAMES, dtype=torch.long),
        "pixel_values_videos": torch.randn(
            batch, FRAMES, 3, IMAGE, IMAGE),
    }
    resized = backend.adapt_lm_inputs(inputs, session, effective_keep_total=6)
    assert (resized["input_ids"] == VIDEO_TOKEN).sum(dim=1).tolist() == [6, 6]
    assert resized["attention_mask"].sum(dim=1).tolist() == [6, 6]


@pytest.mark.parametrize("boundary", [0, BOUNDARY])
def test_global_folder_merges_cells_before_pooling(boundary, monkeypatch):
    from token_pruner.cells import predict_cells

    calls = []

    class Folder:
        def reduce(self, cls, tokens, keep):
            calls.append(tokens.shape)
            return tokens.mean(1, keepdim=True).expand(-1, keep, -1).clone()

    monkeypatch.setattr("token_pruner.engine.folder_wrapper", Folder)
    model = _model()
    plan = LayerPruningPlan(boundary, PruningConfig(scope="global", reducer="global_folder",
        keep_per_partition=CELLS // 2, keep_total=FRAMES * CELLS // 2))
    kept = predict_cells(plan, batch_size=1, num_groups=FRAMES, grid_side=GRID, cell_side=STRIDE)
    context = EncoderPruningContext(num_groups=FRAMES, plan=plan, expected_group_cells=kept)
    cleanup = install_pruning(model.model.vision_tower, context, cell_side=STRIDE)
    restore = install_feature_shim(model, dict(cells=kept, grid_side=GRID, num_groups=FRAMES))
    try:
        with torch.no_grad():
            output = model.model.get_video_features(torch.randn(1, FRAMES, 3, IMAGE, IMAGE)).pooler_output[0]
        assert calls == [torch.Size([1, FRAMES * CELLS, 32 * STRIDE ** 2])]
        assert output.shape == (FRAMES, CELLS // 2, 32)
        assert torch.isfinite(output).all()
    finally:
        cleanup()
        restore()


@pytest.mark.parametrize("boundary", [0, BOUNDARY])
def test_pruned_video_generates_with_cache(boundary):
    from token_pruner.cells import predict_cells

    model = _model()
    plan = _plan(CELLS // 2)
    plan = LayerPruningPlan(boundary, plan.config, plan.signal)
    cells = predict_cells(
        plan, batch_size=1, num_groups=FRAMES, grid_side=GRID, cell_side=STRIDE
    )
    context = EncoderPruningContext(num_groups=FRAMES, plan=plan)
    context.expected_group_cells = cells
    state = dict(cells=cells, grid_side=GRID, num_groups=FRAMES)
    cleanup = install_pruning(model.model.vision_tower, context, cell_side=STRIDE)
    restore = install_feature_shim(model, state)
    ids = torch.tensor([[1] + [VIDEO_TOKEN] * sum(len(row) for row in cells[0]) + [2]])
    try:
        with torch.no_grad():
            result = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values_videos=torch.randn(1, FRAMES, 3, IMAGE, IMAGE),
                max_new_tokens=3, min_new_tokens=3, do_sample=False,
                eos_token_id=None, pad_token_id=0,
            )
        assert result.shape[1] == ids.shape[1] + 3
    finally:
        cleanup()
        restore()
