"""Contract tests for the Qwen3-VL pruning integration."""

from token_pruner.run_qwen3_vl import QWEN3_VL_LEGAL_MODES
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from token_pruner.run_qwen3_vl import (
    kept_cell_indices,
    prune_flat_visual_tokens,
)
from token_pruner.run_qwen3_vl import (
    Qwen3VlHfBackend,
    resize_video_placeholders,
)
from token_pruner.run_qwen3_vl import Qwen3VlPruning
from token_pruner.selection import LayerPruningPlan
from token_pruner.tokens import PruningConfig, PruningScope, SignalSource, TokenReducer
from token_pruner.selection import validate_pruning_modes
from token_pruner.selectors import TubeletSelector, spatial_patch_scores
from token_pruner.tokens import PruningSignal

VIDEO_TOKEN = 900


def validate_qwen_modes(_family, stage, scope, i_mode, p_mode):
    return validate_pruning_modes(
        QWEN3_VL_LEGAL_MODES,
        "qwen3_vl",
        stage,
        scope,
        i_mode,
        p_mode,
    )


def resolve_qwen_partitions(vision_encoder, num_frames):
    backend = Qwen3VlHfBackend.__new__(Qwen3VlHfBackend)
    backend.temporal_patch_size = int(vision_encoder.config.temporal_patch_size)
    return backend.partition_count(num_frames)


def resolve_qwen_stage(prune_layer, *, stage, num_layers, boundaries):
    return Qwen3VlHfBackend.stage_layer(
        None,
        prune_layer,
        stage=stage,
        num_layers=num_layers,
        boundaries=boundaries,
    )


def gather_plan(
    *,
    cells_per_group,
    keep_per_partition,
    num_groups,
    anchors=None,
    layer=0,
    batch_size=1,
    strides=None,
):
    """A bound gather plan; ``strides`` gives each sample a different pick."""

    strides = strides or [2] * batch_size
    selection = torch.stack(
        [
            torch.cat(
                [
                    torch.arange(keep_per_partition, dtype=torch.long) * strides[b]
                    + group * cells_per_group
                    for group in range(num_groups)
                ]
            )
            for b in range(batch_size)
        ]
    )
    anchor_partitions = torch.zeros(batch_size, num_groups, dtype=torch.bool)
    for index in anchors or ():
        anchor_partitions[:, index] = True
    keep_total = keep_per_partition * num_groups + len(anchors or ()) * (
        cells_per_group - keep_per_partition
    )
    config = PruningConfig(
        scope=PruningScope.LOCAL,
        reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE,
        keep_per_partition=keep_per_partition,
        keep_total=keep_total,
        signal_source=SignalSource.RANDOM,
    )
    signal = PruningSignal(visible_indices=selection, anchor_partitions=anchor_partitions)
    return LayerPruningPlan(layer, config).bind(signal)


def test_kept_cells_report_the_selection_and_preserve_anchor_groups():
    plan = gather_plan(cells_per_group=16, keep_per_partition=4, num_groups=3, anchors=[0])
    cells = kept_cell_indices(plan, batch_size=1, num_groups=3, cells_per_group=16)

    assert cells[0][0].tolist() == list(range(16))
    assert cells[0][1].tolist() == [0, 2, 4, 6]
    assert cells[0][2].tolist() == [0, 2, 4, 6]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_per_group_cells_device_cuda_matches_tower_when_signal_is_on_gpu():
    """Prompt-side cell selection must use the signal's tower device."""

    device = torch.device("cuda")
    cells_per_group, num_groups = 4, 3
    config = PruningConfig(
        scope=PruningScope.LOCAL,
        reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE,
        keep_per_partition=1,
        keep_total=cells_per_group + (num_groups - 1),
        signal_source=SignalSource.RANDOM,
    )
    signal = PruningSignal(
        # Group 1 keeps cell 0 (flat index 4); group 2 keeps cell 1 (9).
        visible_indices=torch.tensor([[4, 9]], device=device),
        anchor_partitions=torch.tensor([[True, False, False]], device=device),
    )
    plan = LayerPruningPlan(0, config).bind(signal)

    backend = Qwen3VlHfBackend.__new__(Qwen3VlHfBackend)
    backend.device = device
    session = SimpleNamespace(
        context=SimpleNamespace(plan=plan),
        total_patches=cells_per_group,
    )

    prompt_cells = backend.per_group_cells(session, num_groups)
    hidden = torch.arange(
        num_groups * cells_per_group, device=device, dtype=torch.float32
    ).unsqueeze(1)
    positions = torch.arange(
        num_groups * cells_per_group, device=device, dtype=torch.float32
    ).unsqueeze(1)
    _, _, _, tower_cells = prune_flat_visual_tokens(
        hidden,
        (positions, positions.clone()),
        batch_size=1,
        num_groups=num_groups,
        tokens_per_group=cells_per_group,
        merge_unit=1,
        plan=plan,
    )

    assert [row.tolist() for row in prompt_cells[0]] == [[0, 1, 2, 3], [0], [1]]
    assert [row.tolist() for row in tower_cells[0]] == [
        row.tolist() for row in prompt_cells[0]
    ]
    assert all(
        row.device.type == "cuda"
        and row.device == signal.visible_indices.device
        for row in prompt_cells[0]
    )

    # A CPU partition topology must normalize the visible CUDA indices.
    from token_pruner.engine import selected_partition_indices

    cpu_partition_ids = torch.arange(num_groups).repeat_interleave(cells_per_group)
    selected = selected_partition_indices(
        signal,
        config,
        partition_ids=cpu_partition_ids,
        partition_idx=1,
        device=None,
    )
    assert selected.device == cpu_partition_ids.device
    assert selected.tolist() == [0]


def test_pruning_keeps_whole_merge_cells_and_their_position_rows():
    merge_unit, cells_per_group, num_groups = 4, 16, 2
    tokens_per_group = cells_per_group * merge_unit
    seq_len = tokens_per_group * num_groups
    hidden = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    cos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    plan = gather_plan(
        cells_per_group=cells_per_group, keep_per_partition=4, num_groups=num_groups
    )

    pruned, (kept_cos, _), cu_seqlens, cells = prune_flat_visual_tokens(
        hidden,
        (cos, cos.clone()),
        batch_size=1,
        num_groups=num_groups,
        tokens_per_group=tokens_per_group,
        merge_unit=merge_unit,
        plan=plan,
    )

    # Cells 0, 2, 4, 6 of each group, each contributing its four sub-tokens.
    expected = [
        group * tokens_per_group + cell * merge_unit + offset
        for group in range(num_groups)
        for cell in (0, 2, 4, 6)
        for offset in range(merge_unit)
    ]
    assert pruned[:, 0].tolist() == expected
    assert kept_cos[:, 0].tolist() == expected
    assert cu_seqlens.tolist() == [0, 16, 32]
    assert [cell.tolist() for cell in cells[0]] == [[0, 2, 4, 6]] * num_groups


def test_pruning_rejects_a_selection_that_disagrees_with_the_prompt():
    merge_unit, cells_per_group, num_groups = 4, 16, 2
    tokens_per_group = cells_per_group * merge_unit
    hidden = torch.zeros(tokens_per_group * num_groups, 3)
    cos = torch.zeros(tokens_per_group * num_groups, 2)
    plan = gather_plan(
        cells_per_group=cells_per_group, keep_per_partition=4, num_groups=num_groups
    )

    with pytest.raises(RuntimeError, match="placeholders were resized to"):
        prune_flat_visual_tokens(
            hidden,
            (cos, cos.clone()),
            batch_size=1,
            num_groups=num_groups,
            tokens_per_group=tokens_per_group,
            merge_unit=merge_unit,
            plan=plan,
            expected_group_cells=[[torch.tensor([1, 3, 5, 7])] * num_groups],
        )


def _prompt_row(group_cells_per_group, num_groups, pad=0):
    """``[pad] text <video xN> text <video xN> ... text``."""

    row = [1] * pad + [10]
    for _ in range(num_groups):
        row += [VIDEO_TOKEN] * group_cells_per_group + [11]
    return row


def _video_types(input_ids):
    return (input_ids == VIDEO_TOKEN).long() * 2


def test_placeholder_resize_drops_exactly_the_unselected_cells():
    row = _prompt_row(8, 2)
    inputs = {
        "input_ids": torch.tensor([row]),
        "attention_mask": torch.ones(1, len(row), dtype=torch.long),
        "mm_token_type_ids": _video_types(torch.tensor([row])),
    }
    cells = [[torch.tensor([0, 3, 5]), torch.tensor([1, 2, 7])]]

    resized, keep_masks = resize_video_placeholders(
        inputs, VIDEO_TOKEN, cells, padding_side="left"
    )

    assert resized["input_ids"][0].tolist() == _prompt_row(3, 2)
    assert torch.equal(resized["mm_token_type_ids"], _video_types(resized["input_ids"]))
    kept = keep_masks[0].nonzero().flatten().tolist()
    # Text at 0, group one starts at 1, its "<video>" run ends at 9.
    assert [index - 1 for index in kept if 1 <= index <= 8] == [0, 3, 5]
    assert [index - 10 for index in kept if 10 <= index <= 17] == [1, 2, 7]


def test_placeholder_resize_requires_one_run_per_group():
    row = _prompt_row(8, 3)
    inputs = {
        "input_ids": torch.tensor([row]),
        "attention_mask": torch.ones(1, len(row), dtype=torch.long),
        "mm_token_type_ids": _video_types(torch.tensor([row])),
    }

    with pytest.raises(RuntimeError, match="3 video placeholder runs"):
        resize_video_placeholders(inputs, VIDEO_TOKEN, [[torch.tensor([0])] * 2])


class _RecordingProcessor:
    """Capture what would be handed to the checkpoint's chat template."""

    vision_start_token = "<|vision_start|>"
    video_token = "<|video_pad|>"
    vision_end_token = "<|vision_end|>"

    def __init__(self):
        self.rendered = None

    def apply_chat_template(self, messages, **_kwargs):
        self.rendered = messages
        return "<|vision_start|><|video_pad|><|vision_end|>Q?"


def _render(messages):
    backend = Qwen3VlHfBackend.__new__(Qwen3VlHfBackend)
    backend.processor = _RecordingProcessor()
    backend._prompt_from_messages(messages)
    return backend.processor.rendered


def test_prompt_leaves_frame_markers_to_the_processor():
    backend = Qwen3VlHfBackend.__new__(Qwen3VlHfBackend)
    backend.processor = _RecordingProcessor()

    prompt = backend._prompt_from_messages(
        [{"role": "user", "content": [{"type": "text", "text": "Q?"}]}]
    )

    assert prompt == "<|video_pad|>Q?"


def test_prompt_keeps_one_video_placeholder_when_the_message_carries_one():
    # Preserve the video block emitted by ChatMessages.to_hf_messages().
    rendered = _render(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/videos/clip.mp4"},
                    {"type": "text", "text": "What happens?"},
                ],
            }
        ]
    )

    assert rendered == [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "What happens?"},
            ],
        }
    ]


def test_prompt_adds_the_placeholder_when_the_message_has_only_text():
    rendered = _render([{"role": "user", "content": [{"type": "text", "text": "Q?"}]}])

    assert rendered[0]["content"][0] == {"type": "video"}


def test_prompt_places_the_video_in_the_first_user_turn_only():
    rendered = _render(
        [
            {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/videos/a.mp4"},
                    {"type": "text", "text": "One"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Ok"}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/videos/b.mp4"},
                    {"type": "text", "text": "Two"},
                ],
            },
        ]
    )

    placeholders = [
        block
        for message in rendered
        for block in message["content"]
        if block.get("type") == "video"
    ]
    assert len(placeholders) == 1
    assert rendered[1]["content"][0] == {"type": "video"}
    assert [message["role"] for message in rendered] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_pin_frames_changes_only_spatial_geometry():
    backend = Qwen3VlHfBackend.__new__(Qwen3VlHfBackend)
    backend.frame_size = 4
    backend.temporal_patch_size = 2
    frames = np.stack(
        [np.full((2, 3, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    )

    pinned = backend._pin_frames(frames)

    assert pinned.shape == (3, 4, 4, 3)
    assert pinned[:, 0, 0, 0].tolist() == [10, 20, 30]


def test_qwen_partition_count_uses_native_temporal_patches():
    vision_encoder = SimpleNamespace(
        config=SimpleNamespace(temporal_patch_size=2)
    )

    assert resolve_qwen_partitions(vision_encoder, 64) == 32
    assert resolve_qwen_partitions(vision_encoder, 63) == 32


def test_qwen_session_budgets_model_partitions_not_source_frames(monkeypatch):
    vision_encoder = SimpleNamespace(
        config=SimpleNamespace(
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        _token_pruner_grid_side=2,
        blocks=[object(), object()],
        deepstack_visual_indexes=[1],
    )
    backend = Qwen3VlHfBackend.__new__(Qwen3VlHfBackend)
    backend.device = "cpu"
    backend.model = SimpleNamespace(model=SimpleNamespace(visual=vision_encoder))
    backend.patch_size = 2
    backend.spatial_merge_size = 2
    backend.temporal_patch_size = 2
    backend.visual_grid_side = 2
    backend.install_pruning = lambda _context: None

    session = Qwen3VlPruning.create(
        backend,
        num_frames=5,
        prune_stage="input",
        prune_mode="local",
        i_mode="random",
        p_mode="random",
        k_keep_rate=0.5,
        run_mode="pruned",
        hevc_n_parallel=1,
        hevc_dir="/tmp/qwen-pruning-test",
    )

    # Five frames -> three temporal patches of four cells: budgets are 12/6.
    assert session.context.num_groups == 3
    assert session.total_patches_clip == 12
    assert session.target_keep_total == 6
    assert session.signal_provider.num_partitions == 3
    assert session.current_keep_budgets(1) == [(6, 0, 6)]


def test_tubelet_selector_pads_only_the_last_odd_frame():
    selector = TubeletSelector(
        patch_size=2,
        tubelet_size=2,
        tubelet_budget=1,
        image_shape=(1, 3, 5, 4, 4),
        prune_mode="local",
    )
    saliency = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)

    selected = selector.tubelet_score_cal(
        spatial_patch_scores(saliency, 2),
        keep_tubelets=1,
        tubelet_size=2,
        prune_mode="local",
    )

    # Partition 0 is the implicit HEVC anchor.
    assert selected.numel() == 2
    assert selected.div(4, rounding_mode="floor").tolist() == [1, 2]


def test_placeholder_resize_pads_on_the_left_for_ragged_batches():
    """Ragged rows keep their pads at the front: decoding resumes at the end."""

    short = [10, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, 11]
    long = [10, 10, 10, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, 11]
    width = len(long)
    input_ids = torch.tensor([[0] * (width - len(short)) + short, long])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.tensor(
            [[0] * (width - len(short)) + [1] * len(short), [1] * width]
        ),
        "mm_token_type_ids": _video_types(input_ids),
    }
    cells = [[torch.tensor([0, 2])], [torch.tensor([0, 2])]]

    resized, keep_masks = resize_video_placeholders(
        inputs, VIDEO_TOKEN, cells, pad_token_id=0, padding_side="left"
    )

    rows = resized["input_ids"].tolist()
    mask = resized["attention_mask"].tolist()
    # Short row keeps 2 of its 4 placeholders -> 4 valid tokens, padded to 6.
    assert rows[0] == [0, 0, 10, VIDEO_TOKEN, VIDEO_TOKEN, 11]
    assert mask[0] == [0, 0, 1, 1, 1, 1]
    assert rows[1] == [10, 10, 10, VIDEO_TOKEN, VIDEO_TOKEN, 11]
    assert torch.equal(resized["mm_token_type_ids"], _video_types(resized["input_ids"]))
    assert mask[1] == [1, 1, 1, 1, 1, 1]
    # Masks are over the original row's valid tokens.
    assert keep_masks[0].tolist() == [True, True, False, True, False, True]


def test_last_stage_is_the_earliest_deepstack_boundary():
    boundaries = [8, 16, 24, 27]

    assert (
        resolve_qwen_stage(
            None, stage="last", num_layers=27, boundaries=boundaries
        )
        == 8
    )
    assert (
        resolve_qwen_stage(
            None, stage="input", num_layers=27, boundaries=boundaries
        )
        == 0
    )
    assert (
        resolve_qwen_stage(
            -1, stage="last", num_layers=27, boundaries=boundaries
        )
        == 27
    )


@pytest.mark.parametrize("stage", ("input", "last"))
def test_index_selecting_local_modes_are_legal(stage):
    validate_qwen_modes("qwen3_vl", stage, "local", "preserve", "hevc")
    validate_qwen_modes("qwen3_vl", stage, "local", "random", "random")
    validate_qwen_modes("qwen3_vl", stage, "local", "uniform", "uniform")


@pytest.mark.parametrize(
    "i_mode,p_mode",
    (
        ("preserve", "folder"),
        ("folder", "hevc"),
        ("folder", "folder"),
        ("preserve", "shared_folding"),
    ),
)
def test_merging_reducers_are_rejected(i_mode, p_mode):
    # This tower re-applies a rotary embedding per token, so merging has no position.
    with pytest.raises(RuntimeError, match="Unsupported qwen3vl"):
        validate_qwen_modes("qwen3_vl", "last", "local", i_mode, p_mode)


def test_batched_pruning_uses_each_samples_own_selection():
    """Two samples with different picks share one packed sequence."""

    merge_unit, cells_per_group, num_groups, batch_size = 4, 16, 2, 2
    tokens_per_group = cells_per_group * merge_unit
    seq_len = tokens_per_group * num_groups * batch_size
    hidden = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    cos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    plan = gather_plan(
        cells_per_group=cells_per_group,
        keep_per_partition=4,
        num_groups=num_groups,
        batch_size=batch_size,
        strides=[2, 3],          # sample 0 keeps 0,2,4,6; sample 1 keeps 0,3,6,9
    )

    pruned, (kept_cos, _), cu_seqlens, cells = prune_flat_visual_tokens(
        hidden,
        (cos, cos.clone()),
        batch_size=batch_size,
        num_groups=num_groups,
        tokens_per_group=tokens_per_group,
        merge_unit=merge_unit,
        plan=plan,
    )

    picks = {0: (0, 2, 4, 6), 1: (0, 3, 6, 9)}
    expected = [
        (sample * num_groups + group) * tokens_per_group + cell * merge_unit + off
        for sample in range(batch_size)
        for group in range(num_groups)
        for cell in picks[sample]
        for off in range(merge_unit)
    ]
    assert pruned[:, 0].tolist() == expected
    assert kept_cos[:, 0].tolist() == expected
    # One segment per (sample, group), each 4 cells x 4 sub-tokens.
    assert cu_seqlens.tolist() == [0, 16, 32, 48, 64]
    assert [c.tolist() for c in cells[0]] == [list(picks[0])] * num_groups
    assert [c.tolist() for c in cells[1]] == [list(picks[1])] * num_groups


def test_batched_placeholder_resize_is_per_row():
    row = _prompt_row(8, 2)
    inputs = {
        "input_ids": torch.tensor([row, row]),
        "attention_mask": torch.ones(2, len(row), dtype=torch.long),
        "mm_token_type_ids": _video_types(torch.tensor([row, row])),
    }
    cells = [
        [torch.tensor([0, 1, 2]), torch.tensor([5, 6, 7])],
        [torch.tensor([1, 3, 5]), torch.tensor([0, 2, 4])],
    ]

    _, keep_masks = resize_video_placeholders(
        inputs, VIDEO_TOKEN, cells, padding_side="left"
    )

    # Run one spans [1, 9), run two spans [10, 18).
    for row_index, expected in enumerate(cells):
        kept = keep_masks[row_index].nonzero().flatten().tolist()
        assert [i - 1 for i in kept if 1 <= i <= 8] == expected[0].tolist()
        assert [i - 10 for i in kept if 10 <= i <= 17] == expected[1].tolist()


def test_global_and_shared_scopes_are_supported_for_qwen3():
    for i_mode, p_mode in (
        ("preserve", "hevc"),
        ("random", "random"),
        ("uniform", "uniform"),
    ):
        assert (
            validate_qwen_modes("qwen3_vl", "last", "global", i_mode, p_mode)
            .value
            == "global"
        )
    assert (
        validate_qwen_modes(
            "qwen3_vl", "last", "shared", "shared_hevc", "shared_hevc"
        ).value
        == "shared"
    )
    with pytest.raises(RuntimeError, match="Unsupported qwen3vl"):
        validate_qwen_modes("qwen3_vl", "last", "global", "folder", "hevc")


def test_local_qwen_budget_distributes_an_arbitrary_remainder_exactly():
    config = PruningConfig(
        scope=PruningScope.LOCAL,
        reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE,
        keep_per_partition=3,
        keep_total=11,
        signal_source=SignalSource.UNIFORM,
    )
    signal = PruningSignal(
        visible_indices=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
        anchor_partitions=torch.zeros(1, 3, dtype=torch.bool),
    )
    plan = LayerPruningPlan(0, config).bind(signal)

    cells = kept_cell_indices(
        plan, batch_size=1, num_groups=3, cells_per_group=4
    )

    assert [cell.tolist() for cell in cells[0]] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2],
    ]
    assert sum(cell.numel() for cell in cells[0]) == 11
