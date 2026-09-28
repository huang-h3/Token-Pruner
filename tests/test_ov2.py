"""LLaVA-OneVision-2 packed-sequence contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from token_pruner.run_llava_onevision2 import (
    LlavaOnevision2TowerAdapter,
    row_cell_counts,
    window_cu_seqlens,
)
from token_pruner.packed_cells import kept_cell_indices
from token_pruner.layer_boundary import (
    EncoderPruningContext,
)
from token_pruner.selection import LayerPruningPlan
from token_pruner.tokens import (
    PruningConfig,
    PruningScope,
    PruningSignal,
    SignalSource,
    TokenReducer,
)

FRAMES, HEIGHT, WIDTH, CELLS = 5, 4, 4, 4


class _Block(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rotary_pos_emb=None,
        output_attentions=False,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        del attention_mask, output_attentions, cu_seqlens, max_seqlen
        return (hidden_states + rotary_pos_emb[..., : hidden_states.shape[-1]] / 100,)


class _Merger(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_positions = None

    def forward(self, hidden_states, patch_positions=None):
        self.patch_positions = patch_positions.clone()
        batch, length, width = hidden_states.shape
        return hidden_states.reshape(batch, length // 4, 4, width).sum(2)


class _Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(frame_windows_size=4)
        self.encoder = SimpleNamespace(layers=nn.ModuleList([_Block(), _Block()]))
        self.merger = _Merger()

    def forward(self, hidden_state, grid_thw=None, patch_positions=None):
        hidden = hidden_state.unsqueeze(0)
        rotary = (
            patch_positions.float().sum(-1, keepdim=True).repeat(1, hidden.shape[-1])
        )
        rotary = rotary.unsqueeze(0)
        cu = torch.tensor([0, 64, 80], dtype=torch.int32)
        for layer in self.encoder.layers:
            hidden = layer(
                hidden,
                rotary_pos_emb=rotary,
                cu_seqlens=cu,
                max_seqlen=64,
            )[0]
        return self.merger(hidden, patch_positions=patch_positions)


def _inputs():
    tokens = FRAMES * HEIGHT * WIDTH
    hidden = torch.arange(tokens * 3, dtype=torch.float32).reshape(tokens, 3)
    positions = torch.arange(tokens * 3).reshape(tokens, 3)
    grid = torch.tensor([[FRAMES, HEIGHT, WIDTH]])
    return hidden, grid, positions


def _plan(layer, batch=1):
    selected = torch.tensor([4, 5, 8, 9, 12, 13, 16, 17])
    config = PruningConfig(
        scope=PruningScope.LOCAL,
        reducer=TokenReducer.GATHER,
        anchor_reducer=TokenReducer.PRESERVE,
        keep_per_partition=2,
        keep_total=12,
        signal_source=SignalSource.HEVC,
    )
    signal = PruningSignal(
        visible_indices=[selected] * batch,
        anchor_partitions=torch.tensor([[True, False, False, False, False]] * batch),
    )
    return LayerPruningPlan(layer, config, signal)


def test_disabled_install_is_bit_exact():
    hidden, grid, positions = _inputs()
    baseline = _Tower()
    adapted = _Tower()
    adapted.load_state_dict(baseline.state_dict())
    context = EncoderPruningContext(
        enabled=False,
        num_groups=FRAMES,
        plan=_plan(0),
    )
    cleanup = LlavaOnevision2TowerAdapter().install(adapted, context)
    try:
        assert torch.equal(
            adapted(hidden, grid_thw=grid, patch_positions=positions),
            baseline(hidden, grid_thw=grid, patch_positions=positions),
        )
        assert torch.equal(adapted.merger.patch_positions, positions)
    finally:
        cleanup()


@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_pruning_keeps_cell_rows_positions_and_window_lengths(boundary):
    hidden, grid, positions = _inputs()
    tower = _Tower()
    context = EncoderPruningContext(
        enabled=True,
        num_groups=FRAMES,
        plan=_plan(boundary),
    )
    context.expected_group_cells = kept_cell_indices(
        context.plan,
        batch_size=1,
        num_groups=FRAMES,
        cells_per_group=CELLS,
    )
    adapter = LlavaOnevision2TowerAdapter()
    cleanup = adapter.install(tower, context)
    try:
        output = tower(hidden, grid_thw=grid, patch_positions=positions)
    finally:
        cleanup()

    rows = []
    for frame, cells in enumerate(context.expected_group_cells[0]):
        base = frame * HEIGHT * WIDTH
        for cell in cells:
            rows.extend(range(base + int(cell) * 4, base + int(cell) * 4 + 4))
    expected_positions = positions[torch.tensor(rows)]
    assert torch.equal(tower.merger.patch_positions, expected_positions)
    assert output.shape == (1, 12, 3)
    assert adapter.state.cu_seqlens.tolist() == [0, 40, 48]
    assert int(adapter.state.cu_seqlens[-1]) == len(rows)
    assert row_cell_counts(grid, context.expected_group_cells) == [12]


def test_expanded_frame_rows_remain_independent_attention_segments():
    grid = torch.tensor([[1, 4, 4], [1, 4, 4], [1, 4, 4]])
    kept = [[torch.arange(4), torch.arange(2), torch.arange(3)]]
    cu = window_cu_seqlens(grid, kept, 4, torch.device("cpu"))
    assert cu.tolist() == [0, 16, 24, 36]
    assert row_cell_counts(grid, kept) == [4, 2, 3]


def test_feature_shim_splits_the_pruned_merger_output_by_grid_row():
    from token_pruner.run_llava_onevision2 import (
        _ForwardState,
        install_feature_shim,
    )

    class Visual:
        embeddings = SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=torch.empty(1))
        )

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(last_hidden_state=torch.arange(24).reshape(6, 4))

    class Model:
        def __init__(self):
            self.visual = Visual()

        def get_image_features(self, *_args, **_kwargs):
            return ["stock"]

    model = Model()
    context = SimpleNamespace(enabled=True)
    state = _ForwardState(row_cells=[4, 2])
    cleanup = install_feature_shim(model, context, state)
    try:
        features = model.get_image_features(torch.empty(1), torch.tensor([[1, 1, 1]]))
    finally:
        cleanup()
    assert [part.shape for part in features] == [(4, 4), (2, 4)]


def test_backend_resizes_each_frame_placeholder_run_to_the_same_cells():
    from token_pruner.task_io import placeholder_runs
    from token_pruner.run_llava_onevision2 import (
        LlavaOnevision2Backend,
    )

    backend = LlavaOnevision2Backend.__new__(LlavaOnevision2Backend)
    backend.device = torch.device("cpu")
    backend.model = SimpleNamespace(config=SimpleNamespace(image_token_id=99))
    backend.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0, padding_side="left")
    )
    context = SimpleNamespace(
        plan=_plan(1, batch=2),
        num_groups=FRAMES,
        expected_group_cells=None,
    )
    session = SimpleNamespace(
        pruning_active=True,
        context=context,
        total_patches=CELLS,
    )
    row = torch.tensor([7, 99, 99, 99, 99] * FRAMES + [8])
    inputs = {
        "input_ids": torch.stack([row, row]),
        "attention_mask": torch.ones(2, row.numel(), dtype=torch.long),
    }
    resized = backend.adapt_lm_inputs(inputs, session, effective_keep_total=12)
    for tokens in resized["input_ids"]:
        runs = placeholder_runs(tokens, 99)
        assert [end - start for start, end in runs] == [4, 2, 2, 2, 2]
        assert int((tokens == 99).sum()) == 12
