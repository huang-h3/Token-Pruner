"""Cell selection shared by vision backends."""

import torch
from .engine import prune_tokens, selected_partition_indices
from .tokens import PruneTokens, PruningScope, TokenReducer


def _cell_view(hidden_states, batch_size, num_groups, tokens_per_group, merge_unit):
    """``[B*G*h*w, D]`` -> ``[B, G, cells, merge_unit*D]``."""

    hidden_dim = hidden_states.shape[-1]
    cells = tokens_per_group // merge_unit
    return hidden_states.reshape(batch_size, num_groups, cells, merge_unit * hidden_dim)


def prune_flat_visual_tokens(
    hidden_states,
    position_embeddings,
    *,
    batch_size,
    num_groups,
    tokens_per_group,
    merge_unit,
    plan,
    expected_group_cells=None,
):
    """Reduce one packed visual batch to the planned per-group cell budget."""

    hidden_dim = hidden_states.shape[-1]
    request = PruneTokens.from_partitioned(
        _cell_view(hidden_states, batch_size, num_groups, tokens_per_group, merge_unit)
    )
    signal = plan.signal
    pruned = prune_tokens(request, plan.config, signal)

    kept_cells = kept_cell_indices(
        plan,
        batch_size=batch_size,
        num_groups=num_groups,
        cells_per_group=tokens_per_group // merge_unit,
        device=hidden_states.device,
    )
    if expected_group_cells is not None:
        # Cross-check against the cells the prompt was shrunk to.
        mismatched = [
            (sample_idx, group_idx)
            for sample_idx, (predicted_row, actual_row) in enumerate(
                zip(expected_group_cells, kept_cells)
            )
            for group_idx, (predicted, actual) in enumerate(
                zip(predicted_row, actual_row)
            )
            if not torch.equal(predicted.to(actual.device), actual)
        ]
        shape_changed = [len(row) for row in expected_group_cells] != [
            len(row) for row in kept_cells
        ]
        if mismatched or shape_changed:
            raise RuntimeError(
                "Surviving cells differ from the ones the language-model "
                f"placeholders were resized to; (sample, group)={mismatched}."
            )

    cos, sin = position_embeddings
    offsets = torch.arange(merge_unit, device=hidden_states.device)
    kept_rows, lengths, segments = [], [], []
    for sample_idx in range(batch_size):
        for group_idx, group in enumerate(pruned.partitions[sample_idx]):
            cells = kept_cells[sample_idx][group_idx]
            # Row offsets are global, laid out sample-major then group.
            base = (sample_idx * num_groups + group_idx) * tokens_per_group
            rows = (
                base + cells.unsqueeze(1) * merge_unit + offsets.unsqueeze(0)
            ).reshape(-1)
            kept_rows.append(rows)
            lengths.append(int(rows.numel()))
            segments.append(group.reshape(-1, hidden_dim))

    kept_rows = torch.cat(kept_rows)
    hidden_states = torch.cat(segments, dim=0)
    cu_seqlens = torch.zeros(
        len(lengths) + 1, dtype=torch.int32, device=hidden_states.device
    )
    cu_seqlens[1:] = torch.tensor(
        lengths, dtype=torch.int32, device=hidden_states.device
    ).cumsum(0)
    return (
        hidden_states,
        (cos[kept_rows], sin[kept_rows]),
        cu_seqlens,
        kept_cells,
    )


def kept_cell_indices(plan, *, batch_size, num_groups, cells_per_group, device=None):
    """Surviving cell indices as ``[b][g]``, in the order the engine emits them."""

    signal = plan.signal
    anchors = _anchor_rows(signal, plan.config, batch_size, num_groups)
    full = torch.arange(cells_per_group, dtype=torch.long, device=device)
    partition_ids = torch.arange(
        num_groups, dtype=torch.long, device=device
    ).repeat_interleave(cells_per_group)
    kept = []
    for sample_idx in range(batch_size):
        row = []
        for group_idx in range(num_groups):
            if anchors[sample_idx][group_idx]:
                row.append(full)
                continue
            indices = selected_partition_indices(
                signal,
                plan.config,
                batch_idx=sample_idx,
                partition_ids=partition_ids,
                partition_idx=group_idx,
                batch_size=batch_size,
                device=device,
            )
            row.append(indices)
        kept.append(row)
    return kept


def _anchor_rows(signal, config, batch_size, num_groups):
    """Map the selector's model-topology metadata to ``[B][G]`` booleans."""

    if config.scope == PruningScope.SHARED:
        return [[False] * num_groups for _ in range(batch_size)]
    anchors = signal.anchor_partitions if signal is not None else None
    if anchors is None:
        return [
            [group_idx == 0 for group_idx in range(num_groups)]
            for _ in range(batch_size)
        ]
    anchors = torch.as_tensor(anchors, dtype=torch.bool)
    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0)
    if anchors.shape[0] == 1 and batch_size > 1:
        anchors = anchors.expand(batch_size, -1)
    return anchors.tolist()
