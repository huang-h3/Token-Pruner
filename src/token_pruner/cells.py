"""Cell selection shared by vision backends."""

from math import isqrt
import torch
from .engine import prune_tokens, selected_partition_indices
from .selection import resolve_pruning_budget
from .tokens import PruneTokens, PruningScope, TokenReducer


def cell_members(grid_side, cell_side, *, device=None):
    """Return row-major patch indices for every cell."""

    grid_side, cell_side = int(grid_side), int(cell_side)
    cells_per_side = grid_side // cell_side
    rows = torch.arange(cells_per_side, device=device).repeat_interleave(cells_per_side)
    columns = torch.arange(cells_per_side, device=device).repeat(cells_per_side)
    origins = rows * cell_side * grid_side + columns * cell_side
    offsets = torch.arange(cell_side, device=device)
    block = (offsets[:, None] * grid_side + offsets[None, :]).flatten()
    return origins[:, None] + block[None, :]


def gather_cells(tokens, *, grid_side, cell_side, cells=None):
    """View selected cells as B x cells x members x hidden."""

    members = cell_members(grid_side, cell_side, device=tokens.device)
    if cells is not None:
        cells = torch.as_tensor(cells, dtype=torch.long, device=tokens.device)
        members = members.index_select(0, cells)
    batch, _, hidden = tokens.shape
    gathered = tokens.index_select(1, members.flatten())
    return gathered.reshape(batch, members.shape[0], members.shape[1], hidden)


def gather_kept_cells(tokens, *, grid_side, cell_side, cells):
    """Restore cell membership after the tower gathered patch positions."""

    cells = torch.as_tensor(cells, dtype=torch.long, device=tokens.device)
    members = cell_members(grid_side, cell_side, device=tokens.device).index_select(
        0, cells
    )
    assert tokens.shape[1] == members.numel(), "token count differs from selected cells"
    survivors = members.flatten().sort().values
    positions = torch.full(
        (grid_side * grid_side,), -1, dtype=torch.long, device=tokens.device
    )
    positions[survivors] = torch.arange(
        survivors.numel(), dtype=torch.long, device=tokens.device
    )
    batch, _, hidden = tokens.shape
    gathered = tokens.index_select(1, positions[members].flatten())
    return gathered.reshape(batch, cells.numel(), members.shape[1], hidden)


def predict_cells(plan, *, batch_size, num_groups, grid_side, cell_side, device=None):
    """Predict the cell slots used to shrink placeholders in the language model."""

    per_group = cell_members(grid_side, cell_side, device=device).shape[0]
    batch_size, num_groups = int(batch_size), int(num_groups)
    config, signal = plan.config, plan.signal
    anchors = _anchor_rows(signal, config, batch_size, num_groups)
    partition_sizes = dict.fromkeys(range(num_groups), int(per_group))
    partition_ids = torch.arange(
        num_groups, dtype=torch.long, device=device
    ).repeat_interleave(per_group)
    full = torch.arange(per_group, dtype=torch.long, device=device)

    predictions = []
    for sample in range(batch_size):
        anchor_ids = [group for group, value in enumerate(anchors[sample]) if value]
        budget = resolve_pruning_budget(
            config,
            partition_sizes=partition_sizes,
            anchor_partitions=anchor_ids,
        )
        row = []
        for group in range(num_groups):
            if group in budget.anchor_partitions:
                keep = budget.anchor_allocations[group]
                reducer = config.anchor_reducer
            else:
                keep = budget.other_allocations[group]
                reducer = config.reducer

            if reducer == TokenReducer.PRESERVE:
                cells = full
            elif reducer == TokenReducer.GATHER:
                cells = selected_partition_indices(
                    signal,
                    config,
                    partition_ids=partition_ids,
                    partition_idx=group,
                    batch_idx=sample,
                    batch_size=batch_size,
                    device=device,
                )
            else:
                cells = torch.arange(keep, dtype=torch.long, device=device)
            row.append(cells)
        predictions.append(row)
    return predictions


def cell_view_cls(cls_tokens, *, num_groups, cell_side):
    """Expand each CLS vector to one cell-vector width."""

    rows, _, hidden = cls_tokens.shape
    return cls_tokens.reshape(
        rows // int(num_groups), int(num_groups), 1, hidden
    ).repeat(1, 1, 1, int(cell_side) ** 2)


def split_group_features(features, cells, *, num_groups, cell_side, has_cls=True):
    """Split rectangular or sample-packed tower output into groups."""

    batch, num_groups = len(cells), int(num_groups)
    assert all(len(row) == num_groups for row in cells), "group topology changed"
    if features.shape[0] == batch * num_groups:
        assert all(
            features.shape[1] == int(has_cls) + len(selection) * int(cell_side) ** 2
            for row in cells
            for selection in row
        ), "rectangular group widths differ from selected cells"
        return [
            [
                features[sample * num_groups + group : sample * num_groups + group + 1]
                for group in range(num_groups)
            ]
            for sample in range(batch)
        ]

    member_count = int(cell_side) ** 2
    lead = int(has_cls)
    rows = []
    for sample, selections in enumerate(cells):
        offset = 0
        row = []
        for selection in selections:
            width = lead + len(selection) * member_count
            row.append(features[sample : sample + 1, offset : offset + width])
            offset += width
        rows.append(row)
    return rows


def join_group_features(rows):
    """Restore rectangular output when possible, otherwise pack per sample."""

    parts = [part for row in rows for part in row]
    if len({part.shape[1] for part in parts}) == 1:
        return torch.cat(parts, dim=0)
    return torch.cat([torch.cat(row, dim=1) for row in rows], dim=0)


def reduce_cells(
    patches,
    plan,
    *,
    num_groups,
    grid_side,
    cell_side,
    expected=None,
    cls_tokens=None,
):
    """Run the pruning engine on cell vectors."""

    num_groups = int(num_groups)
    batch, remainder = divmod(patches.shape[0], num_groups)
    per_group = cell_members(grid_side, cell_side, device=patches.device).shape[0]
    view = gather_cells(patches, grid_side=grid_side, cell_side=cell_side).reshape(
        batch, num_groups, per_group, -1
    )
    result = prune_tokens(
        PruneTokens.from_partitioned(view, cls_tokens=cls_tokens),
        plan.config,
        plan.signal,
    )

    reduced = []
    for sample in range(batch):
        row = []
        for group in range(num_groups):
            mask = result.partition_ids[sample] == group
            cells = result.samples[sample][mask]
            sources = result.source_indices[sample][mask]
            slots = (
                sources.remainder(per_group)
                if sources.numel() and bool((sources >= 0).all())
                else torch.arange(cells.shape[0], dtype=torch.long, device=cells.device)
            )
            row.append((cells, slots))
        reduced.append(row)

    if expected is not None:
        actual = [[slots for _, slots in row] for row in reduced]
        matches = len(expected) == len(actual) and all(
            len(wanted_row) == len(actual_row)
            and all(
                torch.equal(torch.as_tensor(wanted, device=slots.device), slots)
                for wanted, slots in zip(wanted_row, actual_row)
            )
            for wanted_row, actual_row in zip(expected, actual)
        )
        assert matches, "prompt/tower topology placed cells in different slots"
    return reduced


def lay_down_cells(
    hidden_states,
    reduced,
    *,
    num_groups,
    grid_side,
    cell_side,
    has_cls=True,
):
    """Lay reduced cells back into tower token order."""

    members = cell_members(grid_side, cell_side, device=hidden_states.device)
    lead = int(has_cls)
    hidden = hidden_states.shape[-1]
    rows = []
    for index in range(hidden_states.shape[0]):
        cells, slots = reduced[index // int(num_groups)][index % int(num_groups)]
        positions = members.index_select(0, slots.to(members.device)).flatten()
        tokens = cells.reshape(-1, hidden).index_select(0, positions.argsort())
        patches = tokens.unsqueeze(0)
        rows.append(
            torch.cat((hidden_states[index : index + 1, :lead], patches), dim=1)
            if lead
            else patches
        )
    if len({row.shape[1] for row in rows}) == 1:
        return torch.cat(rows, dim=0)
    return rows


def grid_side_from_tokens(tokens):
    """Return the square grid side represented by a token count."""

    side = isqrt(int(tokens))
    if side * side != int(tokens):
        raise RuntimeError(f"{tokens} patch tokens do not form a square grid.")
    return side


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
