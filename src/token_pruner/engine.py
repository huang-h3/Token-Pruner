"""Model-neutral pruning over flat/ragged content-token sequences."""

from dataclasses import asdict
from functools import cache

import torch

from .selection import resolve_pruning_budget
from .folder import folder_wrapper
from .shared_folding import fold_spatial_blocks
from .tokens import PruneResult, PruningScope, SignalSource, TokenReducer


def _folder_reduce(load_reducer, cls_token, tokens, keep):
    keep = int(keep)
    if keep == 0:
        return tokens.new_empty((0, tokens.shape[-1]))
    if keep == tokens.shape[0]:
        return tokens
    return load_reducer().reduce(cls_token, tokens.unsqueeze(0), keep).squeeze(0)


def _sample_value(values, batch_idx, batch_size):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.ndim == 1:
            return values
        return values[batch_idx if values.shape[0] > 1 else 0]
    if isinstance(values, (list, tuple)) and len(values) == batch_size:
        return values[batch_idx]
    return values


def _anchor_partition_rows(request, config, signal):
    values = signal.anchor_partitions if signal is not None else None
    shared = config.scope == PruningScope.SHARED
    baseline = config.signal_source in {SignalSource.RANDOM, SignalSource.UNIFORM}
    rows = []
    for batch_idx, count in enumerate(request.partition_counts):
        device = request.samples[batch_idx].device
        if values is not None and not shared:
            rows.append(torch.as_tensor(
                _sample_value(values, batch_idx, request.batch_size),
                dtype=torch.bool, device=device,
            ).flatten())
            continue
        row = torch.zeros(count, dtype=torch.bool, device=device)
        if count and not shared and not baseline:
            row[0] = True
        rows.append(row)
    return rows


def selected_partition_indices(
    signal,
    config,
    *,
    partition_ids,
    partition_idx,
    batch_idx=0,
    batch_size=1,
    device=None,
):
    """Return local indices for one partition; ``shared`` reuses one map everywhere."""

    values = signal.visible_indices
    partition_ids = torch.as_tensor(
        partition_ids, dtype=torch.long, device=device
    ).flatten()
    source = torch.where(partition_ids == int(partition_idx))[0]

    sample = _sample_value(values, batch_idx, batch_size)
    selected = torch.as_tensor(sample, dtype=torch.long).flatten().to(
        partition_ids.device
    )
    if config.scope == PruningScope.SHARED:
        return selected
    chosen = selected[partition_ids[selected] == int(partition_idx)]
    if chosen.numel() == 0:
        return chosen
    inverse = torch.full(
        (partition_ids.numel(),), -1, dtype=torch.long, device=partition_ids.device
    )
    inverse[source] = torch.arange(source.numel(), device=partition_ids.device)
    return inverse[chosen]


def _prune_flat(request, config, signal):
    anchor_rows = _anchor_partition_rows(request, config, signal)
    # The reference implementation is loaded only when a merge actually runs.
    reducer = cache(folder_wrapper)
    output_samples = []
    output_partitions = []
    output_sources = []

    for batch_idx, (tokens, partition_ids, partition_count, anchor_row) in enumerate(
        zip(
            request.samples,
            request.partition_rows,
            request.partition_counts,
            anchor_rows,
        )
    ):
        sizes = {
            partition: int((partition_ids == partition).sum())
            for partition in range(partition_count)
        }
        anchors = anchor_row.nonzero().flatten().tolist()
        budget = resolve_pruning_budget(
            config,
            partition_sizes=sizes,
            anchor_partitions=anchors,
        )
        parts = [None] * partition_count
        sources = [None] * partition_count

        for partition in budget.anchor_partitions:
            source = torch.where(partition_ids == partition)[0]
            part = tokens[source]
            if config.anchor_reducer == TokenReducer.PRESERVE:
                parts[partition] = part
                sources[partition] = source
            else:
                keep = budget.anchor_allocations[partition]
                parts[partition] = _folder_reduce(
                    reducer,
                    request.cls_for_partition(batch_idx, partition),
                    part,
                    keep,
                )
                sources[partition] = torch.full(
                    (keep,), -1, dtype=torch.long, device=tokens.device
                )

        if config.reducer == TokenReducer.PRESERVE:
            for partition in budget.other_partitions:
                source = torch.where(partition_ids == partition)[0]
                parts[partition] = tokens[source]
                sources[partition] = source
        elif config.reducer == TokenReducer.GATHER:
            for partition in budget.other_partitions:
                source = torch.where(partition_ids == partition)[0]
                local = selected_partition_indices(
                    signal,
                    config,
                    partition_ids=partition_ids,
                    partition_idx=partition,
                    batch_idx=batch_idx,
                    batch_size=request.batch_size,
                    device=tokens.device,
                )
                parts[partition] = tokens[source[local]]
                sources[partition] = source[local]
        elif config.reducer in {TokenReducer.FOLDER, TokenReducer.GLOBAL_FOLDER}:
            if config.scope == PruningScope.GLOBAL and budget.other_partitions:
                source_parts = [
                    torch.where(partition_ids == partition)[0]
                    for partition in budget.other_partitions
                ]
                merged = _folder_reduce(
                    reducer,
                    request.cls_for_partitions(
                        batch_idx, list(budget.other_partitions)
                    ),
                    tokens[torch.cat(source_parts)],
                    budget.other_tokens,
                )
                offset = 0
                for partition in budget.other_partitions:
                    keep = budget.other_allocations[partition]
                    parts[partition] = merged[offset : offset + keep]
                    sources[partition] = torch.full(
                        (keep,), -1, dtype=torch.long, device=tokens.device
                    )
                    offset += keep
            else:
                for partition in budget.other_partitions:
                    source = torch.where(partition_ids == partition)[0]
                    keep = budget.other_allocations[partition]
                    parts[partition] = _folder_reduce(
                        reducer,
                        request.cls_for_partition(batch_idx, partition),
                        tokens[source],
                        keep,
                    )
                    sources[partition] = torch.full(
                        (keep,), -1, dtype=torch.long, device=tokens.device
                    )
        else:
            raise RuntimeError(f"Unsupported reducer {config.reducer.value}.")

        sample = torch.cat(parts, dim=0)
        row = torch.cat(
            [
                torch.full(
                    (part.shape[0],),
                    partition,
                    dtype=torch.long,
                    device=tokens.device,
                )
                for partition, part in enumerate(parts)
            ]
        )
        output_samples.append(sample)
        output_partitions.append(row)
        output_sources.append(torch.cat(sources))

    return PruneResult(
        samples=output_samples,
        partition_ids=output_partitions,
        partition_counts=request.partition_counts,
        cls_tokens=request.cls_tokens,
        source_indices=output_sources,
    )


def _uniform_partition_tensor(request):
    return torch.stack([
        torch.stack([sample[row == partition] for partition in range(count)])
        for sample, row, count in zip(
            request.samples, request.partition_rows, request.partition_counts
        )
    ])


def _prune_shared_folding(request, config, signal):
    options = asdict(config.shared_folding)
    patch_width = options.pop("patch_width")
    scores = signal.scores if signal is not None else None
    folded, keep = fold_spatial_blocks(
        _uniform_partition_tensor(request),
        patch_width=int(patch_width),
        hevc_scores=scores,
        **options,
    )
    partitions = folded.shape[1]
    samples = list(folded.flatten(1, 2).unbind(0))
    rows = [
        torch.arange(partitions, device=sample.device).repeat_interleave(int(keep))
        for sample in samples
    ]
    return PruneResult(
        samples=samples,
        partition_ids=rows,
        partition_counts=[partitions] * request.batch_size,
        cls_tokens=request.cls_tokens,
        source_indices=[
            torch.full(
                (sample.shape[0],), -1, dtype=torch.long, device=sample.device
            )
            for sample in samples
        ],
    )


def prune_tokens(request, config, signal=None):
    if config.reducer == TokenReducer.SHARED_FOLDING:
        return _prune_shared_folding(request, config, signal)
    return _prune_flat(request, config, signal)
