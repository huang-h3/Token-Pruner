"""Selection modes, budgets, and concrete index selection."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .tokens import (
    PruningConfig,
    PruningScope,
    PruningSignal,
    SharedFoldingOptions,
    SignalSource,
    TokenReducer,
)


BASELINE_MODES = ("random", "uniform")
GROUPED_MODE_COMBINATIONS = {
    ("global", "preserve", "hevc"),
    ("global", "preserve", "folder"),
    ("global", "folder", "hevc"),
    ("local", "preserve", "hevc"),
    ("local", "preserve", "folder"),
    ("local", "folder", "hevc"),
    ("local", "folder", "folder"),
}

_SELECTORS = frozenset({
    None,
    "clip_topk",
    "framewise_topk",
    "shared_map",
    "patch_scores",
    "tubelet",
    "tubelet_shared",
    "baseline",
})
_SELECTOR_IDS = frozenset({"patch", "tubelet"})
_PARTITION_RULES = frozenset({
    "clip_free",
    "even_share",
    "fixed_k",
    "shared_k",
    "folding",
})


@dataclass(frozen=True)
class ModeSemantics:
    selector: str | None
    partition_rule: str

    def __post_init__(self):
        if self.selector not in _SELECTORS:
            raise RuntimeError(f"Unknown pruning selector {self.selector!r}.")
        if self.partition_rule not in _PARTITION_RULES:
            raise RuntimeError(
                f"Unknown pruning partition rule {self.partition_rule!r}."
            )


def _partition_rule(scope, uniform_partitions):
    if scope == "shared":
        return "shared_k"
    if scope == "global":
        return "clip_free"
    return "fixed_k" if uniform_partitions else "even_share"


def _route_semantics(scope, i_mode, p_mode, *, uniform_partitions, selector_id):
    rule = _partition_rule(scope, uniform_partitions)
    if p_mode in BASELINE_MODES:
        return ModeSemantics("baseline", rule)
    if p_mode == "shared_hevc":
        selector = "tubelet_shared" if selector_id == "tubelet" else "shared_map"
        return ModeSemantics(selector, rule)
    if p_mode == "shared_folding":
        return ModeSemantics("patch_scores", "folding")
    if p_mode in {"folder", "global_folder"}:
        return ModeSemantics(None, rule)
    if p_mode != "hevc":
        raise RuntimeError(f"Unsupported pruning p_mode={p_mode!r}.")
    if selector_id == "tubelet":
        selector = "tubelet_shared" if scope == "shared" else "tubelet"
        return ModeSemantics(selector, rule)
    if scope == "shared":
        return ModeSemantics("shared_map", rule)
    if scope == "local" and i_mode == "folder" and uniform_partitions:
        return ModeSemantics("framewise_topk", rule)
    return ModeSemantics("clip_topk", rule)


_GROUPED = tuple(GROUPED_MODE_COMBINATIONS)


def _baseline_keys(scopes):
    return tuple(
        (scope, p_mode, p_mode)
        for scope in scopes
        for p_mode in BASELINE_MODES
    )


_COMMON_MODE_KEYS = (
    *_GROUPED,
    ("global", "global_folder", "global_folder"),
    ("shared", "shared_hevc", "shared_hevc"),
    ("shared", "shared_folding", "shared_folding"),
    *_baseline_keys(("global", "local")),
)


def _semantic_table(selector_id, uniform_partitions):
    return {
        key: _route_semantics(
            *key,
            uniform_partitions=uniform_partitions,
            selector_id=selector_id,
        )
        for key in _COMMON_MODE_KEYS
    }


def mode_semantics(
    scope,
    i_mode,
    p_mode,
    *,
    uniform_partitions=False,
    selector_id="patch",
):
    scope = getattr(scope, "value", str(scope))
    i_mode = str(i_mode)
    p_mode = str(p_mode)
    selector_id = "patch" if selector_id is None else str(selector_id)
    if selector_id not in _SELECTOR_IDS:
        raise RuntimeError(f"Unknown pruning selector id {selector_id!r}.")
    key = (scope, p_mode, p_mode) if p_mode in BASELINE_MODES else (scope, i_mode, p_mode)
    try:
        return _semantic_table(selector_id, bool(uniform_partitions))[key]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported pruning semantics for {(scope, i_mode, p_mode)}."
        ) from exc


_REDUCERS = {
    "hevc": TokenReducer.GATHER,
    "shared_hevc": TokenReducer.GATHER,
    "folder": TokenReducer.FOLDER,
    "global_folder": TokenReducer.GLOBAL_FOLDER,
    "random": TokenReducer.GATHER,
    "uniform": TokenReducer.GATHER,
}
_SIGNAL_SOURCES = {
    "hevc": SignalSource.HEVC,
    "shared_hevc": SignalSource.HEVC,
    "folder": SignalSource.NONE,
    "global_folder": SignalSource.NONE,
    "random": SignalSource.RANDOM,
    "uniform": SignalSource.UNIFORM,
}
_ANCHOR_REDUCERS = {
    "preserve": TokenReducer.PRESERVE,
    "shared_hevc": TokenReducer.PRESERVE,
    "shared_folding": TokenReducer.PRESERVE,
    "folder": TokenReducer.FOLDER,
    "global_folder": TokenReducer.PRESERVE,
    "random": TokenReducer.PRESERVE,
    "uniform": TokenReducer.PRESERVE,
}


def normalize_pruning_scope(model_family, stage, prune_mode):
    try:
        return PruningScope(str(prune_mode))
    except ValueError as exc:
        raise RuntimeError(
            f"Unsupported prune_mode={prune_mode!r} for {model_family} at {stage}."
        ) from exc


def validate_pruning_modes(
    legal_modes,
    model_family,
    stage,
    prune_mode,
    i_mode,
    p_mode,
):
    family = str(model_family).strip().lower().replace("-", "").replace("_", "")
    scope = normalize_pruning_scope(family, stage, prune_mode)
    combination = (str(stage), scope.value, str(i_mode), str(p_mode))
    if combination not in frozenset(legal_modes):
        raise RuntimeError(
            f"Unsupported {family} pruning configuration {combination}; "
            f"supported={sorted(legal_modes)}."
        )
    return scope


def build_model_pruning_config(
    *,
    model_family,
    stage,
    prune_mode,
    i_mode,
    p_mode,
    keep_per_partition,
    keep_total,
    legal_modes,
    uniform_partitions=False,
    min_per_partition=0,
):
    scope = validate_pruning_modes(
        legal_modes, model_family, stage, prune_mode, i_mode, p_mode
    )
    return PruningConfig(
        scope=scope,
        reducer=_REDUCERS[p_mode],
        anchor_reducer=_ANCHOR_REDUCERS[i_mode],
        keep_per_partition=int(keep_per_partition),
        keep_total=int(keep_total),
        signal_source=_SIGNAL_SOURCES[p_mode],
        uniform_partitions=bool(uniform_partitions),
        min_per_partition=int(min_per_partition),
    )


def build_shared_folding_config(
    *, options, keep_total, legal_modes, model_family="model", stage="input"
):
    if not isinstance(options, SharedFoldingOptions):
        options = SharedFoldingOptions(**dict(options))
    combination = (str(stage), "shared", "shared_folding", "shared_folding")
    if combination not in frozenset(legal_modes):
        raise RuntimeError(
            f"shared folding is not supported by {model_family} at stage={stage}."
        )
    return PruningConfig(
        scope=PruningScope.SHARED,
        reducer=TokenReducer.SHARED_FOLDING,
        anchor_reducer=TokenReducer.PRESERVE,
        keep_per_partition=options.global_k,
        keep_total=int(keep_total),
        uniform_partitions=True,
        signal_source=(
            SignalSource.HEVC if options.mode == "hevc-avg" else SignalSource.NONE
        ),
        shared_folding=options,
    )


def _as_quota(value, size, name):
    if value is None:
        return [None] * int(size)
    if isinstance(value, (int, float)):
        return [int(value)] * int(size)
    values = [int(item) for item in value]
    if len(values) != int(size):
        raise RuntimeError(f"{name} must have one value per block ({size}).")
    return values


def _rank(values, count, stable):
    count = int(count)
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    if count > values.numel():
        raise RuntimeError(f"cannot select {count} items from {values.numel()}.")
    if stable:
        return torch.argsort(values, descending=True, stable=True)[:count]
    return torch.topk(values, k=count).indices


def _select_one(
    scores,
    budget,
    block_ids,
    floor,
    ceiling,
    order_by,
    stable,
    clamp,
):
    groups, items = scores.shape
    block_ids = torch.as_tensor(block_ids, dtype=torch.long, device=scores.device)
    if block_ids.numel() != groups:
        raise RuntimeError("block_ids must have one entry per score block.")
    floors = _as_quota(floor, groups, "floor")
    ceilings = _as_quota(ceiling, groups, "ceiling")
    selected = []
    selected_mask = torch.zeros_like(scores, dtype=torch.bool)

    for group in range(groups):
        lower = max(0, int(floors[group] or 0))
        upper = items if ceilings[group] is None else int(ceilings[group])
        if clamp:
            lower = min(lower, items)
            upper = min(max(upper, 0), items)
        if lower > upper:
            raise RuntimeError(f"floor exceeds ceiling for block {group}.")
        if lower:
            local = _rank(scores[group], lower, stable)
            selected_mask[group, local] = True
            selected.append(group * items + local)

    selected_count = sum(item.numel() for item in selected)
    target = int(budget)
    if target < selected_count:
        raise RuntimeError(
            f"budget={target} is smaller than the required floor {selected_count}."
        )

    extra_count = target - selected_count
    if extra_count:
        eligible = torch.full_like(scores, -float("inf"))
        available = 0
        for group in range(groups):
            capacity = ceilings[group]
            if capacity is None:
                capacity = items
            capacity = min(max(int(capacity), 0), items) if clamp else int(capacity)
            used = int(selected_mask[group].sum())
            if used < capacity:
                available += capacity - used
                eligible[group] = scores[group].masked_fill(
                    selected_mask[group], -float("inf")
                )
        if extra_count > available:
            if not clamp:
                raise RuntimeError(
                    f"budget={target} exceeds the available selection capacity."
                )
            extra_count = available
        if extra_count:
            local = _rank(eligible.reshape(-1), extra_count, stable)
            selected.append(local)

    if not selected:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    flat = torch.cat(selected)
    group_index = torch.div(flat, items, rounding_mode="floor")
    item_index = flat.remainder(items)
    actual = block_ids[group_index] * items + item_index
    if order_by is None:
        return actual.sort().values
    order_by = torch.as_tensor(order_by, device=scores.device)
    keys = order_by.reshape(-1)[flat]
    order = torch.argsort(keys, stable=bool(stable))
    return actual[order]


def select_with_floor(
    scores,
    budget,
    *,
    block_ids=None,
    floor=0,
    ceiling=None,
    order_by=None,
    clamp=True,
    stable=False,
):
    """Select a score-ranked budget with per-block lower/upper bounds."""
    scores = torch.as_tensor(scores)
    if scores.ndim not in (2, 3):
        raise RuntimeError("scores must have shape [blocks, items] or [batch, blocks, items].")
    batched = scores.ndim == 3
    if not batched:
        scores = scores.unsqueeze(0)
    _, groups, items = scores.shape
    if items <= 0 or groups <= 0:
        raise RuntimeError("scores must contain at least one block and one item.")

    if block_ids is None:
        block_ids = torch.arange(groups, device=scores.device)
    block_ids = torch.as_tensor(block_ids, dtype=torch.long, device=scores.device)
    if block_ids.ndim == 1:
        block_ids = block_ids.unsqueeze(0).expand(scores.shape[0], -1)
    if block_ids.shape != scores.shape[:2]:
        raise RuntimeError("block_ids must have shape [blocks] or [batch, blocks].")

    if order_by is not None:
        order_by = torch.as_tensor(order_by, device=scores.device)
        if order_by.ndim == 2:
            order_by = order_by.unsqueeze(0)
        if order_by.shape != scores.shape:
            raise RuntimeError("order_by must have the same shape as scores.")

    outputs = []
    for batch in range(scores.shape[0]):
        outputs.append(
            _select_one(
                scores[batch],
                budget,
                block_ids[batch],
                floor,
                ceiling,
                None if order_by is None else order_by[batch],
                stable,
                clamp,
            )
        )
    output = torch.stack(outputs)
    return output[0] if not batched else output


def distribute_budget(total, partitions, capacities=None):
    """Distribute an exact budget across partitions; remainders go to earlier labels."""

    partitions = [int(partition) for partition in partitions]
    if not partitions:
        return {}
    if capacities is None:
        base, extra = divmod(max(int(total), 0), len(partitions))
        return {
            partition: base + (order < extra)
            for order, partition in enumerate(partitions)
        }

    capacities = {int(key): max(int(value), 0) for key, value in capacities.items()}
    allocations = {partition: 0 for partition in partitions}
    remaining = min(
        max(int(total), 0),
        sum(capacities[partition] for partition in partitions),
    )
    while remaining:
        eligible = [
            partition
            for partition in partitions
            if allocations[partition] < capacities[partition]
        ]
        if not eligible:
            break
        share, extra = divmod(remaining, len(eligible))
        used = 0
        for order, partition in enumerate(eligible):
            requested = share + (order < extra)
            if requested == 0:
                continue
            room = capacities[partition] - allocations[partition]
            addition = min(requested, room)
            allocations[partition] += addition
            used += addition
        if used == 0:
            break
        remaining -= used
    return allocations


@dataclass(frozen=True)
class PruningBudget:
    """Resolved token counts and per-partition allocations."""

    anchor_partitions: tuple[int, ...]
    other_partitions: tuple[int, ...]
    anchor_allocations: dict[int, int]
    other_allocations: dict[int, int]

    @property
    def anchor_tokens(self):
        return sum(self.anchor_allocations.values())

    @property
    def other_tokens(self):
        return sum(self.other_allocations.values())

    @property
    def total_tokens(self):
        return self.anchor_tokens + self.other_tokens


def resolve_pruning_budget(
    config,
    *,
    partition_sizes,
    anchor_partitions=(),
):
    """Resolve output counts from flat-token topology and pruning policy."""

    sizes = {
        int(partition): max(int(size), 0)
        for partition, size in dict(partition_sizes).items()
    }
    partitions = tuple(sorted(sizes))
    anchors = tuple(
        partition
        for partition in sorted({int(value) for value in anchor_partitions})
        if partition in sizes
    )
    if config.scope == PruningScope.SHARED or config.reducer == TokenReducer.GLOBAL_FOLDER:
        anchors = ()
    others = tuple(partition for partition in partitions if partition not in anchors)

    if config.reducer == TokenReducer.SHARED_FOLDING:
        allocations = distribute_budget(
            config.keep_total,
            partitions,
            capacities=sizes,
        )
        return PruningBudget(
            anchor_partitions=(),
            other_partitions=partitions,
            anchor_allocations={},
            other_allocations=allocations,
        )

    if config.scope == PruningScope.SHARED:
        keep = int(config.keep_per_partition)
        return PruningBudget(
            anchor_partitions=(),
            other_partitions=partitions,
            anchor_allocations={},
            other_allocations={
                partition: min(keep, sizes[partition]) for partition in partitions
            },
        )

    keep = int(config.keep_per_partition)
    anchor_allocations = {
        partition: (
            sizes[partition]
            if config.anchor_reducer == TokenReducer.PRESERVE
            else min(keep, sizes[partition])
        )
        for partition in anchors
    }
    anchor_cost = sum(anchor_allocations.values())
    keep_total = None if config.keep_total is None else int(config.keep_total)
    if keep_total is not None:
        if anchor_cost > keep_total:
            raise RuntimeError(
                "Dense GOP anchors exceed the configured total token budget: "
                f"anchor_cost={anchor_cost}, keep_total={keep_total}."
            )
        minimum_other_cost = int(config.min_per_partition) * len(others)
        if anchor_cost + minimum_other_cost > keep_total:
            raise RuntimeError(
                f"anchor_cost={anchor_cost} + minimum_other_cost="
                f"{minimum_other_cost} exceeds keep_total={keep_total}"
            )
    if config.reducer == TokenReducer.PRESERVE:
        other_allocations = {partition: sizes[partition] for partition in others}
    elif config.uniform_partitions and (
        config.scope == PruningScope.LOCAL or config.reducer == TokenReducer.GLOBAL_FOLDER
    ):
        other_allocations = {
            partition: min(keep, sizes[partition]) for partition in others
        }
    else:
        target = (
            keep * len(others)
            if keep_total is None
            else max(keep_total - anchor_cost, 0)
        )
        other_allocations = distribute_budget(
            target,
            others,
            capacities=sizes,
        )

    return PruningBudget(
        anchor_partitions=anchors,
        other_partitions=others,
        anchor_allocations=anchor_allocations,
        other_allocations=other_allocations,
    )


@dataclass(frozen=True)
class LayerPruningPlan:
    """Apply ``config`` immediately before encoder layer ``layer``."""

    layer: int
    config: PruningConfig
    signal: PruningSignal | None = None

    def bind(self, signal: PruningSignal | None) -> "LayerPruningPlan":
        """Return the same plan with the runtime signal attached."""

        return replace(self, signal=signal)


def resolve_pruning_layer(
    layer: int | None,
    *,
    stage: str,
    num_layers: int,
) -> int:
    """Resolve a stage or Python-style layer index to an encoder boundary."""

    if layer is None:
        layer = 0 if stage == "input" else num_layers - 1
    requested = int(layer)
    layer = requested + num_layers if requested < 0 else requested
    if not 0 <= layer <= num_layers:
        raise ValueError(
            f"prune_layer={requested} is outside the {num_layers}-layer encoder."
        )
    return layer


@dataclass(frozen=True)
class VisualBudget:
    """How many tokens survive, per partition and in total."""

    keep_per_partition: int
    total_per_partition: int
    num_partitions: int
    keep_total: int
    pruning_active: bool

    @property
    def total_tokens(self):
        return self.total_per_partition * self.num_partitions


def resolve_keep_budget(image_size, patch_size, *, k_keep_rate=None, k_keep=None):
    """Patches kept per partition, and how many there are before pruning."""

    total_patches = (int(image_size) // int(patch_size)) ** 2
    if k_keep is not None:
        keep = int(k_keep)
    else:
        keep = max(1, int(total_patches * float(k_keep_rate)))
    return min(keep, total_patches), total_patches


def resolve_visual_budget(*, image_size, patch_size, num_partitions,
                          k_keep_rate, run_mode, run_pruned, p_mode):
    """Resolve the whole visual budget for one configuration."""

    num_partitions = int(num_partitions)
    if num_partitions <= 0:
        raise RuntimeError(f"invalid partition count {num_partitions}")
    effective_rate = 1.0 if run_mode == "full" else float(k_keep_rate)
    keep_patches, total_patches = resolve_keep_budget(
        image_size, patch_size, k_keep_rate=effective_rate)
    total_clip = total_patches * num_partitions
    if run_mode == "full":
        keep_total = total_clip
    elif p_mode == "shared_folding":
        keep_total = keep_patches * num_partitions
    else:
        keep_total = max(1, int(total_clip * effective_rate))
    return VisualBudget(
        keep_per_partition=keep_patches,
        total_per_partition=total_patches,
        num_partitions=num_partitions,
        keep_total=keep_total,
        pruning_active=bool(run_pruned and keep_patches < total_patches),
    )


def build_folding_options(*, keep_patches, patch_width, folding_mode,
                          fold_block_size, fold_pooling):
    """Match block slots to K when possible, otherwise use shared global K."""

    block_size = int(fold_block_size)
    if block_size <= 0:
        raise RuntimeError("fold_block_size must be positive.")
    blocks_per_side, grid_remainder = divmod(int(patch_width), block_size)
    slots_per_block, block_wise = 1, False
    if grid_remainder == 0:
        num_blocks = blocks_per_side ** 2
        slots_per_block, budget_remainder = divmod(int(keep_patches), num_blocks)
        block_wise = budget_remainder == 0 and 0 < slots_per_block <= block_size ** 2
    return SharedFoldingOptions(
        mode=str(folding_mode), patch_width=int(patch_width),
        block_size=block_size,
        slots_per_block=slots_per_block if block_wise else 1,
        pooling=str(fold_pooling), block_wise=block_wise,
        global_k=None if block_wise else int(keep_patches),
    )


def build_pruning_config(*, family, stage, scope, i_mode, p_mode, legal_modes,
                         budget, patch_width, uniform_partitions,
                         min_per_partition=0, folding_mode="temporal-diff",
                         fold_block_size=2, fold_pooling="weighted"):
    """The engine configuration one policy resolves to, or ``None`` for a full run."""

    if not budget.pruning_active:
        return None
    if legal_modes is None:
        raise RuntimeError(f"{family} must declare legal_modes before pruning.")
    if p_mode == "shared_folding":
        validate_pruning_modes(legal_modes, family, stage, scope, i_mode, p_mode)
        return build_shared_folding_config(
            options=build_folding_options(
                keep_patches=budget.keep_per_partition, patch_width=patch_width,
                folding_mode=folding_mode, fold_block_size=fold_block_size,
                fold_pooling=fold_pooling,
            ),
            keep_total=budget.keep_total, model_family=family, stage=stage,
            legal_modes=legal_modes,
        )
    return build_model_pruning_config(
        model_family=family, stage=stage, prune_mode=scope,
        legal_modes=legal_modes, uniform_partitions=uniform_partitions,
        i_mode=i_mode, p_mode=p_mode,
        keep_per_partition=budget.keep_per_partition,
        keep_total=budget.keep_total, min_per_partition=int(min_per_partition),
    )
