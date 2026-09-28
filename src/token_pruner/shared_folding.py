"""Spatial-block folding on hidden features, and its option resolution."""

from dataclasses import replace

import torch
import torch.nn.functional as nnf

from .selection import select_with_floor
from .tokens import SharedFoldingOptions


FOLDING_MODES = (
    "uniform",
    "activation",
    "temporal-diff",
    "temporal-cosine",
    "activation-temporal-diff",
    "activation-temporal-cosine",
    "hevc-avg",
)
FOLDING_POOLING_MODES = ("weighted", "representative", "coverage-hard")
SCORE_NORMALIZATIONS = ("auto", "none", "minmax", "zscore")
TEMPORAL_FEATURE_NORMS = ("none", "layernorm", "l2")
TEMPORAL_SPATIAL_NORMALIZATIONS = ("none", "minmax", "zscore")


def _spatial_blocks(tokens, patch_height, patch_width, block_size):
    B = tokens.shape[0]
    tail = tokens.shape[2:]
    block_area = block_size * block_size
    block_h = patch_height // block_size
    block_w = patch_width // block_size
    return (
        tokens.reshape(B, patch_height, patch_width, *tail)
        .reshape(B, block_h, block_size, block_w, block_size, *tail)
        .permute(0, 1, 3, 2, 4, *range(5, 5 + len(tail)))
        .reshape(B, block_h * block_w, block_area, *tail)
    )


def _normalize_last_dim(scores, normalization, eps, error_message):
    """Normalize over the last axis; shared by score and spatial-map paths."""
    if normalization == "none":
        return scores
    if normalization == "minmax":
        min_score = scores.min(dim=-1, keepdim=True).values
        max_score = scores.max(dim=-1, keepdim=True).values
        return (scores - min_score) / (max_score - min_score).clamp_min(eps)
    if normalization == "zscore":
        mean_score = scores.mean(dim=-1, keepdim=True)
        std_score = scores.std(dim=-1, keepdim=True, unbiased=False)
        return (scores - mean_score) / std_score.clamp_min(eps)
    raise RuntimeError(error_message)


def _normalize_score_vector(scores, normalization, eps=1e-6):
    return _normalize_last_dim(
        scores,
        normalization,
        eps,
        f"score normalization must be one of {SCORE_NORMALIZATIONS}, "
        f"got {normalization}.",
    )


def _normalize_spatial_maps(scores, normalization, eps=1e-6):
    return _normalize_last_dim(
        scores,
        normalization,
        eps,
        f"temporal spatial normalization must be one of "
        f"{TEMPORAL_SPATIAL_NORMALIZATIONS}, got {normalization}.",
    )


def _temporal_features(patches_f32, temporal_feature_norm):
    if temporal_feature_norm == "none":
        return patches_f32
    if temporal_feature_norm == "layernorm":
        return nnf.layer_norm(patches_f32, (patches_f32.shape[-1],))
    if temporal_feature_norm == "l2":
        return nnf.normalize(patches_f32, dim=-1)
    raise RuntimeError(
        f"temporal feature norm must be one of {TEMPORAL_FEATURE_NORMS}, got {temporal_feature_norm}."
    )


def _temporal_component_scores(
    patches_f32,
    mode,
    temporal_feature_norm,
    temporal_spatial_normalization,
):
    B, F, N, D = patches_f32.shape
    if F <= 1:
        return torch.zeros(B, N, device=patches_f32.device, dtype=torch.float32)

    temporal_patches = _temporal_features(patches_f32, temporal_feature_norm)
    if mode == "temporal-diff":
        frame_scores = (temporal_patches[:, 1:] - temporal_patches[:, :-1]).norm(dim=-1)
    elif mode == "temporal-cosine":
        frame_scores = 1.0 - nnf.cosine_similarity(
            temporal_patches[:, 1:],
            temporal_patches[:, :-1],
            dim=-1,
        )
    else:
        raise RuntimeError(f"Unknown temporal component mode '{mode}'.")

    frame_scores = _normalize_spatial_maps(frame_scores, temporal_spatial_normalization)
    return frame_scores.mean(dim=1)


def _resolved_score_normalization(mode, score_normalization):
    if score_normalization == "auto":
        if mode in (
            "activation-temporal-diff",
            "activation-temporal-cosine",
            "hevc-avg",
        ):
            return "minmax"
        return "none"
    return score_normalization


def _clip_level_scores(
    patches,
    mode,
    score_alpha=0.5,
    score_normalization="auto",
    temporal_feature_norm="layernorm",
    temporal_spatial_normalization="none",
    hevc_scores=None,
):
    B, F, N, D = patches.shape
    patches_f32 = patches.float()

    if mode == "uniform":
        return torch.zeros(B, N, device=patches.device, dtype=torch.float32)

    normalization = _resolved_score_normalization(mode, score_normalization)
    if mode == "hevc-avg":
        scores = hevc_scores.to(device=patches.device, dtype=torch.float32).mean(dim=1)
        return _normalize_score_vector(scores, normalization)

    activation_scores = patches_f32.norm(dim=-1).mean(dim=1)
    if mode == "activation":
        return _normalize_score_vector(activation_scores, normalization)

    if mode == "temporal-diff":
        scores = _temporal_component_scores(
            patches_f32,
            "temporal-diff",
            temporal_feature_norm,
            temporal_spatial_normalization,
        )
        return _normalize_score_vector(scores, normalization)
    if mode == "temporal-cosine":
        scores = _temporal_component_scores(
            patches_f32,
            "temporal-cosine",
            temporal_feature_norm,
            temporal_spatial_normalization,
        )
        return _normalize_score_vector(scores, normalization)

    if mode in ("activation-temporal-diff", "activation-temporal-cosine"):
        temporal_mode = (
            "temporal-diff" if mode == "activation-temporal-diff" else "temporal-cosine"
        )
        temporal_scores = _temporal_component_scores(
            patches_f32,
            temporal_mode,
            temporal_feature_norm,
            temporal_spatial_normalization,
        )
        activation_scores = _normalize_score_vector(activation_scores, normalization)
        temporal_scores = _normalize_score_vector(temporal_scores, normalization)
        return (
            float(score_alpha) * activation_scores
            + (1.0 - float(score_alpha)) * temporal_scores
        )

    raise RuntimeError(
        f"Unknown folding mode '{mode}'. Expected one of: {', '.join(FOLDING_MODES)}."
    )


def _uniform_slot_groups(block_area, slots_per_block, device):
    group_ids = (
        torch.arange(block_area, device=device)
        * int(slots_per_block)
        // int(block_area)
    )
    return [
        torch.nonzero(group_ids == slot_idx, as_tuple=False).flatten()
        for slot_idx in range(int(slots_per_block))
    ]


def _weighted_pool_slot_group(patch_blocks, score_blocks, group_idx, mode, temperature):
    group_patches = patch_blocks[:, :, :, group_idx]
    group_scores = score_blocks[:, :, group_idx]

    if mode == "uniform" or temperature == 0:
        weights = torch.full_like(group_scores, 1.0 / group_scores.shape[-1])
    else:
        weights = torch.softmax(float(temperature) * group_scores, dim=-1)

    weights = weights.to(dtype=patch_blocks.dtype)
    return (group_patches * weights[:, None, :, :, None]).sum(dim=3)


def _coverage_hard_blocks(
    patch_blocks,
    score_blocks,
    spatial_index_blocks,
    slots_per_block,
    min_slots_per_block,
):
    B, F, num_blocks, block_area, D = patch_blocks.shape
    slots = int(slots_per_block)
    min_slots = int(min_slots_per_block)

    flat_patches = patch_blocks.reshape(B, F, num_blocks * block_area, D)
    spatial_index_blocks = spatial_index_blocks.to(
        device=patch_blocks.device, dtype=torch.long
    ).expand(B, -1, -1)
    selected_flat_idx = select_with_floor(
        score_blocks,
        num_blocks * slots,
        floor=min_slots,
        ceiling=slots,
        order_by=spatial_index_blocks,
        clamp=False,
    )
    patch_index = selected_flat_idx[:, None, :, None].expand(-1, F, -1, D)
    return flat_patches.gather(2, patch_index)


def _global_hard_select(patches, scores, k):
    B, F, N, D = patches.shape
    k = int(k)

    selected_idx = select_with_floor(
        scores.unsqueeze(1),
        k,
        stable=True,
        clamp=False,
    )
    patch_index = selected_idx[:, None, :, None].expand(-1, F, -1, D)
    return patches.gather(2, patch_index)


def _stable_masked_weights(score_blocks, mask, mode, temperature):
    mask_f = mask.to(dtype=torch.float32)
    if mode == "uniform" or temperature == 0:
        raw = mask_f
    else:
        shifted = float(temperature) * (
            score_blocks.float() - score_blocks.float().max(dim=-1, keepdim=True).values
        )
        raw = torch.exp(shifted) * mask_f
    return raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _top_representative_residual_blocks(
    patch_blocks,
    score_blocks,
    slots_per_block,
    mode,
    temperature,
    residual_scale,
    assignment,
    block_size,
):
    B, F, num_blocks, block_area, D = patch_blocks.shape
    slots = int(slots_per_block)

    rep_indices = torch.topk(score_blocks, k=slots, dim=-1).indices
    rep_indices = torch.sort(rep_indices, dim=-1).values
    patch_rep_index = rep_indices[:, None, :, :, None].expand(-1, F, -1, -1, D)
    rep_patches = patch_blocks.gather(3, patch_rep_index)

    if slots == block_area or residual_scale == 0:
        return rep_patches

    selected_mask = torch.zeros(
        B, num_blocks, block_area, dtype=torch.bool, device=patch_blocks.device
    )
    selected_mask.scatter_(dim=-1, index=rep_indices, value=True)
    unselected_mask = ~selected_mask

    if assignment == "similarity":
        mean_features = patch_blocks.float().mean(dim=1)
        rep_mean = mean_features.gather(
            2, rep_indices[:, :, :, None].expand(-1, -1, -1, D)
        )
        mean_norm = nnf.normalize(mean_features, dim=-1)
        rep_norm = nnf.normalize(rep_mean, dim=-1)
        assign_scores = torch.einsum("bmad,bmsd->bmas", mean_norm, rep_norm)
        assignment_idx = assign_scores.argmax(dim=-1)
    elif assignment == "spatial":
        local = torch.arange(block_area, device=patch_blocks.device)
        coords = torch.stack(
            (local // int(block_size), local % int(block_size)), dim=-1
        ).float()
        rep_coords = coords[rep_indices]
        distances = (
            (coords.view(1, 1, block_area, 1, 2) - rep_coords[:, :, None, :, :]) ** 2
        ).sum(dim=-1)
        assignment_idx = distances.argmin(dim=-1)
    else:
        raise RuntimeError(f"assignment must be similarity or spatial, got {assignment}.")

    residual_slots = []
    for slot_idx in range(slots):
        assigned_mask = unselected_mask & (assignment_idx == slot_idx)
        weights = _stable_masked_weights(
            score_blocks, assigned_mask, mode, temperature
        ).to(dtype=patch_blocks.dtype)
        representative = rep_patches[:, :, :, slot_idx, :]
        residual = (
            (patch_blocks - representative[:, :, :, None, :])
            * weights[:, None, :, :, None]
        ).sum(dim=3)
        residual_slots.append(representative + float(residual_scale) * residual)
    return torch.stack(residual_slots, dim=3)


def fold_spatial_blocks(
    patches,
    patch_width,
    block_size=2,
    mode="uniform",
    temperature=1.0,
    slots_per_block=1,
    pooling="weighted",
    residual_scale=0.2,
    assignment="similarity",
    min_slots_per_block=1,
    score_alpha=0.5,
    score_normalization="auto",
    temporal_feature_norm="layernorm",
    temporal_spatial_normalization="none",
    block_wise=True,
    global_k=None,
    hevc_scores=None,
):
    """Fold ``[B, F, N, D]`` patches with one spatial map per clip."""

    batch_size, frames, num_patches, hidden_dim = patches.shape
    scores = _clip_level_scores(
        patches,
        mode,
        score_alpha=score_alpha,
        score_normalization=score_normalization,
        temporal_feature_norm=temporal_feature_norm,
        temporal_spatial_normalization=temporal_spatial_normalization,
        hevc_scores=hevc_scores,
    )

    if not block_wise:
        folded = _global_hard_select(patches, scores, global_k)
        return folded, folded.shape[2]

    block_size = int(block_size)
    block_area = block_size**2
    slots_per_block = int(slots_per_block)
    min_slots_per_block = int(min_slots_per_block)

    patch_width = int(patch_width)
    patch_height = num_patches // patch_width
    patch_blocks = _spatial_blocks(
        patches.reshape(batch_size * frames, num_patches, hidden_dim),
        patch_height,
        patch_width,
        block_size,
    ).reshape(batch_size, frames, -1, block_area, hidden_dim)
    score_blocks = _spatial_blocks(
        scores.unsqueeze(-1), patch_height, patch_width, block_size
    ).squeeze(-1)

    if pooling == "coverage-hard":
        spatial_index_blocks = _spatial_blocks(
            torch.arange(num_patches, device=patches.device).view(1, num_patches, 1),
            patch_height,
            patch_width,
            block_size,
        ).squeeze(-1)
        folded = _coverage_hard_blocks(
            patch_blocks,
            score_blocks,
            spatial_index_blocks,
            slots_per_block,
            min_slots_per_block,
        )
    elif pooling == "representative":
        folded = _top_representative_residual_blocks(
            patch_blocks,
            score_blocks,
            slots_per_block,
            mode,
            temperature,
            residual_scale,
            assignment,
            block_size,
        )
    elif slots_per_block == 1:
        folded = _weighted_pool_slot_group(
            patch_blocks,
            score_blocks,
            torch.arange(block_area, device=patches.device),
            mode,
            temperature,
        )
    else:
        folded = torch.stack(
            [
                _weighted_pool_slot_group(
                    patch_blocks, score_blocks, group_idx, mode, temperature
                )
                for group_idx in _uniform_slot_groups(
                    block_area, slots_per_block, patches.device
                )
            ],
            dim=3,
        )

    folded = folded.reshape(batch_size, frames, -1, hidden_dim)
    return folded, folded.shape[2]


def resolve_shared_folding_keep_count(k_keep_rate, total_columns):
    return int(round(int(total_columns) * float(k_keep_rate)))


def resolve_shared_folding_options(options, *, total_columns, k_keep_rate):
    """Resolve grid-dependent options and the effective kept-column budget."""

    if not isinstance(options, SharedFoldingOptions):
        options = SharedFoldingOptions(**dict(options))

    total_columns = int(total_columns)
    if options.block_wise:
        block_size = int(options.block_size)
        slots_per_block = int(options.slots_per_block)
        block_area = block_size**2
        keep_columns = total_columns // block_area * slots_per_block
        resolved = replace(options, global_k=None)
    else:
        keep_columns = resolve_shared_folding_keep_count(
            k_keep_rate,
            total_columns,
        )
        resolved = replace(
            options,
            block_wise=False,
            global_k=keep_columns,
        )
    return resolved, keep_columns
