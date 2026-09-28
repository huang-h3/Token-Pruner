"""Reusable token selectors for HEVC and baseline pruning signals."""

import os
import zlib

import cv2
import numpy as np
import torch

from .hevc import open_hevc_feature_reader, resolve_sampled_gop_size
from .hevc_cache import HevcScoreStore
from .selection import distribute_budget, select_with_floor
from .tokens import PruningSignal


def as_clip_batch(video_paths, frame_indices):
    """Normalize one clip or a batch to aligned lists."""
    if isinstance(video_paths, (str, bytes, os.PathLike)):
        return [video_paths], [frame_indices], True
    paths = list(video_paths)
    indices = list(frame_indices)
    return paths, indices, False


def as_cache_key_batch(cache_keys, batch_size):
    if cache_keys is None:
        return [None] * int(batch_size)
    if isinstance(cache_keys, (str, bytes)):
        cache_keys = [cache_keys]
    else:
        cache_keys = list(cache_keys)
    assert len(cache_keys) == int(batch_size), "one cache key per video"
    return [None if key is None else str(key) for key in cache_keys]


def select_group_topk(
    scores,
    groups,
    budget,
    items_per_group,
    prune_mode,
    *,
    clamp_to_group=False,
    floor=0,
):
    """Pick ``budget`` flattened indices from ``scores`` restricted to ``groups``."""
    groups = list(groups)
    budget = int(budget)
    items_per_group = int(items_per_group)
    if not groups or budget <= 0:
        return torch.empty(0, dtype=torch.long)

    if prune_mode == "global":
        return select_with_floor(
            scores[groups],
            budget,
            block_ids=groups,
            floor=floor,
            ceiling=None,
            clamp=clamp_to_group,
        )

    shares = distribute_budget(budget, groups)
    if any(shares[group] > items_per_group for group in groups) and not clamp_to_group:
        keep = max(shares[group] for group in groups)
        raise RuntimeError(
            f"Local budget {keep} exceeds the {items_per_group} available items."
        )
    return select_with_floor(
        scores[groups],
        budget,
        block_ids=groups,
        floor=[shares[group] for group in groups],
        ceiling=[shares[group] for group in groups],
        clamp=clamp_to_group,
    )


class BaselineTokenSelector:
    """Sample flat token indices globally or within optional partitions."""

    def __init__(
        self,
        mode,
        total_tokens,
        keep_total,
        num_partitions,
        seed=42,
        keep_per_partition=None,
        local=False,
        min_per_partition=0,
    ):
        self.mode = str(mode)
        self.total_tokens = int(total_tokens)
        self.keep_total = int(keep_total)
        self.num_partitions = int(num_partitions)
        self.seed = int(seed)
        self.tokens_per_partition = self.total_tokens // self.num_partitions
        self.local = bool(local or keep_per_partition is not None)
        self.min_per_partition = int(min_per_partition)
        self.keep_per_partition = (
            int(keep_per_partition) if keep_per_partition is not None else None
        )

    def _partition_allocations(self):
        partitions = list(range(self.num_partitions))
        if self.keep_per_partition is not None:
            return {
                partition: self.keep_per_partition for partition in partitions
            }
        return distribute_budget(
            self.keep_total,
            partitions,
            capacities={
                partition: self.tokens_per_partition for partition in partitions
            },
        )

    def _generator(self, *parts):
        identity = ":".join(str(part) for part in (self.seed, *parts)).encode("utf-8")
        seed = zlib.crc32(identity)
        return torch.Generator(device="cpu").manual_seed(seed)

    @staticmethod
    def _even_indices(keep, total):
        positions = torch.arange(int(keep), dtype=torch.long)
        return ((2 * positions + 1) * int(total)) // (2 * int(keep))

    def _random_indices(self, sample_id):
        return torch.randperm(
            self.total_tokens,
            generator=self._generator(sample_id),
        )[: self.keep_total].sort().values

    def _uniform_indices(self):
        return self._even_indices(self.keep_total, self.total_tokens)

    def _minimum_uniform_indices(self):
        selected = self._uniform_indices()
        counts = torch.bincount(
            selected // self.tokens_per_partition,
            minlength=self.num_partitions,
        )
        if bool((counts >= self.min_per_partition).all()):
            return selected
        forced = torch.cat([
            self._even_indices(self.min_per_partition, self.tokens_per_partition)
            + partition * self.tokens_per_partition
            for partition in range(self.num_partitions)
        ])
        forced = set(int(value) for value in forced.tolist())
        ordered = [int(value) for value in selected.tolist() if int(value) not in forced]
        ordered = list(forced) + ordered[: self.keep_total - len(forced)]
        return torch.tensor(sorted(ordered), dtype=torch.long)

    def _partition_indices(self, sample_id=None):
        parts = []
        for partition, keep in self._partition_allocations().items():
            if self.mode == "random":
                local = torch.randperm(
                    self.tokens_per_partition,
                    generator=self._generator(sample_id, partition),
                )[:keep].sort().values
            else:
                local = self._even_indices(keep, self.tokens_per_partition)
            parts.append(local + partition * self.tokens_per_partition)
        return torch.cat(parts).sort().values

    def select(self, sample_ids, *_args, device=None, **_kwargs):
        # HEVC-selector signature so the provider does not branch; only ids used.
        if isinstance(sample_ids, (str, bytes, os.PathLike)):
            sample_ids = [sample_ids]
        sample_ids = [str(sample_id) for sample_id in sample_ids]
        if self.local:
            visible_indices = torch.stack(
                [self._partition_indices(sample_id) for sample_id in sample_ids]
            )
        elif self.mode == "random":
            visible_indices = torch.stack(
                [
                    (
                        self._partition_indices(sample_id)
                        if self.min_per_partition
                        else self._random_indices(sample_id)
                    )
                    for sample_id in sample_ids
                ]
            )
        else:
            indices = (
                self._minimum_uniform_indices()
                if self.min_per_partition
                else self._uniform_indices()
            )
            visible_indices = indices.unsqueeze(0).expand(
                len(sample_ids), *indices.shape
            ).clone()
        return PruningSignal(
            visible_indices=visible_indices,
            anchor_partitions=torch.zeros(
                len(sample_ids),
                self.num_partitions,
                dtype=torch.bool,
            ),
        ).to(device)


def video_anchor_mask(num_frames, anchor_frames):
    mask = torch.zeros(int(num_frames), dtype=torch.bool)
    for frame_idx in anchor_frames or []:
        frame_idx = int(frame_idx)
        if 0 <= frame_idx < num_frames:
            mask[frame_idx] = True
    if num_frames and not mask.any():
        mask[0] = True
    return mask


def anchor_temporal_groups(anchor_frames, num_frames, group_size, temporal_groups):
    groups = set()
    for frame_idx in anchor_frames or []:
        frame_idx = int(frame_idx)
        if 0 <= frame_idx < num_frames:
            groups.add(min(frame_idx // int(group_size), temporal_groups - 1))
    if not groups:
        groups.add(0)
    return groups


def spatial_patch_scores(saliency, patch_size, device=None):
    """Aggregate pixel saliency into flattened ``[frames, patches]`` scores."""
    frames, height, width = saliency.shape
    patch_size = int(patch_size)
    patch_h = height // patch_size
    patch_w = width // patch_size
    return (
        torch.as_tensor(saliency, dtype=torch.float32, device=device)
        .reshape(frames, patch_h, patch_size, patch_w, patch_size)
        .sum(dim=(2, 4))
        .reshape(frames, patch_h * patch_w)
    )


def framewise_spatial_indices_from_scores(
    scores, keep_patches, anchor_frames=None
):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    num_frames, total_patches = scores.shape
    keep = min(int(keep_patches), total_patches)
    anchors = video_anchor_mask(num_frames, anchor_frames)
    visible = torch.zeros(num_frames, keep, dtype=torch.long)
    candidates = [frame_idx for frame_idx in range(num_frames) if not anchors[frame_idx]]
    if candidates:
        selected = select_with_floor(
            scores[candidates],
            keep * len(candidates),
            block_ids=candidates,
            floor=keep,
            ceiling=keep,
            clamp=False,
        )
        for position, frame_idx in enumerate(candidates):
            start = position * keep
            visible[frame_idx] = selected[start : start + keep].remainder(total_patches)
    return anchors, visible


def shared_spatial_indices_from_scores(
    scores,
    keep_patches,
    reduce="max",
    p_mode="hevc",
    anchor_frames=None,
):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    num_frames, total_patches = scores.shape
    keep = min(int(keep_patches), total_patches)
    if keep <= 0:
        return torch.empty(0, dtype=torch.long)
    if keep == total_patches:
        return torch.arange(total_patches, dtype=torch.long)
    anchors = video_anchor_mask(num_frames, anchor_frames)
    candidates = (
        ~anchors if p_mode in {"hevc", "shared_hevc"} else torch.zeros_like(anchors)
    )
    if not candidates.any():
        return torch.linspace(0, total_patches - 1, steps=keep).round().long()
    frame_scores = scores[candidates]
    reducers = {
        "max": lambda value: value.max(dim=0).values,
        "mean": lambda value: value.mean(dim=0),
        "sum": lambda value: value.sum(dim=0),
    }
    return torch.topk(reducers[reduce](frame_scores), k=keep).indices.sort().values


def _extract_hevc_patch_scores(
    video_path,
    frame_indices,
    target_height,
    target_width,
    patch_size,
    n_parallel=1,
    *,
    expected_gop_size=None,
    hevc_anchor_policy="first",
    cache_key=None,
    score_cache_dir=None,
):
    frame_indices = [int(index) for index in frame_indices]
    anchor_policy = str(hevc_anchor_policy)

    def compute():
        saliency, anchors = _extract_hevc_saliency(
            video_path,
            frame_indices,
            target_height,
            target_width,
            n_parallel,
            expected_gop_size=expected_gop_size,
            hevc_anchor_policy=anchor_policy,
        )
        return spatial_patch_scores(saliency, patch_size), anchors

    cached = HevcScoreStore(score_cache_dir).get_or_compute(
        artifact_key=cache_key,
        frame_indices=frame_indices,
        patch_size=patch_size,
        target_height=target_height,
        target_width=target_width,
        expected_gop_size=expected_gop_size,
        anchor_policy=anchor_policy,
        compute_fn=compute,
    )
    return cached.scores, cached.anchors


class HevcPatchSelector:
    """Score a clip's patches from HEVC; ``reduce`` maps (scores, anchors) to a signal."""

    def __init__(self, patch_size, target_hw, reduce, *, hevc_gop_size=0,
                 hevc_anchor_policy="first", consumes_anchors=True,
                 ragged_fields=()):
        self.patch_size = int(patch_size)
        self.target_h, self.target_w = (int(value) for value in target_hw)
        self.reduce = reduce
        self.hevc_gop_size = hevc_gop_size
        self.hevc_anchor_policy = (
            str(hevc_anchor_policy) if consumes_anchors else "none"
        )
        #: Fields the config does not fix per clip; always a list.
        self.ragged_fields = frozenset(ragged_fields)

    def clip_scores(
        self, video_path, frame_indices, n_parallel=1, cache_key=None,
        score_cache_dir=None,
    ):
        return _extract_hevc_patch_scores(
            video_path, frame_indices, self.target_h, self.target_w,
            self.patch_size, n_parallel,
            expected_gop_size=self.hevc_gop_size,
            hevc_anchor_policy=self.hevc_anchor_policy,
            cache_key=cache_key,
            score_cache_dir=score_cache_dir,
        )

    def select(self, video_paths, frame_indices, n_parallel=1, device=None,
               cache_keys=None, score_cache_dir=None):
        paths, indices_batch, single = as_clip_batch(video_paths, frame_indices)
        keys = as_cache_key_batch(cache_keys, len(paths))
        fields = [
            self.reduce(*self.clip_scores(
                path, indices, n_parallel, key, score_cache_dir
            ))
            for path, indices, key in zip(paths, indices_batch, keys)
        ]
        signal = {}
        for name in ("visible_indices", "anchor_partitions", "scores"):
            values = [row[name] for row in fields if row.get(name) is not None]
            if len(values) != len(fields):
                continue
            widths = {tuple(value.shape) for value in values}
            # Single-clip answers with batch axis 1; only ragged batches stay lists.
            ragged = not single and name in self.ragged_fields
            uniform = len(widths) == 1 and not ragged
            signal[name] = torch.stack(values) if uniform else list(values)
        return PruningSignal(**signal).to(device)


def patch_selector(patch_size, patch_budget, image_shape,
                   hevc_gop_size=0, hevc_anchor_policy="first"):
    """A fixed budget inside every frame, anchors kept whole."""

    patch_size = int(patch_size)
    budget = int(patch_budget)
    _, _, _, height, width = image_shape
    tokens_per_partition = (height // patch_size) * (width // patch_size)

    def reduce(scores, anchors):
        anchor_mask, per_frame = framewise_spatial_indices_from_scores(
            scores, budget, anchors)
        # Per-frame indices are frame-local; the engine wants clip-flat ones.
        offsets = torch.arange(per_frame.shape[0]).unsqueeze(1)
        return {
            "anchor_partitions": anchor_mask,
            "visible_indices": (
                per_frame + offsets * tokens_per_partition).flatten(),
        }

    return HevcPatchSelector(
        patch_size, (height, width), reduce, hevc_gop_size=hevc_gop_size,
        hevc_anchor_policy=hevc_anchor_policy,
    )


def shared_spatial_patch_selector(patch_size, patch_budget, image_shape,
                                  p_mode="hevc", score_reduce="max",
                                  hevc_gop_size=0, hevc_anchor_policy="first"):
    """One spatial map reused by every frame."""

    _, _, _, height, width = image_shape
    budget = int(patch_budget)

    def reduce(scores, anchors):
        return {
            "visible_indices": shared_spatial_indices_from_scores(
                scores, budget, score_reduce, p_mode, anchors,
            )
        }

    return HevcPatchSelector(
        patch_size, (height, width), reduce, hevc_gop_size=hevc_gop_size,
        hevc_anchor_policy=hevc_anchor_policy,
        consumes_anchors=False,
    )


def spatial_patch_score_selector(patch_size, image_shape, hevc_gop_size=0,
                                 hevc_anchor_policy="first"):
    """Dense ``[B, F, N]`` patch scores; shared folding does its own selection."""

    _, _, _, height, width = image_shape
    return HevcPatchSelector(
        patch_size, (height, width), lambda scores, anchors: {"scores": scores},
        hevc_gop_size=hevc_gop_size, hevc_anchor_policy=hevc_anchor_policy,
        consumes_anchors=False,
    )


def build_spatial_patch_selector(
    *,
    prune_mode,
    patch_size,
    keep_patches,
    num_frames,
    image_size,
    p_mode="hevc",
    score_reduce="max",
    hevc_gop_size=0,
    hevc_anchor_policy="first",
):
    """Build the HEVC selector shared by classification and VLM integrations."""
    common_kwargs = {
        "patch_size": int(patch_size),
        "patch_budget": int(keep_patches),
        "image_shape": (
            1,
            3,
            int(num_frames),
            int(image_size),
            int(image_size),
        ),
        "hevc_gop_size": hevc_gop_size,
        "hevc_anchor_policy": hevc_anchor_policy,
    }
    if prune_mode == "local":
        return patch_selector(**common_kwargs)
    return shared_spatial_patch_selector(
        **common_kwargs,
        p_mode=p_mode,
        score_reduce=score_reduce,
    )


def video_patch_selector(patch_size, keep_patches, total_keep_patches,
                         num_frames, image_size, prune_mode, i_mode,
                         hevc_gop_size=0, hevc_anchor_policy="first",
                         min_per_partition=0):
    """Clip-wide budget: anchors keep their share, the rest compete globally."""

    keep_patches = int(keep_patches)
    total_keep_patches = int(total_keep_patches)
    num_frames = int(num_frames)

    def reduce(scores, anchor_frames):
        frames, num_patches = scores.shape
        anchors = video_anchor_mask(frames, anchor_frames)
        p_frames = torch.where(~anchors)[0].tolist()
        anchor_keep = num_patches if i_mode == "preserve" else keep_patches
        p_budget = min(
            max(total_keep_patches - int(anchors.sum()) * anchor_keep, 0),
            len(p_frames) * num_patches,
        )
        return {
            "anchor_partitions": anchors,
            "visible_indices": select_group_topk(
                scores, p_frames, p_budget, num_patches, prune_mode,
                floor=int(min_per_partition)),
        }

    return HevcPatchSelector(
        patch_size, (image_size, image_size), reduce,
        hevc_gop_size=hevc_gop_size, hevc_anchor_policy=hevc_anchor_policy,
        ragged_fields=("visible_indices",),
    )


class TubeletSelector:
    """Select ViViT tubelets from HEVC saliency."""

    def __init__(
        self,
        patch_size,
        tubelet_size,
        tubelet_budget,
        image_shape,
        prune_mode="global",
        preserve_anchor_tubelets=False,
        total_tubelet_budget=None,
        min_per_group=0,
        hevc_gop_size=0,
        hevc_anchor_policy="first",
    ):
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.tubelet_budget = int(tubelet_budget)
        _, _, _, self.target_height, self.target_width = image_shape
        self.num_frames = int(image_shape[2])
        self.prune_mode = prune_mode
        self.preserve_anchor_tubelets = bool(preserve_anchor_tubelets)
        self.total_tubelet_budget = (
            int(total_tubelet_budget) if total_tubelet_budget is not None else None
        )
        self.hevc_gop_size = hevc_gop_size
        self.hevc_anchor_policy = str(hevc_anchor_policy)
        self.min_per_group = int(min_per_group)

    def gen_visible_indices_anchors(
        self, video_paths, frame_indices, n_parallel=1, cache_keys=None,
        score_cache_dir=None,
    ):
        paths, indices_batch, single = as_clip_batch(video_paths, frame_indices)
        keys = as_cache_key_batch(cache_keys, len(paths))
        results = [
            self._clip_tubelets(
                path, indices, n_parallel, cache_key, score_cache_dir,
            )
            for path, indices, cache_key in zip(paths, indices_batch, keys)
        ]
        if single:
            return results[0]
        visible, anchors = zip(*results)
        if len({item.shape[1] for item in visible}) != 1:
            return [item.squeeze(0) for item in visible], list(anchors)
        return torch.cat(visible), list(anchors)

    def select(
        self,
        video_paths,
        frame_indices,
        n_parallel=1,
        device=None,
        cache_keys=None,
        score_cache_dir=None,
    ):
        visible_indices, anchor_frames = self.gen_visible_indices_anchors(
            video_paths,
            frame_indices,
            n_parallel=n_parallel,
            cache_keys=cache_keys,
            score_cache_dir=score_cache_dir,
        )
        if isinstance(anchor_frames, torch.Tensor):
            anchor_frames = anchor_frames.tolist()
        batch_size = (
            int(visible_indices.shape[0])
            if isinstance(visible_indices, torch.Tensor)
            else len(visible_indices)
        )
        if batch_size == 1 and (not anchor_frames or isinstance(anchor_frames[0], int)):
            anchor_frames = [anchor_frames]
        temporal_groups = (
            self.num_frames + self.tubelet_size - 1
        ) // self.tubelet_size
        anchor_partitions = torch.zeros(
            batch_size,
            temporal_groups,
            dtype=torch.bool,
        )
        if self.prune_mode == "shared":
            anchor_frames = [[] for _ in range(batch_size)]
        for batch_idx, frames in enumerate(anchor_frames):
            frames = [int(frame_idx) for frame_idx in frames]
            groups = anchor_temporal_groups(
                frames,
                self.num_frames,
                self.tubelet_size,
                temporal_groups,
            )
            anchor_partitions[batch_idx, list(groups)] = True
        return PruningSignal(
            visible_indices=visible_indices,
            anchor_partitions=anchor_partitions,
        ).to(device)

    def tubelet_score_cal(
        self,
        frame_scores,
        keep_tubelets,
        tubelet_size,
        prune_mode="global",
        anchor_frames=None,
        preserve_anchor_tubelets=False,
        total_keep_tubelets=None,
    ):
        frame_scores = torch.as_tensor(frame_scores, dtype=torch.float32)
        source_frames = int(frame_scores.shape[0])
        tubelet_size = int(tubelet_size)
        padding = (-source_frames) % tubelet_size
        if padding:
            frame_scores = torch.cat(
                (frame_scores, frame_scores[-1:].expand(padding, -1)), dim=0
            )
        temporal_groups = int(frame_scores.shape[0]) // tubelet_size
        scores = (
            frame_scores.reshape(temporal_groups, tubelet_size, -1)
            .max(dim=1)
            .values
        )
        patches_per_group = scores.shape[1]
        total = temporal_groups * patches_per_group
        keep_tubelets = int(keep_tubelets)

        if prune_mode == "shared":
            aggregate = scores.max(dim=0).values
            return select_group_topk(
                aggregate.unsqueeze(0),
                [0],
                keep_tubelets,
                patches_per_group,
                "global",
                clamp_to_group=True,
                floor=self.min_per_group,
            )

        target = total_keep_tubelets
        if target is None:
            target = (
                keep_tubelets
                if prune_mode == "global"
                else keep_tubelets * temporal_groups
            )
        target = min(max(int(target), 0), total)
        anchor_groups = anchor_temporal_groups(
            anchor_frames, source_frames, tubelet_size, temporal_groups
        )
        groups = [idx for idx in range(temporal_groups) if idx not in anchor_groups]
        anchor_cost = (
            patches_per_group if preserve_anchor_tubelets else keep_tubelets
        )
        budget = min(
            max(target - len(anchor_groups) * anchor_cost, 0),
            len(groups) * patches_per_group,
        )
        return select_group_topk(
            scores,
            groups,
            budget,
            patches_per_group,
            prune_mode,
            clamp_to_group=True,
            floor=self.min_per_group,
        )

    def _clip_tubelets(
        self,
        video_path,
        frame_indices,
        n_parallel=1,
        cache_key=None,
        score_cache_dir=None,
    ):
        frame_scores, anchors = _extract_hevc_patch_scores(
            video_path,
            frame_indices,
            self.target_height,
            self.target_width,
            self.patch_size,
            n_parallel,
            expected_gop_size=self.hevc_gop_size,
            hevc_anchor_policy=self.hevc_anchor_policy,
            cache_key=cache_key,
            score_cache_dir=score_cache_dir,
        )
        visible = self.tubelet_score_cal(
            frame_scores,
            self.tubelet_budget,
            self.tubelet_size,
            self.prune_mode,
            anchors,
            self.preserve_anchor_tubelets,
            self.total_tubelet_budget,
        )
        return visible.unsqueeze(0), anchors


def _extract_hevc_saliency(
    video_path,
    frame_indices,
    target_height,
    target_width,
    n_parallel=1,
    *,
    expected_gop_size=None,
    hevc_anchor_policy="first",
):
    frame_indices = [int(frame_idx) for frame_idx in frame_indices]
    if not frame_indices:
        return np.empty((0, target_height, target_width), dtype=np.float32), []
    reader = open_hevc_feature_reader(
        video_path, nb_frames=None, n_parallel=n_parallel
    )
    encoded_num_frames = int(getattr(reader, "nb_frames", 0) or 0)
    if encoded_num_frames > 0:
        frame_indices = [
            min(frame_idx, encoded_num_frames - 1) for frame_idx in frame_indices
        ]
    positions = {}
    for pos, frame_idx in enumerate(frame_indices):
        positions.setdefault(frame_idx, []).append(pos)
    saliency_frames = [None] * len(frame_indices)
    codec_idr_frames = set()
    protected_anchor_frames = set()
    last_fused = None
    decoded_num_frames = 0
    try:
        for frame_idx, frame_tuple in enumerate(reader.nextFrame()):
            decoded_num_frames = frame_idx + 1
            frame_type = int(frame_tuple[0])
            if frame_type == 0:
                codec_idr_frames.add(frame_idx)
            if frame_idx not in positions:
                continue
            (
                _frame_type,
                _quadtree,
                _rgb,
                mv_x_l0,
                mv_y_l0,
                _mv_x_l1,
                _mv_y_l1,
                _ref_l0,
                _ref_l1,
                _size,
                residual,
            ) = frame_tuple
            # Anchors come from the codec IDR frame type.
            is_codec_idr = frame_type == 0
            is_protected = (
                hevc_anchor_policy == "all" and is_codec_idr
            )
            if is_protected:
                protected_anchor_frames.update(positions[frame_idx])
            if (hevc_anchor_policy == "none" and frame_idx == 0) or is_protected:
                fused = np.zeros((reader.height, reader.width), dtype=np.float32)
            else:
                residual_y = (
                    cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]
                    if residual.ndim == 3
                    else residual
                )
                height, width = residual_y.shape
                mvx = reader._upsample_mv_to_hw(mv_x_l0.astype(np.float32))
                mvy = reader._upsample_mv_to_hw(mv_y_l0.astype(np.float32))
                residual_norm = _residual_energy_norm(residual_y)
                mv_norm = _mv_energy_norm(mvx, mvy, height, width)
                fused = fuse_energy(mv_norm, residual_norm)
            fused = cv2.resize(
                fused, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
            last_fused = fused
            for pos in positions[frame_idx]:
                saliency_frames[pos] = fused
    finally:
        reader.close()
    if decoded_num_frames <= 0:
        raise RuntimeError("Failed to decode video: no HEVC frames were produced.")
    if last_fused is None:
        last_fused = np.zeros((target_height, target_width), dtype=np.float32)
    saliency_frames = [
        last_fused.copy() if item is None else item for item in saliency_frames
    ]
    if hevc_anchor_policy == "first":
        saliency_frames[0] = np.zeros_like(saliency_frames[0])
    expected = {0}
    if expected_gop_size is not None:
        effective_gop_size = resolve_sampled_gop_size(
            decoded_num_frames, expected_gop_size
        )
        expected = set(range(0, decoded_num_frames, effective_gop_size))
    if codec_idr_frames != expected:
        if expected == {0}:
            detail = (
                "requires exactly one IDR at frame 0; "
                f"expected {sorted(expected)}, got {sorted(codec_idr_frames)}"
            )
        else:
            detail = (
                f"expected {sorted(expected)}, got {sorted(codec_idr_frames)}"
            )
        raise RuntimeError(f"HEVC codec IDR anchor contract {detail}.")
    if hevc_anchor_policy == "all":
        protected = sorted(protected_anchor_frames)
    else:
        protected = [0] if hevc_anchor_policy == "first" else []
    return np.stack(saliency_frames), protected


def fuse_energy(norm_mv, norm_res):
    return np.clip((norm_mv + norm_res) / 2.0, 0.0, 1.0).astype(np.float32)


def _residual_energy_norm(res_y, pct=95.0):
    values = np.abs(res_y.astype(np.float32) - 128.0)
    scale = max(float(np.percentile(values, pct)), 1.0)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def _mv_energy_norm(mvx, mvy, H, W, mv_unit_div=4.0, pct=95.0):
    vx = mvx.astype(np.float32) / float(mv_unit_div)
    vy = mvy.astype(np.float32) / float(mv_unit_div)
    magnitude = np.sqrt(vx * vx + vy * vy)
    scale = max(float(np.percentile(magnitude, pct)), 1e-6)
    normalized = cv2.resize(
        np.clip(magnitude / scale, 0.0, 1.0), (W, H), interpolation=cv2.INTER_NEAREST
    )
    return normalized.astype(np.float32)
