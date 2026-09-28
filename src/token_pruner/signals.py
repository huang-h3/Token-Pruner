"""Signal options, per-clip and per-batch signal production, selector routing."""

from dataclasses import dataclass
from pathlib import Path
import time

from .measurements import cuda_sync
from .hevc_cache import HevcSelectionResult, prepare_hevc_batch_selection
from .selection import mode_semantics
from .selectors import (
    BaselineTokenSelector,
    build_spatial_patch_selector,
    spatial_patch_score_selector,
    video_patch_selector,
)
from .tokens import PruningConfig, PruningScope, SignalSource, TokenReducer


@dataclass(frozen=True)
class SignalOptions:
    """Encoding and selector knobs that are neither video nor pruning policy."""

    work_dir: Path | None = None
    permanent_dir: Path | None = None
    n_parallel: int = 6
    requested_gop_size: int = 0
    encode_scope: str = "sampled-clip"

    @property
    def full_video(self):
        """Full-video artifacts keep the caller's frame indices; clips renumber them."""

        return str(self.encode_scope) == "full-video"

    def selection_kwargs(self):
        kwargs = {
            "sampled_gop_size": self.requested_gop_size,
            "encode_scope": self.encode_scope,
        }
        if self.permanent_dir is not None:
            kwargs["hevc_permanent_dir"] = self.permanent_dir
        return kwargs

    @classmethod
    def from_session(cls, session):
        """Read the HEVC options a pruning session was configured with."""

        return cls(
            work_dir=session.hevc_dir,
            permanent_dir=session.hevc_permanent_dir,
            n_parallel=int(session.hevc_n_parallel),
            requested_gop_size=session.hevc_gop_size,
            encode_scope=session.hevc_encode_scope,
        )


def _direct_signal(provider, videos, *, device, kept_indices):
    """Selector-only path: no encode, so the whole cost is selector compute."""

    start = time.perf_counter()
    signal = provider.select(videos, device=device)
    cuda_sync(device)
    return HevcSelectionResult(
        selector_outputs=signal,
        selector_compute_ms=(time.perf_counter() - start) * 1000.0,
        kept_indices=kept_indices,
    )


def _selector_indices(frame_indices, sampled_frames, options):
    return list(frame_indices) if options.full_video else list(range(len(sampled_frames)))


def signal_for_batch(
    provider, video_paths, frame_indices, sampled_frames, *,
    options, device, needs_hevc,
):
    """Produce the signal for a batch of clips, one artifact per clip."""

    if provider is None:
        return HevcSelectionResult()
    if not needs_hevc:
        return _direct_signal(
            provider, [str(path) for path in video_paths], device=device,
            kept_indices=list(range(len(video_paths))),
        )

    if options.full_video and sampled_frames is None:
        sampled_frames = [None] * len(video_paths)
    selector_indices = [
        _selector_indices(indices, frames, options)
        for indices, frames in zip(frame_indices, sampled_frames)
    ]

    def selector_fn(paths, source_indices, cache_keys, score_cache_dir):
        return provider.select(
            [str(path) for path in paths],
            [selector_indices[index] for index in source_indices],
            n_parallel=options.n_parallel, device=device, cache_keys=cache_keys,
            score_cache_dir=score_cache_dir,
        )

    result = prepare_hevc_batch_selection(
        video_paths, options.work_dir, selector_fn, device,
        sampled_frames=None if options.full_video else sampled_frames,
        **options.selection_kwargs(),
    )
    if result.failed_indices:
        raise RuntimeError(
            f"HEVC encoding failed for batch indices {result.failed_indices}."
        )
    return result


def build_signal_provider(
    *,
    config: PruningConfig,
    patch_size,
    num_frames,
    image_size,
    score_reduce="max",
    selection_seed=42,
    selector_id="patch",
    hevc_gop_size=0,
    hevc_anchor_policy="first",
):
    routed_p_mode = (
        "shared_folding"
        if config.reducer == TokenReducer.SHARED_FOLDING
        else (
            config.signal_source.value
            if config.signal_source in {
                SignalSource.HEVC,
                SignalSource.RANDOM,
                SignalSource.UNIFORM,
            }
            else "folder"
        )
    )
    if config.scope == PruningScope.SHARED and routed_p_mode == "hevc":
        routed_p_mode = "shared_hevc"
    i_mode = (
        routed_p_mode
        if config.scope == PruningScope.SHARED
        else ("preserve" if config.anchor_reducer == TokenReducer.PRESERVE
              else "folder")
    )
    semantics = mode_semantics(
        config.scope.value,
        i_mode,
        routed_p_mode,
        uniform_partitions=bool(config.uniform_partitions),
        selector_id=selector_id,
    )
    if semantics.selector == "baseline":
        return BaselineTokenSelector(
            config.signal_source.value,
            total_tokens=(
                int(num_frames) * (int(image_size) // int(patch_size)) ** 2
            ),
            keep_total=config.keep_total,
            num_partitions=num_frames,
            seed=selection_seed,
            local=config.scope == PruningScope.LOCAL,
            keep_per_partition=(
                config.keep_per_partition
                if config.scope == PruningScope.LOCAL
                and config.uniform_partitions
                else None
            ),
            min_per_partition=config.min_per_partition,
        )
    image_shape = (
        1,
        3,
        int(num_frames),
        int(image_size),
        int(image_size),
    )
    if semantics.selector == "patch_scores":
        return spatial_patch_score_selector(
            patch_size=patch_size,
            image_shape=image_shape,
            hevc_gop_size=hevc_gop_size,
            hevc_anchor_policy=hevc_anchor_policy,
        )
    if semantics.selector == "clip_topk":
        return video_patch_selector(
            patch_size=patch_size,
            keep_patches=config.keep_per_partition,
            total_keep_patches=config.keep_total,
            num_frames=num_frames,
            image_size=image_size,
            prune_mode=config.scope.value,
            i_mode=i_mode,
            hevc_gop_size=hevc_gop_size,
            hevc_anchor_policy=hevc_anchor_policy,
            min_per_partition=config.min_per_partition,
        )
    if semantics.selector in {"framewise_topk", "shared_map"}:
        return build_spatial_patch_selector(
            prune_mode=(
                "local" if semantics.selector == "framewise_topk" else "shared"
            ),
            patch_size=patch_size,
            keep_patches=config.keep_per_partition,
            num_frames=num_frames,
            image_size=image_size,
            p_mode=routed_p_mode,
            score_reduce=score_reduce,
            hevc_gop_size=hevc_gop_size,
            hevc_anchor_policy=hevc_anchor_policy,
        )
    raise RuntimeError(
        f"No selector route for {config.scope.value}/{i_mode}/"
        f"{config.signal_source.value}: {semantics}"
    )
