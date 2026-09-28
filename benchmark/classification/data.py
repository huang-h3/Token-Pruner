"""Dataset preparation helpers for classification benchmarks."""

from pathlib import Path

from token_pruner.video_io import sample_video_uniform

from .datasets import collate_video_batch


DEFAULT_DATASET = "k400"
MIN_VALID_VIDEO_BYTES = 1024


def read_rgb_frames(video_path, num_frames):
    return sample_video_uniform(
        video_path,
        num_frames,
        strategy="segment_midpoint",
    )


def append_skipped_video(skipped_videos_path, video_name):
    if skipped_videos_path is None:
        return
    skipped_videos_path = Path(skipped_videos_path)
    skipped_videos_path.parent.mkdir(parents=True, exist_ok=True)
    with skipped_videos_path.open("a") as output:
        output.write(f"{video_name}\n")


def build_filtered_loader(dataset, batch_size, skipped_videos_path=None):
    """Filter an instantiated dataset and build its deterministic loader."""
    from torch.utils.data import DataLoader

    print(f"Dataset candidates selected: {len(dataset)}", flush=True)
    skipped_videos_path = (
        Path(skipped_videos_path) if skipped_videos_path is not None else None
    )
    print(
        f"Filtering selected videos by file size: {len(dataset)} candidates...",
        flush=True,
    )
    total = len(dataset)
    valid_samples = []
    skipped_videos = list(dataset.skipped_videos)
    for sample in dataset.samples:
        video_path = Path(sample["video_path"])
        if video_path.stat().st_size < MIN_VALID_VIDEO_BYTES:
            skipped_videos.append(video_path.name)
            continue
        valid_samples.append(sample)
    dataset.samples = valid_samples
    dataset.skipped_videos = skipped_videos
    filter_stats = {
        "total": total,
        "kept": len(valid_samples),
        "skipped": total - len(valid_samples),
    }
    print(
        f"Filtered dataset: kept={filter_stats['kept']} "
        f"skipped={filter_stats['skipped']} "
        f"total={filter_stats['total']}",
        flush=True,
    )
    if not dataset:
        raise RuntimeError("No valid videos available after dataset filtering.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_video_batch,
    )
    return filter_stats, loader, skipped_videos_path
