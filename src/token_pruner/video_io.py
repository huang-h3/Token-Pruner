"""Frame sampling and decoding."""

from functools import lru_cache
import os
from pathlib import Path
import zlib

import decord
import numpy as np

VIDEO_ABLATION_ENV = "VIDEO_ABLATION"
VIDEO_ABLATIONS = ("none", "shuffle", "static", "noise", "black")
_VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".webm", ".mov")
_DECORD_BATCH_SIZE = 16


@lru_cache(maxsize=32)
def _sibling_videos(directory):
    return tuple(sorted(
        str(path) for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
    ))


def _donor_path(video_path):
    """Another real video, keyed on this one's path."""

    others = [
        path for path in _sibling_videos(str(Path(video_path).parent))
        if path != str(video_path)
    ]
    if not others:
        return None
    return others[zlib.crc32(str(video_path).encode()) % len(others)]


def _apply_ablation(frames, mode, video_path):
    if mode == "static":
        return np.repeat(frames[:1], frames.shape[0], axis=0)
    if mode == "black":
        return np.zeros_like(frames)
    rng = np.random.default_rng(zlib.crc32(str(video_path).encode()))
    return rng.integers(0, 256, size=frames.shape, dtype=np.uint8)


def uniform_frame_indices(total_frames, num_frames, *, strategy="linspace"):
    total_frames = int(total_frames)
    num_frames = int(num_frames)
    if strategy == "linspace":
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    segment = float(total_frames - 1) / num_frames
    return [int(segment * index + segment / 2) for index in range(num_frames)]


def _read_frames(video_path, indices):
    """Long random access, one reader per batch to bound decoder state."""

    frames = None
    for start in range(0, len(indices), _DECORD_BATCH_SIZE):
        reader = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        batch = reader.get_batch(indices[start:start + _DECORD_BATCH_SIZE]).asnumpy()
        if frames is None:
            frames = np.empty((len(indices), *batch.shape[1:]), dtype=batch.dtype)
        frames[start:start + len(batch)] = batch
    return frames


def sample_video_uniform(video_path, num_frames, *, strategy="linspace", with_fps=False):
    mode = os.environ.get(VIDEO_ABLATION_ENV, "none").strip().lower() or "none"
    read_path = _donor_path(video_path) if mode == "shuffle" else str(Path(video_path))
    reader = decord.VideoReader(read_path, ctx=decord.cpu(0), num_threads=1)
    indices = uniform_frame_indices(len(reader), num_frames, strategy=strategy)
    fps = float(reader.get_avg_fps()) if with_fps else None
    if len(indices) <= _DECORD_BATCH_SIZE:
        frames = reader.get_batch(indices).asnumpy()
    else:
        del reader
        frames = _read_frames(read_path, indices)
    if mode not in {"none", "shuffle"}:
        frames = _apply_ablation(frames, mode, video_path)
    if with_fps:
        return frames, indices, fps
    return frames, indices
