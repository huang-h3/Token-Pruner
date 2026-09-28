"""Bitstream and patch-score caches, and the selection they drive."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from tqdm import tqdm

from .hevc import (
    encode_hevc_for_inference,
    pick_encoder,
    probe_video_frame_count,
    resolve_hevc_encode_scope,
    resolve_sampled_gop_size,
)
from .measurements import cuda_sync
from .records import atomic_write_text

STORE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HevcRecipe:
    """Encoding settings that affect an artifact."""

    encode_scope: str = "sampled-clip"
    requested_encoder: str = ""
    sampled_gop_size: int = 0
    effective_gop_size: int | None = None
    crf: str = "23"
    hevc_min_dim: str = "128"
    skip_audio: str = "1"
    pix_fmt: str = "yuv420p"

    @classmethod
    def from_environment(
        cls,
        *,
        encode_scope="sampled-clip",
        requested_encoder=None,
        sampled_gop_size=0,
        effective_gop_size=None,
    ):
        return cls(
            encode_scope=str(encode_scope),
            requested_encoder=str(requested_encoder or pick_encoder()),
            sampled_gop_size=int(sampled_gop_size or 0),
            effective_gop_size=(
                None if effective_gop_size is None else int(effective_gop_size)
            ),
            crf=os.getenv("CRF", "23"),
            hevc_min_dim=os.getenv("HEVC_MIN_DIM", "128"),
            skip_audio=os.getenv("SKIP_AUDIO", "1"),
        )


@dataclass(frozen=True)
class StoredHevcArtifact:
    path: Path
    metadata_path: Path
    encode_ms: float
    paid_encode_ms: float
    cache_hit: bool
    persistent: bool
    video_path: Path
    sampled_frames: object | None
    effective_gop_size: int | None
    cache_key: str
    requested_encoder: str
    actual_encoder: str
    identity: dict


_MEMO_PATH: Path | None = None
_MEMO: dict[str, str] | None = None


def _revision(path_str, size, mtime_ns):
    return f"{path_str}|{int(size)}|{int(mtime_ns)}"


def _load_memo():
    global _MEMO
    if _MEMO is not None:
        return _MEMO
    _MEMO = {}
    if _MEMO_PATH and _MEMO_PATH.is_file():
        for line in _MEMO_PATH.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                _MEMO[record["key"]] = record["value"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return _MEMO


def _remember(key, value):
    _load_memo()[key] = value
    if not _MEMO_PATH:
        return
    try:
        _MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _MEMO_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value}) + "\n")
    except OSError:
        pass


@lru_cache(maxsize=4096)
def _source_digest_cached(path_str, size, mtime_ns):
    """SHA-256 of a source file, memoized on disk by path, size and mtime."""

    memo_key = _revision(path_str, size, mtime_ns)
    if value := _load_memo().get(memo_key):
        return value
    digest = hashlib.sha256()
    with Path(path_str).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _remember(memo_key, value)
    return value


@lru_cache(maxsize=4096)
def _video_frame_count_cached(path_str, size, mtime_ns):
    memo_key = f"frames:{_revision(path_str, size, mtime_ns)}"
    if value := _load_memo().get(memo_key):
        return int(value)
    value = probe_video_frame_count(Path(path_str))
    _remember(memo_key, str(value))
    return value


def _frame_digest(frames):
    if frames is None:
        return None
    digest = hashlib.sha256()
    digest.update(str(frames.shape).encode())
    digest.update(str(frames.dtype).encode())
    digest.update(frames.tobytes(order="C"))
    return digest.hexdigest()


def _effective_gop(path, frames, requested, explicit):
    if explicit is not None:
        return int(explicit)
    if frames is not None:
        return resolve_sampled_gop_size(int(frames.shape[0]), requested)
    stat = Path(path).stat()
    frames = _video_frame_count_cached(str(path), stat.st_size, stat.st_mtime_ns)
    return resolve_sampled_gop_size(frames, requested)


def _identity(path, frames, recipe: HevcRecipe, *, actual_encoder=None):
    effective = _effective_gop(
        path, frames, recipe.sampled_gop_size, recipe.effective_gop_size
    )
    actual = str(actual_encoder or recipe.requested_encoder)
    normalized = asdict(recipe)
    normalized.update(
        effective_gop_size=effective,
        requested_encoder=actual,
        actual_encoder=actual,
    )
    stat = Path(path).stat()
    source = (
        _frame_digest(frames)
        if frames is not None
        else _source_digest_cached(str(path), stat.st_size, stat.st_mtime_ns)
    )
    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "source_sha256": source,
        "encode_scope": recipe.encode_scope,
        "recipe": normalized,
    }


def _key(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class HevcArtifactStore:
    """Lookup or atomically create permanent HEVC artifacts."""

    def __init__(self, permanent_dir):
        self.permanent_dir = Path(permanent_dir).expanduser().resolve()
        self.artifact_dir = self.permanent_dir / "artifacts"
        self.lock_dir = self.permanent_dir / "locks"

        global _MEMO_PATH, _MEMO
        memo_path = self.permanent_dir / "source_digests.jsonl"
        if memo_path != _MEMO_PATH:
            _MEMO_PATH = memo_path
            _MEMO = None

    def _paths(self, key):
        bucket = self.artifact_dir / key[:2]
        return bucket / f"{key}.mp4", bucket / f"{key}.json"

    def _load(self, key, identity, video_path, sampled_frames):
        path, metadata_path = self._paths(key)
        if not path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        validation = metadata.get("validation", {})
        size = int(validation.get("size_bytes", path.stat().st_size))
        if (
            metadata.get("identity") != identity
            or metadata.get("complete") is not True
            or path.stat().st_size != size
        ):
            return None
        recipe = identity["recipe"]
        return StoredHevcArtifact(
            path=path,
            metadata_path=metadata_path,
            encode_ms=float(metadata["encode_ms"]),
            paid_encode_ms=0.0,
            cache_hit=True,
            persistent=True,
            video_path=video_path,
            sampled_frames=sampled_frames,
            effective_gop_size=recipe["effective_gop_size"],
            cache_key=key,
            requested_encoder=str(metadata.get("requested_encoder", "")),
            actual_encoder=str(metadata.get("actual_encoder", "")),
            identity=identity,
        )

    def _lookup(self, identities, video_path, sampled_frames):
        for identity in identities:
            artifact = self._load(_key(identity), identity, video_path, sampled_frames)
            if artifact is not None:
                return artifact
        return None

    def get_or_encode(
        self,
        video_path,
        *,
        recipe=None,
        sampled_frames=None,
        sampled_gop_size=None,
        effective_gop_size=None,
        encode_scope="sampled-clip",
        encode_fn=encode_hevc_for_inference,
        requested_encoder=None,
    ):
        video_path = Path(video_path).expanduser().resolve()
        requested = str(
            requested_encoder
            or (recipe.requested_encoder if recipe else "")
            or pick_encoder()
        )
        scope = resolve_hevc_encode_scope(
            recipe.encode_scope if recipe else encode_scope
        )

        requested_gop = (
            sampled_gop_size
            if sampled_gop_size is not None
            else recipe.sampled_gop_size if recipe else 0
        )
        effective = _effective_gop(
            video_path,
            sampled_frames,
            requested_gop,
            (
                effective_gop_size
                if effective_gop_size is not None
                else (recipe.effective_gop_size if recipe else None)
            ),
        )
        recipe = (
            HevcRecipe.from_environment(
                encode_scope=scope,
                requested_encoder=requested,
                sampled_gop_size=requested_gop,
                effective_gop_size=effective,
            )
            if recipe is None
            else replace(
                recipe,
                encode_scope=scope,
                requested_encoder=requested,
                effective_gop_size=effective,
            )
        )
        encoders = [requested] + ([] if requested == "libx265" else ["libx265"])
        identities = [
            _identity(video_path, sampled_frames, recipe, actual_encoder=encoder)
            for encoder in encoders
        ]
        if artifact := self._lookup(identities, video_path, sampled_frames):
            return artifact

        request_key = _key(identities[0])
        with _lock(self.lock_dir / f"{request_key}.lock"):
            if artifact := self._lookup(identities, video_path, sampled_frames):
                return artifact
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.artifact_dir / f".{request_key}.{uuid.uuid4().hex}.mp4"
            try:
                encoded = encode_fn(
                    video_path,
                    temporary,
                    preferred_encoder=requested,
                    sampled_frames=sampled_frames,
                    sampled_gop_size=requested_gop,
                    effective_gop_size=effective,
                )
                if encoded is None or not temporary.is_file():
                    return None

                elapsed = float(encoded.elapsed_ms)
                actual = str(encoded.actual_encoder)
                identity = _identity(
                    video_path, sampled_frames, recipe, actual_encoder=actual
                )
                key = _key(identity)
                path, metadata_path = self._paths(key)
                path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, path)
                metadata = {
                    "schema_version": STORE_SCHEMA_VERSION,
                    "identity": identity,
                    "complete": True,
                    "encode_ms": elapsed,
                    "requested_encoder": requested,
                    "actual_encoder": actual,
                    "created_at": time.time(),
                    "validation": {
                        "size_bytes": path.stat().st_size,
                        **dict(encoded.validation or {}),
                    },
                }
                atomic_write_text(
                    metadata_path,
                    json.dumps(metadata, sort_keys=True, indent=2),
                )
                return StoredHevcArtifact(
                    path=path,
                    metadata_path=metadata_path,
                    encode_ms=elapsed,
                    paid_encode_ms=elapsed,
                    cache_hit=False,
                    persistent=True,
                    video_path=video_path,
                    sampled_frames=sampled_frames,
                    effective_gop_size=effective,
                    cache_key=key,
                    requested_encoder=requested,
                    actual_encoder=actual,
                    identity=identity,
                )
            finally:
                temporary.unlink(missing_ok=True)


SCORE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HevcPatchScores:
    scores: torch.Tensor
    anchors: list[int]
    cache_hit: bool


def _score_identity(
    artifact_key,
    frame_indices,
    patch_size,
    target_height,
    target_width,
    expected_gop_size,
    anchor_policy,
):
    return {
        "schema_version": SCORE_CACHE_SCHEMA_VERSION,
        "artifact_key": str(artifact_key),
        "frame_indices": [int(index) for index in frame_indices],
        "patch_size": int(patch_size),
        "target_height": int(target_height),
        "target_width": int(target_width),
        "expected_gop_size": (
            None if expected_gop_size is None else int(expected_gop_size)
        ),
        "anchor_policy": str(anchor_policy),
    }


def _score_key(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _score_result(scores, anchors, cache_hit):
    return HevcPatchScores(
        scores=torch.as_tensor(scores, dtype=torch.float32).detach().cpu(),
        anchors=[int(index) for index in anchors],
        cache_hit=cache_hit,
    )


def _load_scores(path):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            return _score_result(payload["scores"], payload["anchors"], True)
    except (OSError, ValueError, KeyError):
        return None


class HevcScoreStore:
    def __init__(self, permanent_dir):
        self.directory = (
            Path(permanent_dir).expanduser() / "patch_scores" if permanent_dir else None
        )

    def get_or_compute(
        self,
        *,
        artifact_key,
        frame_indices,
        patch_size,
        target_height,
        target_width,
        expected_gop_size,
        anchor_policy,
        compute_fn,
    ):
        if not self.directory or not artifact_key:
            return _score_result(*compute_fn(), False)

        identity = _score_identity(
            artifact_key,
            frame_indices,
            patch_size,
            target_height,
            target_width,
            expected_gop_size,
            anchor_policy,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{_score_key(identity)}.npz"
        if cached := _load_scores(path):
            return cached

        result = _score_result(*compute_fn(), False)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as output:
                np.savez(
                    output,
                    scores=result.scores.numpy(),
                    anchors=np.asarray(result.anchors, dtype=np.int64),
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return result


@dataclass
class HevcSelectionResult:
    selector_outputs: object = None
    hevc_encode_ms: float = 0.0
    hevc_encode_paid_ms: float = 0.0
    selector_compute_ms: float = 0.0
    hevc_encode_ms_per_sample: list[float] = field(default_factory=list)
    hevc_encode_paid_ms_per_sample: list[float] = field(default_factory=list)
    hevc_cache_hits: list[bool] = field(default_factory=list)
    hevc_artifact_keys: list[str] = field(default_factory=list)
    hevc_effective_gop_sizes: list[int] = field(default_factory=list)
    kept_indices: list[int] = field(default_factory=list)
    failed_indices: list[int] = field(default_factory=list)


def prepare_hevc_batch_selection(
    video_paths,
    hevc_dir,
    selector_fn,
    device,
    *,
    sampled_frames=None,
    sampled_gop_size=None,
    encode_scope="sampled-clip",
    hevc_permanent_dir=None,
):
    """Encode a batch and invoke ``selector_fn(paths, indices, cache_keys)`` once."""

    video_paths = [Path(path).expanduser().resolve() for path in video_paths]
    if sampled_frames is None:
        sampled_frames = [None] * len(video_paths)
    else:
        sampled_frames = list(sampled_frames)
    hevc_dir = Path(hevc_dir).expanduser()
    hevc_dir.mkdir(parents=True, exist_ok=True)
    result = HevcSelectionResult()
    permanent_dir = hevc_permanent_dir or (hevc_dir / "permanent")
    store = HevcArtifactStore(permanent_dir)

    active_paths = []
    active_source_indices = []
    active_frames = []
    active_cache_keys = []
    for index, (video_path, frames) in enumerate(zip(video_paths, sampled_frames)):
        artifact = store.get_or_encode(
            video_path, sampled_frames=frames,
            sampled_gop_size=sampled_gop_size,
            encode_scope=encode_scope,
            encode_fn=encode_hevc_for_inference,
        )
        if artifact is None:
            result.failed_indices.append(index)
            continue
        result.hevc_encode_ms += artifact.encode_ms
        result.hevc_encode_paid_ms += artifact.paid_encode_ms
        result.hevc_encode_ms_per_sample.append(artifact.encode_ms)
        result.hevc_encode_paid_ms_per_sample.append(artifact.paid_encode_ms)
        result.hevc_cache_hits.append(artifact.cache_hit)
        result.hevc_artifact_keys.append(str(artifact.cache_key or ""))
        result.hevc_effective_gop_sizes.append(artifact.effective_gop_size)
        result.kept_indices.append(index)
        active_paths.append(artifact.path)
        active_source_indices.append(index)
        active_frames.append(frames)
        active_cache_keys.append(artifact.cache_key)

    if not active_paths:
        return result

    def run_selector():
        start = time.perf_counter()
        try:
            outputs = selector_fn(
                active_paths,
                active_source_indices,
                active_cache_keys,
                store.permanent_dir,
            )
            cuda_sync(device)
            return outputs
        finally:
            result.selector_compute_ms += (
                time.perf_counter() - start
            ) * 1000.0

    try:
        result.selector_outputs = run_selector()
    except RuntimeError as exc:
        encoder = pick_encoder()
        tqdm.write(
            "HEVC preparation rejected the encoded batch produced with "
            f"{encoder}; retrying with libx265."
        )
        for active_index, (source_index, frames) in enumerate(zip(
            active_source_indices, active_frames
        )):
            fallback = store.get_or_encode(
                video_paths[source_index], sampled_frames=frames,
                sampled_gop_size=sampled_gop_size,
                effective_gop_size=result.hevc_effective_gop_sizes[active_index],
                encode_scope=encode_scope,
                encode_fn=encode_hevc_for_inference,
                requested_encoder="libx265",
            )
            if fallback is None:
                raise RuntimeError(
                    f"libx265 fallback encoding failed: {video_paths[source_index]}"
                ) from exc
            active_paths[active_index] = fallback.path
            result.hevc_encode_ms += float(fallback.encode_ms)
            result.hevc_encode_paid_ms += float(fallback.paid_encode_ms)
            result.hevc_encode_ms_per_sample[active_index] += float(fallback.encode_ms)
            result.hevc_encode_paid_ms_per_sample[active_index] += float(fallback.paid_encode_ms)
            result.hevc_cache_hits[active_index] = bool(fallback.cache_hit)
            result.hevc_artifact_keys[active_index] = str(fallback.cache_key or "")
            result.hevc_effective_gop_sizes[active_index] = fallback.effective_gop_size
            active_cache_keys[active_index] = fallback.cache_key
        result.selector_outputs = run_selector()
    return result
