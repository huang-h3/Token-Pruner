"""Permanent HEVC artifact store: lookup, reuse, and encoder identity."""

from pathlib import Path

import numpy as np
import torch

from token_pruner.hevc import EncodeResult
from token_pruner.hevc_cache import HevcArtifactStore, HevcRecipe, prepare_hevc_batch_selection


def _encoder(calls, *, actual=None, elapsed_ms=1.0):
    def encode(src, dst, **kwargs):
        calls.append(kwargs.get("preferred_encoder"))
        Path(dst).write_bytes(b"encoded-hevc")
        requested = kwargs.get("preferred_encoder")
        return EncodeResult(elapsed_ms, requested, actual or requested)

    return encode


def test_hevc_store_lookup_protocol(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    calls = []
    encode = _encoder(calls, elapsed_ms=7.0)

    store = HevcArtifactStore(tmp_path / "permanent")
    first = store.get_or_encode(source, sampled_frames=frames, sampled_gop_size=2, encode_fn=encode)
    second = store.get_or_encode(source, sampled_frames=frames, sampled_gop_size=2, encode_fn=encode)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1
    assert first.path == second.path
    assert first.metadata_path.is_file()
    assert first.encode_ms == second.encode_ms == 7.0


def test_selection_uses_explicit_permanent_store_before_encoding(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    work = tmp_path / "work"
    permanent = tmp_path / "permanent"
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    calls = []

    monkeypatch.setattr(
        "token_pruner.hevc_cache.encode_hevc_for_inference", _encoder(calls)
    )
    first = prepare_hevc_batch_selection(
        [source], work, lambda paths, indices, keys, _cache_dir: (paths, keys),
        torch.device("cpu"), sampled_frames=[frames],
        sampled_gop_size=2, hevc_permanent_dir=permanent,
    )
    second = prepare_hevc_batch_selection(
        [source], work, lambda paths, indices, keys, _cache_dir: (paths, keys),
        torch.device("cpu"), sampled_frames=[frames],
        sampled_gop_size=2, hevc_permanent_dir=permanent,
    )
    assert len(calls) == 1
    assert first.hevc_cache_hits == [False]
    assert second.hevc_cache_hits == [True]
    assert first.selector_outputs[0][0] == second.selector_outputs[0][0]


def test_fallback_actual_encoder_identity_is_reusable(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    calls = []
    encode = _encoder(calls, actual="libx265", elapsed_ms=2.0)
    recipe = HevcRecipe(requested_encoder="hevc_nvenc", sampled_gop_size=2)
    store = HevcArtifactStore(tmp_path / "permanent")
    first = store.get_or_encode(source, recipe=recipe, sampled_frames=frames, encode_fn=encode)
    second = store.get_or_encode(source, recipe=recipe, sampled_frames=frames, encode_fn=encode)
    assert first.actual_encoder == second.actual_encoder == "libx265"
    assert first.requested_encoder == second.requested_encoder == "hevc_nvenc"
    assert second.cache_hit is True
    assert len(calls) == 1


def test_store_metadata_does_not_record_the_source_location(tmp_path):
    source = tmp_path / "private" / "source.mp4"
    source.parent.mkdir()
    source.write_bytes(b"source")
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    store = HevcArtifactStore(tmp_path / "permanent")
    artifact = store.get_or_encode(
        source, sampled_frames=frames, sampled_gop_size=2, encode_fn=_encoder([]),
    )
    assert "private" not in artifact.metadata_path.read_text()
