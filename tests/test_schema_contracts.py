"""Contract tests for cache identities and measurement schemas."""

import numpy as np

from token_pruner.measurements import (
    GENERATION_MEMORY_KEYS,
    GENERATION_TIMING_KEYS,
)
from token_pruner.records import RESULT_SCHEMA_VERSION
from token_pruner.hevc_cache import (
    SCORE_CACHE_SCHEMA_VERSION,
    _score_identity,
)
from token_pruner.hevc_cache import STORE_SCHEMA_VERSION, _identity, _key
from token_pruner.hevc_cache import HevcRecipe


def test_schema_versions_are_pinned():
    """Every version a cached artifact or a stored result is keyed by."""

    assert SCORE_CACHE_SCHEMA_VERSION == 1
    assert STORE_SCHEMA_VERSION == 1
    assert RESULT_SCHEMA_VERSION == 1


def test_measurement_schema_is_pinned():
    """The keys every collected CSV is written against."""

    assert GENERATION_MEMORY_KEYS == ("peak_mem_mb", "delta_peak_mem_mb")
    assert len(GENERATION_TIMING_KEYS) == 26
    assert GENERATION_TIMING_KEYS[:5] == (
        "preprocessing_ms", "signal_ms", "model_ms", "inference_ms", "e2e_ms",
    )
    # Spelt out so a rename or a reorder cannot pass as an addition.
    assert set(GENERATION_TIMING_KEYS) == {
        "preprocessing_ms", "signal_ms", "model_ms", "inference_ms", "e2e_ms",
        "hevc_encode_ms", "signal_to_mask_ms", "signal_overhead_ms",
        "vlm_input_adapt_ms", "vlm_generate_ms", "token_reduce_ms",
        "vision_encoder_ms", "projector_ms", "prefill_ms", "decode_ms",
        "decode_steps", "model_ttft_ms", "ttft_ms", "tpot_ms", "output_tokens",
        "hit_token_cap", "hevc_encode_paid_ms", "signal_paid_ms",
        "inference_paid_ms", "e2e_paid_ms", "e2e_wall_ms",
    }


def test_score_cache_identity_fields_are_pinned():
    """Patch-score cache keys chain off the artifact key; both must hold."""

    identity = _score_identity(
        artifact_key="abc", frame_indices=[0, 1, 2], patch_size=14,
        target_height=224, target_width=224, expected_gop_size=8,
        anchor_policy="first",
    )
    assert sorted(identity) == [
        "anchor_policy", "artifact_key", "expected_gop_size", "frame_indices",
        "patch_size", "schema_version", "target_height", "target_width",
    ]
    assert identity["schema_version"] == SCORE_CACHE_SCHEMA_VERSION


def test_store_identity_and_key_are_pinned(tmp_path):
    """The permanent store key: field layout and the digest it produces."""

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"pinned-content")
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    recipe = HevcRecipe(requested_encoder="libx265", encode_scope="sampled-clip")
    identity = _identity(video, frames, recipe, actual_encoder="libx265")

    assert sorted(identity) == [
        "encode_scope", "recipe", "schema_version", "source_sha256",
    ]
    assert identity["schema_version"] == STORE_SCHEMA_VERSION
    # Content-addressed: the key must not depend on where the file lives.
    moved = tmp_path / "nested"
    moved.mkdir()
    other = moved / "different-name.mp4"
    other.write_bytes(b"pinned-content")
    assert _key(_identity(other, frames, recipe, actual_encoder="libx265")) == _key(identity)
    # ...and the requested encoder must not split the key from the actual one.
    fallback = HevcRecipe(requested_encoder="hevc_nvenc", encode_scope="sampled-clip")
    assert _key(_identity(video, frames, fallback, actual_encoder="libx265")) == _key(identity)


def test_store_rejects_a_truncated_sidecar(tmp_path):
    """A half-written sidecar must read as a miss, never as a hit."""

    from token_pruner.hevc_cache import HevcArtifactStore

    store = HevcArtifactStore(tmp_path)
    store.permanent_dir.mkdir(parents=True, exist_ok=True)
    (store.permanent_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    broken = store.permanent_dir / "artifacts" / "ab"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "abcd.mp4").write_bytes(b"x")
    (broken / "abcd.json").write_text("{not json")

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"content")
    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    recipe = HevcRecipe(requested_encoder="libx265", encode_scope="sampled-clip")
    assert store._load(_key(_identity(video, frames, recipe)),
                       _identity(video, frames, recipe), video, frames) is None


def test_store_key_separates_same_basename_different_content(tmp_path):
    """Two clips named alike but holding different bytes are different keys."""

    from token_pruner.hevc_cache import HevcRecipe as Recipe

    recipe = Recipe(requested_encoder="libx265", encode_scope="sampled-clip")
    keys = []
    for index, payload in enumerate((0, 7)):
        directory = tmp_path / f"run{index}"
        directory.mkdir()
        video = directory / "clip.mp4"
        video.write_bytes(b"same-bytes-on-disk")
        frames = np.full((2, 2, 2, 3), payload, dtype=np.uint8)
        keys.append(_key(_identity(video, frames, recipe, actual_encoder="libx265")))
    assert keys[0] != keys[1], "same basename, different content collided"


def test_source_digest_memo_survives_a_new_process_and_invalidates(tmp_path):
    """The digest memo is keyed on path, size and mtime, and survives a process."""

    from token_pruner import hevc_cache as hevc_store

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"original-content")
    store_dir = tmp_path / "store"
    hevc_store.HevcArtifactStore(store_dir)

    stat = video.stat()
    first = hevc_store._source_digest_cached(
        str(video), stat.st_size, stat.st_mtime_ns)
    memo = store_dir / "source_digests.jsonl"
    assert memo.is_file(), "the memo was not written"

    # Stand in for the next arm's process: keep only what is on disk.
    hevc_store._source_digest_cached.cache_clear()
    hevc_store._MEMO = None
    hevc_store.HevcArtifactStore(store_dir)
    stat = video.stat()
    assert hevc_store._source_digest_cached(
        str(video), stat.st_size, stat.st_mtime_ns) == first

    # Rewriting the file must produce a different key, hence a real rehash.
    video.write_bytes(b"different-content-entirely")
    stat = video.stat()
    changed = hevc_store._source_digest_cached(
        str(video), stat.st_size, stat.st_mtime_ns)
    assert changed != first


def test_digest_memo_tolerates_a_torn_final_line(tmp_path):
    """A killed job leaves a half-written line; the rest must stay usable."""

    from token_pruner import hevc_cache as hevc_store

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    (store_dir / "source_digests.jsonl").write_text(
        '{"key": "a|1|2", "value": "deadbeef"}\n{"key": "b|1|2", "val')

    hevc_store._MEMO = None
    hevc_store.HevcArtifactStore(store_dir)
    memo = hevc_store._load_memo()
    assert memo == {"a|1|2": "deadbeef"}


def test_frame_count_memo_is_value_identical_to_probing(tmp_path, monkeypatch):
    """Memoising the frame count must not move a single artifact key."""

    from token_pruner import hevc as hevc_encoding, hevc_cache as hevc_store

    calls = []

    def counting_probe(path):
        calls.append(str(path))
        return 300

    monkeypatch.setattr(hevc_store, "probe_video_frame_count", counting_probe)
    monkeypatch.setattr(hevc_encoding, "probe_video_frame_count", counting_probe)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"content")
    hevc_store._video_frame_count_cached.cache_clear()
    hevc_store._MEMO = None
    hevc_store.HevcArtifactStore(tmp_path / "store")

    first = hevc_store._effective_gop(video, None, 32, None)
    assert len(calls) == 1, "the probe should run once"
    assert first == hevc_encoding.resolve_sampled_gop_size(300, 32), (
        "the memo changed the resolved GOP, which would move every key")

    # A new process keeps the on-disk memo but loses the in-memory one.
    hevc_store._video_frame_count_cached.cache_clear()
    hevc_store._MEMO = None
    hevc_store.HevcArtifactStore(tmp_path / "store")
    assert hevc_store._effective_gop(video, None, 32, None) == first
    assert len(calls) == 1, "the probe ran again despite the memo"


def test_explicit_gop_skips_the_probe(tmp_path, monkeypatch):
    """An explicit GOP is used as given, without probing the source video."""

    from token_pruner import hevc_cache as hevc_store

    def explode(path):
        raise AssertionError("probed despite an explicit GOP")

    monkeypatch.setattr(hevc_store, "probe_video_frame_count", explode)
    assert hevc_store._effective_gop(tmp_path / "missing.mp4", None, 0, 32) == 32
