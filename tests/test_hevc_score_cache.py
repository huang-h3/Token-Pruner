from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from token_pruner import selectors
from token_pruner.hevc_cache import prepare_hevc_batch_selection
from token_pruner.hevc_cache import HevcScoreStore


def test_patch_score_cache_persists_and_invalidates_frame_schedule(tmp_path):
    calls = []

    def compute():
        calls.append(True)
        return torch.arange(12, dtype=torch.float32).reshape(3, 4), [0, 2]

    common = {
        "cache_dir": tmp_path,
        "artifact_key": "encoded-artifact",
        "patch_size": 2,
        "target_height": 4,
        "target_width": 4,
        "expected_gop_size": 8,
        "anchor_policy": "first",
        "compute_fn": compute,
    }
    store = HevcScoreStore(tmp_path)
    common.pop("cache_dir")
    first = store.get_or_compute(frame_indices=[0, 4, 8], **common)
    second = store.get_or_compute(frame_indices=[0, 4, 8], **common)
    changed = store.get_or_compute(frame_indices=[0, 5, 8], **common)

    assert len(calls) == 2
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert changed.cache_hit is False
    assert torch.equal(first.scores, second.scores)
    assert first.anchors == second.anchors == [0, 2]
    assert len(list((tmp_path / "patch_scores").glob("*.npz"))) == 2


def test_qwen_tubelet_keep_rates_share_cached_patch_scores(tmp_path, monkeypatch):
    saliency = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2).numpy()

    def build_selector(total_keep):
        return selectors.TubeletSelector(
            patch_size=1,
            tubelet_size=2,
            tubelet_budget=1,
            image_shape=(1, 3, 4, 2, 2),
            prune_mode="global",
            preserve_anchor_tubelets=True,
            total_tubelet_budget=total_keep,
            min_per_group=1,
            hevc_gop_size=2,
            hevc_anchor_policy="first",
        )

    with mock.patch(
            "token_pruner.selectors._extract_hevc_saliency",
            return_value=(saliency, [0]),
        ) as decode:
        first = build_selector(5).select(
            "encoded.mp4",
            [0, 1, 2, 3],
            cache_keys="encoded-artifact", score_cache_dir=tmp_path,
        )
        second = build_selector(6).select(
            "encoded.mp4",
            [0, 1, 2, 3],
            cache_keys="encoded-artifact", score_cache_dir=tmp_path,
        )

    decode.assert_called_once()
    assert first.visible_indices.shape == (1, 1)
    assert second.visible_indices.shape == (1, 2)
    assert first.anchor_partitions.tolist() == second.anchor_partitions.tolist()


def test_hevc_preparation_passes_artifact_key_to_selector(tmp_path, monkeypatch):
    selector = mock.Mock(return_value="signal")
    artifact_path = tmp_path / "encoded.mp4"
    artifact_path.write_bytes(b"encoded")
    artifact = SimpleNamespace(
        path=artifact_path,
        encode_ms=1.0,
        paid_encode_ms=0.0,
        cache_hit=True,
        persistent=False,
        effective_gop_size=8,
        cache_key="stable-artifact-key",
    )
    class FakeStore:
        def __init__(self, _permanent):
            self.permanent_dir = tmp_path

        def get_or_encode(self, *_args, **_kwargs):
            return artifact

    monkeypatch.setattr(
        "token_pruner.hevc_cache.HevcArtifactStore",
        FakeStore,
    )
    result = prepare_hevc_batch_selection(
        [Path("source.mp4")],
        tmp_path,
        selector,
        torch.device("cpu"),
    )

    assert result.selector_outputs == "signal"
    assert selector.call_args.args[2] == ["stable-artifact-key"]
