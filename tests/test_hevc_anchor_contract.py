import numpy as np
import pytest

from token_pruner import selectors
from token_pruner.hevc_cache import HevcRecipe, _identity
from token_pruner.hevc import (
    build_ffmpeg_cmd,
    build_inference_ffmpeg_cmd,
    build_sampled_frames_ffmpeg_cmd,
    resolve_hevc_encode_scope,
)


class _FakeHevcReader:
    height = 2
    width = 2

    def __init__(self, frame_types, residual_value=128):
        self.frame_types = frame_types
        self.residual_value = residual_value
        self.nb_frames = len(frame_types)

    @staticmethod
    def _upsample_mv_to_hw(values):
        return values

    def nextFrame(self):
        zeros = np.zeros((2, 2), dtype=np.int16)
        residual = np.full((2, 2), self.residual_value, dtype=np.uint8)
        for frame_type in self.frame_types:
            yield (
                frame_type,
                None,
                None,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                residual,
            )

    def close(self):
        return None


def _install_reader(monkeypatch, frame_types, residual_value=128):
    monkeypatch.setattr(
        selectors,
        "open_hevc_feature_reader",
        lambda *_args, **_kwargs: _FakeHevcReader(
            frame_types,
            residual_value=residual_value,
        ),
    )


def test_static_p_frame_is_not_misclassified_as_an_anchor(monkeypatch):
    _install_reader(monkeypatch, [0, 2])

    saliency, anchors = selectors._extract_hevc_saliency(
        "unused.mp4", [0, 1], 2, 2
    )

    assert saliency.shape == (2, 2, 2)
    assert anchors == [0]


def test_single_gop_contract_rejects_a_second_codec_idr(monkeypatch):
    _install_reader(monkeypatch, [0, 0])

    with pytest.raises(RuntimeError, match="exactly one IDR"):
        selectors._extract_hevc_saliency("unused.mp4", [0, 1], 2, 2)


def test_multi_gop_idrs_are_valid_but_only_frame_zero_is_protected(monkeypatch):
    frame_types = [0 if index % 16 == 0 else 2 for index in range(64)]
    _install_reader(monkeypatch, frame_types)

    _, anchors = selectors._extract_hevc_saliency(
        "unused.mp4",
        list(range(64)),
        2,
        2,
        expected_gop_size=16,
    )
    assert anchors == [0]

    _, anchors = selectors._extract_hevc_saliency(
        "unused.mp4",
        list(range(64)),
        2,
        2,
        expected_gop_size=16,
        hevc_anchor_policy="all",
    )
    assert anchors == [0, 16, 32, 48]


def test_selector_reads_absolute_source_positions(monkeypatch):
    _install_reader(
        monkeypatch,
        [0, 2, 0, 2, 0],
        residual_value=140,
    )

    scores, anchors = selectors._extract_hevc_saliency(
        "source.mp4", [0, 2, 4], 2, 2, expected_gop_size=2
    )

    assert scores.shape == (3, 2, 2)
    assert anchors == [0]
    assert np.max(scores[0]) == 0
    assert np.max(scores[1]) > 0
    assert np.max(scores[2]) > 0


def test_first_policy_anchors_first_sampled_position(monkeypatch):
    _install_reader(
        monkeypatch,
        [2 if index not in {0, 16, 32} else 0 for index in range(40)],
        residual_value=140,
    )

    scores, anchors = selectors._extract_hevc_saliency(
        "source.mp4", [2, 7, 12, 17, 21, 26, 31, 36], 2, 2,
        expected_gop_size=16,
        hevc_anchor_policy="first",
    )

    assert anchors == [0]
    assert np.max(scores[0]) == 0
    assert np.max(scores[1]) > 0


def test_first_policy_handles_repeated_sampled_source_frames(monkeypatch):
    _install_reader(monkeypatch, [0, 2, 2], residual_value=140)

    scores, anchors = selectors._extract_hevc_saliency(
        "source.mp4", [0, 0, 2], 2, 2, expected_gop_size=0,
        hevc_anchor_policy="first",
    )

    assert anchors == [0]
    assert np.max(scores[0]) == 0
    assert np.max(scores[1]) > 0
    assert np.max(scores[2]) > 0


def test_first_policy_scores_later_idr_but_all_policy_zeros_it(monkeypatch):
    frame_types = [0, 2, 0, 2]
    _install_reader(monkeypatch, frame_types, residual_value=140)

    first_scores, first_anchors = selectors._extract_hevc_saliency(
        "unused.mp4", list(range(4)), 2, 2, expected_gop_size=2,
        hevc_anchor_policy="first",
    )
    all_scores, all_anchors = selectors._extract_hevc_saliency(
        "unused.mp4", list(range(4)), 2, 2, expected_gop_size=2,
        hevc_anchor_policy="all",
    )

    assert first_anchors == [0]
    assert all_anchors == [0, 2]
    assert np.max(first_scores[2]) > 0
    assert np.max(all_scores[2]) == 0


def test_multi_gop_anchor_contract_rejects_missing_or_extra_idrs(monkeypatch):
    _install_reader(
        monkeypatch,
        [0 if index in {0, 16, 48} else 2 for index in range(64)],
    )
    with pytest.raises(RuntimeError, match="codec IDR"):
        selectors._extract_hevc_saliency(
            "unused.mp4", list(range(64)), 2, 2, expected_gop_size=16
        )

    _install_reader(
        monkeypatch,
        [0 if index in {0, 16, 24, 32, 48} else 2 for index in range(64)],
    )
    with pytest.raises(RuntimeError, match="codec IDR"):
        selectors._extract_hevc_saliency(
            "unused.mp4", list(range(64)), 2, 2, expected_gop_size=16
        )


def test_terminal_sampler_index_is_clamped_to_encoded_video(monkeypatch):
    frame_types = [0 if index % 32 == 0 else 2 for index in range(96)]
    _install_reader(monkeypatch, frame_types, residual_value=140)

    scores, anchors = selectors._extract_hevc_saliency(
        "source.mp4", [0, 94, 96], 2, 2,
        expected_gop_size=32,
        hevc_anchor_policy="first",
    )

    assert scores.shape == (3, 2, 2)
    assert anchors == [0]
    assert np.max(scores[-1]) > 0


def test_full_video_scope_falls_back_when_video_ablation_is_active(monkeypatch):
    monkeypatch.setenv("VIDEO_ABLATION", "static")

    assert resolve_hevc_encode_scope("full-video") == "sampled-clip"


def test_nvenc_command_disables_adaptive_i_frames():
    command = build_ffmpeg_cmd(
        "input.mp4",
        "output.mp4",
        "hevc_nvenc",
        gop_size=8,
    )

    assert command[command.index("-g") + 1] == "8"
    assert command[command.index("-force_key_frames") + 1] == (
        "expr:gte(n,n_forced*8)"
    )
    assert command[command.index("-no-scenecut") + 1] == "1"
    assert command[command.index("-strict_gop") + 1] == "1"
    assert command[command.index("-forced-idr") + 1] == "1"


def test_sampled_gop_size_is_encoded_and_defaults_to_one_gop():
    frames = np.zeros((64, 224, 224, 3), dtype=np.uint8)
    command = build_sampled_frames_ffmpeg_cmd(
        frames, "output.mp4", "libx265", gop_size=16
    )
    assert command[command.index("-g") + 1] == "16"
    assert "keyint=16:min-keyint=16:scenecut=0" in command[
        command.index("-x265-params") + 1
    ]

    nvenc = build_sampled_frames_ffmpeg_cmd(
        frames, "output.mp4", "hevc_nvenc", gop_size=16
    )
    assert nvenc[nvenc.index("-g") + 1] == "16"
    assert nvenc[nvenc.index("-no-scenecut") + 1] == "1"
    assert nvenc[nvenc.index("-strict_gop") + 1] == "1"

    command = build_sampled_frames_ffmpeg_cmd(
        frames, "output.mp4", "libx265", gop_size=0
    )
    assert command[command.index("-g") + 1] == "64"


def test_source_video_command_uses_the_full_video_gop():
    command = build_inference_ffmpeg_cmd(
        "source.mp4", "output.mp4", "libx265", gop_size=16
    )
    assert command[command.index("-g") + 1] == "16"
    assert command[command.index("-force_key_frames") + 1] == (
        "expr:gte(n,n_forced*16)"
    )
    assert "keyint=16:min-keyint=16:scenecut=0" in command[
        command.index("-x265-params") + 1
    ]
    assert "open-gop=0" in command[command.index("-x265-params") + 1]
    assert "out_range=tv" in command[command.index("-vf") + 1]


def test_source_identity_uses_effective_gop(monkeypatch, tmp_path):
    from token_pruner import hevc_cache as hevc_store

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(hevc_store, "probe_video_frame_count", lambda _path: 64)
    hevc_store._video_frame_count_cached.cache_clear()
    hevc_store.HevcArtifactStore(tmp_path / "store")

    automatic = HevcRecipe(
        encode_scope="full-video", requested_encoder="libx265"
    )
    explicit = HevcRecipe(
        encode_scope="full-video", requested_encoder="libx265",
        sampled_gop_size=16,
    )
    assert _identity(video, None, automatic)["recipe"]["effective_gop_size"] == 64
    assert _identity(video, None, explicit)["recipe"]["effective_gop_size"] == 16


def test_sampled_frame_count_defines_gop_identity(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    frames = np.zeros((6, 2, 2, 3), dtype=np.uint8)
    automatic = HevcRecipe(requested_encoder="hevc_nvenc")
    explicit = HevcRecipe(requested_encoder="hevc_nvenc", sampled_gop_size=3)

    first = _identity(video, frames, automatic)
    second = _identity(video, frames, explicit)
    assert first["recipe"]["effective_gop_size"] == 6
    assert second["recipe"]["effective_gop_size"] == 3
    assert first != second
