"""HEVC encoding and motion-vector/residual feature reading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import time

from tqdm import tqdm

#: Keyframes are forced every GOP; the codec's IDR frames are the anchors.
HEVC_CODEC_LAYOUT = "fixed-periodic-idr-v1"
#: Anchor frames get zero saliency; every other frame is scored from MV and residual.
HEVC_SIGNAL_POLICY = "protected-anchor-zero-only-v1"


def resolve_hevc_encode_scope(requested=None):
    scope = str(
        os.getenv("HEVC_ENCODE_SCOPE", "sampled-clip")
        if requested is None
        else requested
    ).strip().lower()
    if (
        scope == "full-video"
        and os.getenv("VIDEO_ABLATION", "none").strip().lower() not in {"", "none"}
    ):
        return "sampled-clip"
    return scope


def resolve_sampled_gop_size(num_frames, requested=None):
    """Resolve a requested GOP length against a known frame count."""
    num_frames = int(num_frames)
    if requested is None:
        requested = int(os.getenv("HEVC_SAMPLED_GOP_SIZE", "0"))
    requested = int(requested)
    return num_frames if requested == 0 else min(requested, num_frames)


@lru_cache(maxsize=1)
def require_ffmpeg():
    """Fail once, before a batch starts, when HEVC tooling is unavailable."""

    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"HEVC pruning requires {', '.join(missing)} on PATH"
        )


def _hevc_encoders(listing):
    """HEVC encoder names present in ``ffmpeg -encoders`` output."""

    names = [line.split()[1] for line in listing.splitlines() if len(line.split()) > 1]
    return sorted({name for name in names if "hevc" in name or "265" in name})


@lru_cache(maxsize=None)
def _pick_encoder(preferred):
    """Resolve an encoder once for each explicit preference."""

    candidates = [item for item in (preferred, "libx265", "hevc_nvenc") if item]
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        available = (proc.stdout or "").lower()
    except OSError:
        available = ""
    for encoder in candidates:
        if encoder.lower() in available:
            if preferred and encoder != preferred:
                raise RuntimeError(
                    f"requested HEVC encoder {preferred} is unavailable; "
                    f"ffmpeg provides {_hevc_encoders(available)}"
                )
            return encoder
    return preferred or "libx265"


def pick_encoder():
    """Return the available encoder for the current environment preference."""

    return _pick_encoder(os.getenv("HEVC_ENCODER", "").strip())


def scale_filter():
    """Keep dimensions even and avoid hardware encoder minimum-size failures."""
    minimum = max(2, int(os.getenv("HEVC_MIN_DIM", "128")))
    if minimum % 2:
        minimum += 1
    return (
        f"scale=max({minimum}\\,trunc(iw/2)*2):"
        f"max({minimum}\\,trunc(ih/2)*2)"
    )


def build_ffmpeg_cmd(
    src_path,
    dst_path,
    encoder,
    ff_log="error",
    x265_log="error",
    gop_size=None,
):
    """Build the raw-video to HEVC encode command."""
    gop_size = (
        int(os.getenv("GOP_SIZE", "8"))
        if gop_size is None
        else int(gop_size)
    )
    crf = int(os.getenv("CRF", "23"))
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        str(ff_log),
        "-i",
        str(src_path),
        "-vf",
        scale_filter(),
        "-c:v",
        str(encoder),
        "-pix_fmt",
        "yuv420p",
        "-force_key_frames",
        f"expr:gte(n,n_forced*{gop_size})",
    ]

    if encoder == "libx265":
        command.extend(
            [
                "-preset",
                "fast",
                "-crf",
                str(crf),
                "-g",
                str(gop_size),
                "-x265-params",
                (
                    f"keyint={gop_size}:min-keyint={gop_size}:scenecut=0:"
                    "bframes=0:ref=1:repeat-headers=1:open-gop=0:"
                    f"log-level={x265_log}"
                ),
            ]
        )
    else:
        command.extend(
            [
                "-preset",
                "p5",
                "-cq",
                str(crf),
                "-g",
                str(gop_size),
                "-bf",
                "0",
                "-sc_threshold",
                "0",
                "-rc-lookahead",
                "0",
                "-no-scenecut",
                "1",
                "-strict_gop",
                "1",
                "-forced-idr",
                "1",
            ]
        )

    command.extend(["-tag:v", "hvc1", "-movflags", "+faststart"])
    if os.getenv("SKIP_AUDIO", "1") == "1":
        command.append("-an")
    else:
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    command.append(str(dst_path))
    return command


def build_inference_ffmpeg_cmd(
    video_path,
    hevc_path,
    encoder,
    *,
    gop_size=None,
):
    """Build the full-source online selector command."""
    command = build_ffmpeg_cmd(
        video_path,
        hevc_path,
        encoder,
        gop_size=gop_size,
    )
    filter_index = command.index("-vf") + 1
    command[filter_index] = f"{scale_filter()}:out_range=tv,format=yuv420p"
    return command


def build_sampled_frames_ffmpeg_cmd(
    sampled_frames,
    hevc_path,
    encoder,
    *,
    gop_size=None,
):
    """Build an HEVC command for the exact RGB frames sent to the model."""
    shape = sampled_frames.shape
    num_frames, height, width, _ = (int(value) for value in shape)
    # ffmpeg reads the raw buffer as rgb24; another dtype would be reinterpreted
    effective_gop_size = resolve_sampled_gop_size(num_frames, gop_size)

    command = build_ffmpeg_cmd(
        "pipe:0",
        hevc_path,
        encoder,
        gop_size=effective_gop_size,
    )
    input_index = command.index("-i")
    command[input_index:input_index] = [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        "1",
    ]
    filter_index = command.index("-vf") + 1
    if "format=yuv420p" not in command[filter_index]:
        command[filter_index] += ",format=yuv420p"
    return command


def _probe_stream(video_path, entries, *, extra=()):
    """Read ffprobe stream entries for the first video stream."""

    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", *extra,
            "-show_entries", f"stream={entries}",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip() and line.strip() != "N/A"
    ]


def probe_video_frame_count(video_path):
    # ``nb_read_frames`` forces a decode; ``nb_frames`` is the fallback.
    return int(_probe_stream(
        video_path, "nb_read_frames,nb_frames", extra=("-count_frames",))[0])


def resolve_video_gop_size(video_path, requested=None):
    """Resolve GOP length against the complete source video's frame count."""
    if requested is None:
        requested = os.getenv("HEVC_SAMPLED_GOP_SIZE", "0")
    return resolve_sampled_gop_size(
        probe_video_frame_count(video_path),
        requested,
    )


def probe_video_pix_fmt(video_path):
    values = _probe_stream(video_path, "pix_fmt")
    return values[0] if values else ""


def probe_video_codec(video_path):
    values = _probe_stream(video_path, "codec_name")
    return values[0] if values else ""


def probe_video_keyframes(video_path):
    """Return ``(frame_count, keyframe_indices)`` from per-frame flags."""

    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=key_frame,pict_type",
            "-of", "csv=p=0", str(video_path),
        ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    keyframes = []
    frame_count = 0
    for line in proc.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if not fields or not fields[0].isdigit():
            continue
        if fields[0] == "1":
            keyframes.append(frame_count)
        frame_count += 1
    return frame_count, keyframes


def validate_hevc_artifact(video_path, *, expected_gop_size=None):
    """Check the IDR layout and return the codec facts the selector needs."""
    codec = probe_video_codec(video_path)
    frame_count, keyframes = probe_video_keyframes(video_path)
    pix_fmt = probe_video_pix_fmt(video_path)
    if expected_gop_size is not None:
        gop = int(expected_gop_size)
        # Every expected IDR position must be present, and no others.
        expected_keyframes = list(range(0, frame_count, gop))
        if keyframes != expected_keyframes:
            missing = sorted(set(expected_keyframes) - set(keyframes))
            unexpected = sorted(set(keyframes) - set(expected_keyframes))
            raise RuntimeError(
                f"HEVC keyframes {keyframes!r} do not match GOP={gop} over "
                f"{frame_count} frame(s); expected {expected_keyframes!r}"
                + (f", missing={missing}" if missing else "")
                + (f", unexpected={unexpected}" if unexpected else "")
                + "."
            )
    return {
        "codec": codec, "pix_fmt": pix_fmt, "frame_count": frame_count,
        "keyframe_indices": keyframes,
        "gop_size": expected_gop_size,
    }


@dataclass(frozen=True)
class EncodeResult:
    """Encoding timing plus the encoder that actually produced the file."""

    elapsed_ms: float
    requested_encoder: str
    actual_encoder: str
    pixel_format: str = "yuv420p"
    path: Path | None = None
    validation: object | None = None


def encode_hevc_for_inference(
    video_path,
    hevc_path,
    preferred_encoder=None,
    sampled_frames=None,
    sampled_gop_size=None,
    effective_gop_size=None,
):
    """Encode the sampled clip, or the full source when ``sampled_frames`` is None."""
    require_ffmpeg()
    encoder = preferred_encoder or pick_encoder()
    requested_encoder = str(encoder)
    encoders = [encoder]
    if encoder != "libx265":
        encoders.append("libx265")

    hevc_path = Path(hevc_path)
    frame_bytes = (
        None
        if sampled_frames is None
        else sampled_frames.tobytes(order="C")
    )
    source_gop_size = (
        effective_gop_size
        if sampled_frames is None and effective_gop_size is not None
        else (
            resolve_video_gop_size(video_path, sampled_gop_size)
            if sampled_frames is None
            else None
        )
    )
    total_ms = 0.0
    for encoder in encoders:
        start = time.perf_counter()
        try:
            command = (
                build_inference_ffmpeg_cmd(
                    video_path,
                    hevc_path,
                    encoder,
                    gop_size=source_gop_size,
                )
                if sampled_frames is None
                else build_sampled_frames_ffmpeg_cmd(
                    sampled_frames,
                    hevc_path,
                    encoder,
                    gop_size=sampled_gop_size,
                )
            )
            subprocess.run(
                command,
                check=True,
                input=frame_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            pixel_format = probe_video_pix_fmt(hevc_path)
            encode_ms = (time.perf_counter() - start) * 1000.0
            if pixel_format == "yuv420p":
                expected_gop_size = (
                    source_gop_size
                    if sampled_frames is None
                    else resolve_sampled_gop_size(
                        int(sampled_frames.shape[0]), sampled_gop_size
                    )
                )
                validation = validate_hevc_artifact(
                    hevc_path, expected_gop_size=expected_gop_size,
                )
                return EncodeResult(
                    total_ms + encode_ms, requested_encoder, str(encoder),
                    pixel_format, hevc_path, validation,
                )
            total_ms += encode_ms
            hevc_path.unlink(missing_ok=True)
            tqdm.write(
                f"HEVC encode produced {pixel_format or 'unknown'} with "
                f"{encoder}; retrying with fallback."
            )
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            total_ms += (time.perf_counter() - start) * 1000.0
            try:
                hevc_path.unlink(missing_ok=True)
            except OSError:
                pass
            output = (
                getattr(exc, "stderr", "")
                or getattr(exc, "stdout", "")
                or str(exc)
            )
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            detail_lines = lines[-5:]
            detail = (
                " detail=" + " | ".join(detail_lines)[:800]
                if detail_lines
                else ""
            )
            return_code = getattr(exc, "returncode", "n/a")
            tqdm.write(
                f"HEVC encode failed with {encoder}: rc={return_code}{detail}"
            )
            continue
    return None


def open_hevc_feature_reader(*args, **kwargs):
    from .dataloading.hevc_feature_decoder_mv import HevcFeatureReader

    return HevcFeatureReader(*args, **kwargs)
