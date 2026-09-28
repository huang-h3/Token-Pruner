# Adapted from OneVision-Encoder; trimmed to the feature reader used here and modified for this repository's decoder path.
import json
import math
import os
import subprocess as sp
import time
from pathlib import Path

import cv2
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_HEVC_FEAT_DECODER = str(
    _PROJECT_ROOT / "third_party" / "hevc_feature_decoder" / "bin" / "hevc"
)
_HEVC_FEAT_DECODER = os.environ.get("HEVC_FEAT_DECODER", _DEFAULT_HEVC_FEAT_DECODER)
_SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".hevc", ".h265", ".265"}


def ffprobe(filename, section):
    """Return ffprobe's JSON ``section`` output for the first video stream."""
    result = sp.run(
        ["ffprobe", "-v", "error", section, "-select_streams", "v:0",
         "-print_format", "json", filename],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class HevcFeatureReader:
    """Stream per-frame codec features from the HEVC feature decoder."""

    def __init__(self, filename, nb_frames, n_parallel):
        extension = os.path.splitext(filename)[1]
        self.viddict = (ffprobe(filename, "-show_streams").get("streams") or [{}])[0]
        if nb_frames is None:
            try:
                nb_frames = int(self.viddict.get("nb_frames"))
            except (TypeError, ValueError):
                nb_frames = len(ffprobe(filename, "-show_packets").get("packets") or [])
        self.nb_frames = nb_frames
        self.width = int(self.viddict.get("width"))
        self.height = int(self.viddict.get("height"))
        self.nb_ctus = math.ceil(self.width / 64.0) * math.ceil(self.height / 64.0)
        if self.viddict.get("pix_fmt") != "yuv420p":
            raise NameError("Expect a yuv420p input.")
        assert extension.lower() in _SUPPORTED_EXTENSIONS, (
            "Unknown decoder extension: " + extension.lower()
        )
        if not os.path.isfile(_HEVC_FEAT_DECODER) or not os.access(_HEVC_FEAT_DECODER, os.X_OK):
            raise FileNotFoundError(
                f"HEVC feature decoder not found or not executable at '{_HEVC_FEAT_DECODER}'.\n"
                f"Set env HEVC_FEAT_DECODER to the correct binary path."
            )
        self._devnull = open(os.devnull, "wb")
        self._proc = sp.Popen(
            [_HEVC_FEAT_DECODER, "-i", filename, "-p", str(n_parallel)],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=self._devnull,
        )

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.stdout.close()
            self._terminate(0.2)
        self._proc = None
        if self._devnull is not None:
            self._devnull.close()
            self._devnull = None

    def _terminate(self, timeout=1.0):
        if self._proc is None or self._proc.poll() is not None:
            return
        self._proc.terminate()
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.01)
            if self._proc.poll() is not None:
                break

    def _read_frame_data(self):
        luma = self.width * self.height
        yuv_size = luma + 2 * ((self.width >> 1) * (self.height >> 1))
        mv_size = (self.width >> 2) * (self.height >> 2) * 2
        ref_size = (self.width >> 2) * (self.height >> 2)
        size_size = (self.width >> 3) * (self.height >> 3)
        padding = (3 * self.width * self.height >> 2) - (mv_size * 5 + ref_size * 2)

        def read(num_bytes, dtype=np.uint8):
            buf = self._proc.stdout.read(num_bytes)
            if buf is None or len(buf) != num_bytes:
                self._terminate()
                raise RuntimeError(
                    f"Short read from decoder. Expected {num_bytes} bytes, got {0 if buf is None else len(buf)}."
                )
            return np.frombuffer(buf, dtype=dtype)

        try:
            yuv = read(yuv_size)
            mvs = [read(mv_size, np.int16) for _ in range(4)]
            refs = [read(ref_size) for _ in range(2)]
            size = read(mv_size)[:size_size]
            read(padding)
            meta = read(luma >> 2)
            residual = read(yuv_size)
            assert meta[0] == 4 and meta[1] == 2
        except Exception as exc:
            self._terminate()
            raise RuntimeError(
                "Failed to decode video. video information: ", self.viddict
            ) from exc
        return meta, yuv, *mvs, *refs, size, residual

    def _readFrame(self):
        meta, yuv, mv_x_l0, mv_y_l0, mv_x_l1, mv_y_l1, ref_l0, ref_l1, size, residual = (
            self._read_frame_data()
        )
        planes = (self.height + (self.height >> 1), self.width)
        quarter = (self.height >> 2, self.width >> 2)
        return (
            meta[2],
            meta[1024: 1024 + self.nb_ctus * 12],
            yuv.reshape(planes)[: self.height, : self.width],
            mv_x_l0.reshape(quarter),
            mv_y_l0.reshape(quarter),
            mv_x_l1.reshape(quarter),
            mv_y_l1.reshape(quarter),
            ref_l0.reshape(quarter),
            ref_l1.reshape(quarter),
            size.reshape(self.height >> 3, self.width >> 3),
            residual.reshape(planes)[: self.height, : self.width],
        )

    def nextFrame(self):
        """Yield one feature tuple per decoded frame."""
        for _ in range(self.nb_frames):
            yield self._readFrame()

    def _upsample_mv_to_hw(self, mv):
        """Nearest-neighbour upsample a quarter-resolution MV plane to the frame size."""
        return cv2.resize(mv, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
