"""Progress lines carrying the cumulative mean of persisted measurements."""

from collections import defaultdict
from math import isfinite
import sys

DISPLAY_METRICS = (
    ("delta_peak_mem_mb", "peak delta memory", "MB"),
    ("ttft_ms", "TTFT", "ms"),
    ("tpot_ms", "TPOT", "ms"),
    ("signal_ms", "signal", "ms"),
    ("token_reduce_ms", "token reduce", "ms"),
    ("model_ms", "model", "ms"),
    ("inference_ms", "inference", "ms"),
    ("e2e_ms", "e2e cold", "ms"),
)


class ProgressReporter:
    def __init__(self, total=None, *, every=5, label="samples", stream=None, enabled=True):
        self.total = int(total) if total is not None else None
        self.every = max(1, int(every))
        self.label = str(label)
        self.stream = stream or sys.stdout
        self.enabled = bool(enabled)
        self.completed = 0
        self._next_report = self.every
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)
        self._last_reported = -1

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    def update(self, measurement, *, count=1):
        count = int(count)
        for key, _label, _unit in DISPLAY_METRICS:
            try:
                number = float(measurement[key])
            except (KeyError, TypeError, ValueError):
                continue
            if isfinite(number):
                self._sums[key] += number * count
                self._counts[key] += count
        self.completed += count
        if self.completed >= self._next_report:
            self._render()
            while self._next_report <= self.completed:
                self._next_report += self.every

    def close(self):
        if self.completed and self.completed != self._last_reported:
            self._render()

    def snapshot(self):
        return {
            key: self._sums[key] / self._counts[key] if self._counts[key] else None
            for key, _label, _unit in DISPLAY_METRICS
        }

    def _render(self):
        self._last_reported = self.completed
        if not self.enabled:
            return
        values = self.snapshot()
        fields = " | ".join(
            f"{label}=" + ("N/A" if values[key] is None else f"{values[key]:.1f}{unit}")
            for key, label, unit in DISPLAY_METRICS
        )
        total = self.total if self.total is not None else "?"
        print(f"[{self.completed}/{total} {self.label}] {fields}",
              file=self.stream, flush=True)
