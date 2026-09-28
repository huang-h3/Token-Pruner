"""Timing helpers and measurement records for model runs."""

from dataclasses import dataclass
from statistics import fmean
import time

import torch

from .records import (
    REQUEST_SCOPED_METRICS,
    format_measurement_summary,
    weighted_metric_mean,
)


def cuda_sync(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class SpanAccumulator:
    def __init__(self, device=None):
        self.default_cuda = self._use_cuda(device)
        self._open = []
        self.spans = []

    @staticmethod
    def _use_cuda(device):
        if device is None:
            return None
        device = torch.device(device)
        return device.type == "cuda" and torch.cuda.is_available()

    def clear(self):
        self._open.clear()
        self.spans.clear()

    def begin(self, device=None):
        use_cuda = self._use_cuda(device)
        if use_cuda is None:
            use_cuda = bool(self.default_cuda)
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.perf_counter()
        self._open.append((use_cuda, start))

    def end(self):
        use_cuda, start = self._open.pop()
        if use_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.spans.append((start, end))
        else:
            self.spans.append((time.perf_counter() - start) * 1000.0)

    def record(self, fn, device=None):
        self.begin(device)
        try:
            return fn()
        finally:
            self.end()

    def elapsed(self):
        return [
            span[0].elapsed_time(span[1]) if isinstance(span, tuple) else span
            for span in self.spans
        ]

    def total_ms(self):
        return sum(self.elapsed())


@dataclass
class TimedInferenceResult:
    output: object
    elapsed_ms: float
    peak_mem_mb: float
    delta_peak_mem_mb: float


def timed_inference_call(fn, device):
    device = torch.device(device)
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    cuda_sync(device)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        start_mem = torch.cuda.memory_allocated(device)
    else:
        start_mem = 0

    start = time.perf_counter()
    with torch.inference_mode():
        output = fn()
    cuda_sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_mem = torch.cuda.max_memory_allocated(device) if use_cuda else 0
    scale = 1024**2
    return TimedInferenceResult(
        output=output,
        elapsed_ms=elapsed_ms,
        peak_mem_mb=peak_mem / scale,
        delta_peak_mem_mb=(peak_mem - start_mem) / scale,
    )


GENERATION_TIMING_KEYS = (
    # Stage totals: e2e = preprocessing + inference; inference = signal + model.
    "preprocessing_ms",
    "signal_ms",
    "model_ms",
    "inference_ms",
    "e2e_ms",
    # Signal breakdown (all three sum to signal_ms).
    "hevc_encode_ms",
    "signal_to_mask_ms",
    "signal_overhead_ms",
    # Model breakdown (token_reduce_ms and the module spans are nested).
    "vlm_input_adapt_ms",
    "vlm_generate_ms",
    "token_reduce_ms",
    "vision_encoder_ms",
    "projector_ms",
    "prefill_ms",
    "decode_ms",
    "decode_steps",
    "model_ttft_ms",
    "ttft_ms",
    "tpot_ms",
    "output_tokens",
    # Batch fraction that stopped on the token budget rather than EOS.
    "hit_token_cap",
    # Encode time actually paid on this run; zero on a cache hit.
    "hevc_encode_paid_ms",
    "signal_paid_ms",
    "inference_paid_ms",
    "e2e_paid_ms",
    # Physical batch wall clock; e2e_wall_ms - e2e_paid_ms is unattributed time.
    "e2e_wall_ms",
)
GENERATION_MEMORY_KEYS = (
    "peak_mem_mb",
    "delta_peak_mem_mb",
)

#: Spans and peaks a batch owns as a whole.
BATCH_SCOPED_METRICS = frozenset({
    "preprocessing_ms",
    "signal_to_mask_ms",
    "signal_overhead_ms",
    "signal_ms",
    "signal_paid_ms",
    "vlm_input_adapt_ms",
    "token_reduce_ms",
    "vision_encoder_ms",
    "projector_ms",
    "vlm_generate_ms",
    "prefill_ms",
    "decode_ms",
    "e2e_wall_ms",
    "peak_mem_mb",
    "delta_peak_mem_mb",
    "model_ms",
    "inference_ms",
    "inference_paid_ms",
    "e2e_ms",
    "e2e_paid_ms",
})


def new_generation_measurements():
    timing = {key: [] for key in GENERATION_TIMING_KEYS}
    memory = {key: [] for key in GENERATION_MEMORY_KEYS}
    return timing, memory


def signal_timing(preparation, wall_ms=None):
    """Split the signal stage into encode, selection, and untimed remainder."""

    encode_ms = float(getattr(preparation, "hevc_encode_ms", 0.0))
    encode_paid_ms = float(getattr(preparation, "hevc_encode_paid_ms", encode_ms))
    to_mask_ms = float(getattr(preparation, "selector_compute_ms", 0.0))
    overhead_ms = 0.0
    if wall_ms is not None:
        overhead_ms = max(0.0, float(wall_ms) - encode_paid_ms - to_mask_ms)
    return {
        "hevc_encode_ms": encode_ms,
        "hevc_encode_paid_ms": encode_paid_ms,
        "signal_to_mask_ms": to_mask_ms,
        "signal_overhead_ms": overhead_ms,
        "signal_ms": encode_ms + to_mask_ms + overhead_ms,
        "signal_paid_ms": encode_paid_ms + to_mask_ms + overhead_ms,
    }


def _stage_totals(record):
    """Derive model/inference/e2e totals from the stage breakdown."""

    record["model_ms"] = record["vlm_input_adapt_ms"] + record["vlm_generate_ms"]
    record["inference_ms"] = record["signal_ms"] + record["model_ms"]
    record["inference_paid_ms"] = record["signal_paid_ms"] + record["model_ms"]
    record["e2e_ms"] = record["preprocessing_ms"] + record["inference_ms"]
    record["e2e_paid_ms"] = record["preprocessing_ms"] + record["inference_paid_ms"]
    return record


def build_request_measurement(
    *,
    variant,
    preprocessing_ms,
    inference: TimedInferenceResult,
    fine,
    preparation=None,
    vlm_input_adapt_ms=0.0,
    token_reduce_ms=0.0,
    signal_prepare_wall_ms=None,
    e2e_wall_ms=None,
    batch_scoped=False,
):
    """Build one request-level measurement record."""

    signal = signal_timing(preparation, signal_prepare_wall_ms)
    cache_hits = getattr(preparation, "hevc_cache_hits", []) if preparation else []
    record = {
        "variant": str(variant),
        "measurement_scope": "request",
        "preprocessing_ms": float(preprocessing_ms),
        "hevc_encode_ms": signal["hevc_encode_ms"],
        "hevc_encode_paid_ms": signal["hevc_encode_paid_ms"],
        "hevc_cache_hit": bool(cache_hits[0]) if cache_hits else False,
        "signal_to_mask_ms": signal["signal_to_mask_ms"],
        "signal_overhead_ms": signal["signal_overhead_ms"],
        "signal_ms": signal["signal_ms"],
        "signal_paid_ms": signal["signal_paid_ms"],
        "vlm_input_adapt_ms": float(vlm_input_adapt_ms),
        "token_reduce_ms": float(token_reduce_ms),
        "vision_encoder_ms": float(fine.get("vision_encoder_ms", 0.0)),
        "projector_ms": float(fine.get("projector_ms", 0.0)),
        "vlm_generate_ms": float(inference.elapsed_ms),
        "prefill_ms": float(fine.get("prefill_ms", 0.0)),
        "decode_ms": float(fine.get("decode_ms", 0.0)),
        "output_tokens": int(fine.get("output_tokens", 0)),
        "hit_token_cap": float(bool(fine.get("hit_token_cap", False))),
        "model_ttft_ms": fine.get("model_ttft_ms"),
        "ttft_ms": fine.get("ttft_ms"),
        "tpot_ms": fine.get("tpot_ms"),
        "e2e_wall_ms": float(e2e_wall_ms) if e2e_wall_ms is not None else None,
        "peak_mem_mb": float(inference.peak_mem_mb),
        "delta_peak_mem_mb": float(inference.delta_peak_mem_mb),
    }
    artifact_keys = getattr(preparation, "hevc_artifact_keys", []) if preparation else []
    if artifact_keys:
        record["hevc_artifact_key"] = str(artifact_keys[0])
    if batch_scoped:
        return {key: value for key, value in record.items()
                if key not in BATCH_SCOPED_METRICS}
    return _stage_totals(record)


def build_batch_measurement(
    *,
    variant,
    preprocessing_ms,
    inference,
    fine,
    preparation=None,
    vlm_input_adapt_ms=0.0,
    token_reduce_ms=0.0,
    signal_prepare_wall_ms=None,
    e2e_wall_ms=None,
    sample_count=1,
):
    """Build a model-batch record; its GPU spans are never divided by B."""

    signal = signal_timing(preparation, signal_prepare_wall_ms)
    record = {
        "variant": str(variant),
        "measurement_scope": "batch",
        "sample_count": int(sample_count),
        "preprocessing_ms": float(preprocessing_ms),
        "hevc_cache_hit": bool(getattr(preparation, "hevc_cache_hits", [])) and all(
            preparation.hevc_cache_hits
        ),
        **signal,
        "vlm_input_adapt_ms": float(vlm_input_adapt_ms),
        "token_reduce_ms": float(token_reduce_ms),
        "vision_encoder_ms": float(fine.get("vision_encoder_ms", 0.0)),
        "projector_ms": float(fine.get("projector_ms", 0.0)),
        "vlm_generate_ms": float(inference.elapsed_ms),
        "prefill_ms": float(fine.get("prefill_ms", 0.0)),
        "decode_ms": float(fine.get("decode_ms", 0.0)),
        "decode_steps": int(fine.get("decode_steps", 0)),
        "e2e_wall_ms": (
            float(e2e_wall_ms) if e2e_wall_ms is not None else None
        ),
        "peak_mem_mb": float(inference.peak_mem_mb),
        "delta_peak_mem_mb": float(inference.delta_peak_mem_mb),
    }
    return _stage_totals(record)


def record_measurement(timing, memory, record):
    """Update the flat view, keeping request and batch scopes apart."""

    request_scoped = record.get("measurement_scope") == "request"
    timing_keys = (
        REQUEST_SCOPED_METRICS
        if request_scoped
        else set(GENERATION_TIMING_KEYS) - REQUEST_SCOPED_METRICS
    )
    for key in timing_keys:
        value = record.get(key)
        if value is not None:
            timing[key].append(value)
    for key in (() if request_scoped else GENERATION_MEMORY_KEYS):
        value = record.get(key)
        if value is not None:
            memory[key].append(value)


def _mean(values):
    return fmean(values) if values else 0.0


#: Per request, but printed in a batch total; a per-sample mean would not sum.
SUMMARISED_AT_BATCH_SCOPE = {"hevc_encode_ms", "hevc_encode_paid_ms"}


def print_generation_summary(
    timing,
    memory,
    *,
    request_measurements=None,
    batch_measurements=None,
):
    def mean(key, values):
        request_scoped = (
            key in REQUEST_SCOPED_METRICS
            and key not in SUMMARISED_AT_BATCH_SCOPE
        )
        records = (
            request_measurements
            if request_scoped and request_measurements
            else batch_measurements
        )
        if not records:
            return _mean(values[key])
        # Request records already describe exactly one sample.
        average = weighted_metric_mean(
            records,
            key,
            weighted=not (request_scoped and records is request_measurements),
        )
        return 0.0 if average is None else average

    values = {
        key: mean(key, memory if key.endswith("_mem_mb") else timing)
        for key in (
            "preprocessing_ms",
            "signal_ms",
            "hevc_encode_ms",
            "signal_to_mask_ms",
            "signal_overhead_ms",
            "model_ms",
            "inference_ms",
            "ttft_ms",
            "tpot_ms",
            "decode_steps",
            "e2e_ms",
            "e2e_paid_ms",
            "peak_mem_mb",
            "delta_peak_mem_mb",
        )
    }
    timing_line, memory_line = format_measurement_summary(
        values,
        include_generation=True,
    )
    print(f"Timing: {timing_line}")
    print(f"Memory: {memory_line}")
