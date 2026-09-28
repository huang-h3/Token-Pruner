"""Result files and the incremental response log."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import uuid

import torch


@contextmanager
def atomic_output_path(path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path, text, *, encoding="utf-8", fsync=False):
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding=encoding) as handle:
            handle.write(text)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())


RESULT_SCHEMA_VERSION = 1
REQUEST_SCOPED_METRICS = {
    "hevc_encode_ms",
    "hevc_encode_paid_ms",
    "output_tokens",
    "hit_token_cap",
    "model_ttft_ms",
    "ttft_ms",
    "tpot_ms",
}


def weighted_metric_mean(records, key, *, weighted=True, value_of=None):
    """Sample-weighted mean of ``key`` over measurement records."""
    total = 0.0
    total_weight = 0
    for record in records:
        value = value_of(record, key) if value_of is not None else record.get(key)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        weight = max(1, int(record.get("sample_count", 1))) if weighted else 1
        total += value * weight
        total_weight += weight
    return total / total_weight if total_weight else None


def format_measurement_summary(values, *, include_generation=False):
    """Format the timing/memory summary line."""

    def value(name):
        return float(values.get(name) or 0.0)

    timing = (
        f"preprocessing={value('preprocessing_ms'):.1f}ms, "
        f"signal={value('signal_ms'):.1f}ms "
        f"(hevc={value('hevc_encode_ms'):.1f}, "
        f"to_mask={value('signal_to_mask_ms'):.1f}, "
        f"overhead={value('signal_overhead_ms'):.1f}), "
        f"model={value('model_ms'):.1f}ms, "
        f"inference={value('inference_ms'):.1f}ms"
    )
    if include_generation:
        timing += (
            f", TTFT={value('ttft_ms'):.1f}ms, "
            f"TPOT={value('tpot_ms'):.1f}ms"
        )
    if "decode_steps" in values:
        timing += f", decode_steps={value('decode_steps'):.1f}"
    timing += (
        f", e2e={value('e2e_ms'):.1f}ms "
        f"(paid={value('e2e_paid_ms'):.1f}ms)"
    )
    memory = (
        f"peak={value('peak_mem_mb'):.1f}MB, "
        f"delta={value('delta_peak_mem_mb'):.1f}MB"
    )
    return timing, memory


def _measurement_tensor(value, *, dtype=torch.float32):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.tensor(value, dtype=dtype)


def _cpu_record(value):
    """Detach tensors and move them to CPU for saving."""

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_record(item) for item in value]
    return value


def build_benchmark_result(
    *,
    timing,
    memory,
    params,
    accuracy=None,
    flops=None,
    request_measurements=None,
    batch_measurements=None,
    extra=None,
):
    """Build the in-memory result without writing it."""

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "params": dict(params),
        "timing": {
            key: _measurement_tensor(value)
            for key, value in timing.items()
        },
        "memory": {
            key: _measurement_tensor(value)
            for key, value in memory.items()
        },
    }
    if accuracy is not None:
        result["accuracy"] = {
            key: _measurement_tensor(value)
            for key, value in accuracy.items()
        }
    if flops is not None:
        result["flops"] = {
            key: _measurement_tensor(value, dtype=torch.float64)
            for key, value in flops.items()
        }
    if request_measurements is not None or batch_measurements is not None:
        result["measurements"] = {
            "requests": _cpu_record(request_measurements or []),
            "batches": _cpu_record(batch_measurements or []),
        }
    if extra:
        reserved = set(result).intersection(extra)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise RuntimeError(
                f"extra cannot override reserved result sections: {names}."
            )
        result.update(extra)
    return result


def save_benchmark_result(
    results_path,
    *,
    timing,
    memory,
    params,
    accuracy=None,
    flops=None,
    request_measurements=None,
    batch_measurements=None,
    extra=None,
):
    """Build a result and write it atomically."""

    results_path = Path(results_path).expanduser()
    result = build_benchmark_result(
        timing=timing,
        memory=memory,
        params=params,
        accuracy=accuracy,
        flops=flops,
        request_measurements=request_measurements,
        batch_measurements=batch_measurements,
        extra=extra,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(results_path) as temporary:
        torch.save(result, temporary)
    return results_path


RESUME_ENV = "TOKEN_PRUNER_RESUME_RESPONSES"
_DISABLED = frozenset({"0", "false", "no", "off"})


def resume_enabled():
    """Whether a completed response prefix may be replayed."""

    return os.environ.get(RESUME_ENV, "1").strip().lower() not in _DISABLED


class ResponseStore:
    def __init__(self, results_path):
        self.results_path = Path(results_path).expanduser()
        self.path = self.results_path.with_suffix(".responses.jsonl")
        self.records = []

    def begin(self, *, preserve=False):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if preserve:
            atomic_write_text(
                self.path,
                "".join(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                    for record in self.records
                ),
                fsync=True,
            )
        else:
            self.path.write_text("", encoding="utf-8")
            self.records = []

    def append(self, record):
        self.append_many([record])

    def append_many(self, records):
        """Append one completed generation batch with one flush and fsync."""

        records = list(records)
        if not records:
            return
        self.records.extend(records)
        with self.path.open("a", encoding="utf-8") as output_file:
            output_file.write("".join(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
                for record in records
            ))
            output_file.flush()
            os.fsync(output_file.fileno())

    def _read_jsonl(self):
        if not self.path.is_file():
            return None
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        records = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    return None
                break
        return records

    @staticmethod
    def _matches(record, expected):
        return (
            isinstance(record, dict)
            and isinstance(record.get("response"), str)
            and all(
                str(record.get(key)) == str(value)
                for key, value in expected.items()
            )
        )

    def load_prefix(self, expected_records):
        """Load a valid completed prefix, or return ``None`` if it is stale."""

        if not self.path.is_file():
            self.records = []
            return []
        records = self._read_jsonl()
        if not isinstance(records, list) or len(records) > len(expected_records):
            return None
        for record, expected in zip(records, expected_records):
            if not self._matches(record, expected):
                return None
        complete = 0
        while complete < len(records):
            batch = records[complete].get("batch_measurement")
            if not isinstance(batch, dict):
                return None
            indices = [int(index) for index in batch.get("request_indices", [])]
            if not indices or indices[0] != complete:
                return None
            end = complete + len(indices)
            if end > len(records):
                break
            group = records[complete:end]
            batch_id = batch.get("batch_id")
            if any(
                not isinstance(record.get("batch_measurement"), dict)
                or record["batch_measurement"].get("batch_id") != batch_id
                or record["batch_measurement"].get("request_indices") != indices
                or int(record.get("request_index", -1)) != indices[offset]
                for offset, record in enumerate(group)
            ):
                return None
            complete = end
        records = records[:complete]
        self.records = records
        return [record["response"] for record in records]

    def load(self, expected_records):
        responses = self.load_prefix(expected_records)
        if responses is None or len(responses) != len(expected_records):
            return None
        return responses
