#!/usr/bin/env python3
"""Export compact classification and lmms-eval benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

from benchmark.reporting.io import (
    load_native_result, load_torch_result, result_params,
)
from benchmark.reporting.scores import (
    CORE_SIX_SUITES,
    average_suite_scores,
    expected_average_suites,
    normalized_primary_task_scores,
)
from token_pruner.records import REQUEST_SCOPED_METRICS, atomic_output_path


DEFAULT_CSV_NAME = "results_summary.csv"

#: Model ids that appear in result file names, longest first.
MODEL_NAME_PREFIXES = tuple(sorted(
    (
        "video_llava_hf", "video_llava_official", "llava_next_video",
        "qwen3_vl", "qwen3_vl_thinking", "internvl", "llava_onevision2", "vivit", "timesformer",
    ),
    key=len,
    reverse=True,
))

# Mean per-task score, placed immediately after the leading configuration.
AVG_SCORE_COLUMN = "avg_score"
MEAN6_SCORE_COLUMN = "mean6_score"
SCORE_COVERAGE_COLUMNS = (
    "n_suites",
    "expected_n_suites",
    "missing_suites",
    "invalid_primary_scores",
)

# Per-task mean output-token count, right after avg_score.
PER_TASK_OUTPUT_TOKENS_COLUMN = "avg_output_tokens_per_task"

# Identifier and provenance columns, placed at the end of the table.
TAIL_COLUMNS = ("experiment", "result_type", "source_path", "model_type")

# Timing metrics lead the measurement columns; matched by metric name.
HEADLINE_TIMING_METRICS = (
    # Stage totals. e2e = preprocessing + inference; inference = signal + model.
    "preprocessing_ms",
    "signal_ms",
    "model_ms",
    "inference_ms",
    "e2e_ms",
    # Signal breakdown.
    "hevc_encode_ms",
    "signal_to_mask_ms",
    "signal_overhead_ms",
    # Model breakdown.
    "token_reduce_ms",
    "vision_encoder_ms",
    "projector_ms",
    "prefill_ms",
    "decode_ms",
    "decode_steps",
    "vision_classification_forward_ms",
    # Latency as seen by a caller.
    "ttft_ms",
    "model_ttft_ms",
    "tpot_ms",
    "output_tokens",
)

# Configuration columns lead every CSV row.
CONFIG_COLUMNS = (
    "prune_mode",
    "i_mode",
    "p_mode",
    "prune_stage",
    "prune_layer",
    "k_keep_rate",
    "selection_seed",
    "run_mode",
    "pruning_strategy",
    "score_reduce",
    "folding_mode",
    "fold_block_wise",
    "fold_block_size",
    "fold_pooling",
    "fold_slots_per_block",
    "fold_min_slots_per_block",
    "partition_rule",
    "min_per_partition",
    "realized_keep_per_partition_min",
    "realized_keep_per_partition_max",
    "target_keep_total",
    "effective_keep_total",
    "effective_keep_rate",
    "num_frames",
    "num_samples",
    "requested_samples",
    "batch_size",
)


# Parameters needed to identify and compare runs.
CORE_PARAM_KEYS = (
    "model_type",
    "model_name",
    "model_path",
    "dataset",
    "dataset_split",
    "task",
    "run_mode",
    "pruning_strategy",
    "prune_stage",
    "prune_layer",
    "prune_mode",
    "i_mode",
    "p_mode",
    "selection_seed",
    "score_reduce",
    "folding_mode",
    "fold_block_wise",
    "fold_block_size",
    "fold_pooling",
    "fold_slots_per_block",
    "fold_min_slots_per_block",
    "partition_rule",
    "min_per_partition",
    "realized_keep_per_partition_min",
    "realized_keep_per_partition_max",
    "k_keep_rate",
    "target_keep_total",
    "effective_keep_total",
    "effective_keep_rate",
    "num_frames",
    "num_samples",
    "requested_samples",
    "batch_size",
    "quantization",
    "vision_backend",
    "device",
    "timing_schema",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize saved classification and lmms-eval results as CSV."
    )
    parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="Directory containing benchmark .pt result files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"CSV path (default: RESULTS_ROOT/{DEFAULT_CSV_NAME}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when an unreadable .pt file is encountered.",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Write one row per run instead of the default transposed layout.",
    )
    return parser.parse_args()


def safe_name(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def task_suite_name(value: Any) -> Any:
    """Serialize the task parameter, keeping subtask names verbatim."""
    if value is None:
        return None
    return (
        ",".join(str(item) for item in value)
        if isinstance(value, (list, tuple))
        else str(value)
    )


def csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value) and value.numel() == 1:
        return value.detach().cpu().item()
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False, sort_keys=True)
    return str(value)


def numeric_values(value: Any) -> list[float] | None:
    if torch.is_tensor(value):
        if value.dtype == torch.bool:
            return None
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            nested = numeric_values(item)
            if nested is None:
                return None
            flattened.extend(nested)
        return flattened
    return None


def finite_values(value: Any) -> list[float]:
    values = numeric_values(value)
    if values is None:
        return []
    return [number for number in values if math.isfinite(number)]


def detect_result_type(result: dict[str, Any], params: dict[str, Any]) -> str:
    if isinstance(result.get("accuracy"), dict):
        return "classification"
    if params.get("answers_path"):
        return "lmms_eval"
    return "unsupported"


def add_common_params(row: dict[str, Any], params: dict[str, Any]) -> None:
    for key in CORE_PARAM_KEYS:
        if key not in params:
            continue
        value = task_suite_name(params[key]) if key == "task" else params[key]
        row[key] = csv_scalar(value)


def add_accuracy(row: dict[str, Any], result: dict[str, Any]) -> None:
    accuracy = result.get("accuracy")
    if not isinstance(accuracy, dict):
        return
    for mode, value in accuracy.items():
        scores = numeric_values(value)
        if scores is None:
            continue
        for index, score in enumerate(scores):
            label = "top1" if index == 0 else "top5" if index == 1 else f"value_{index}"
            row[f"accuracy__{safe_name(mode)}__{label}"] = score


def _measurement_records(result):
    measurements = result.get("measurements")
    if not isinstance(measurements, dict):
        return []
    batches = measurements.get("batches")
    requests = measurements.get("requests")
    batches = (
        [record for record in batches if isinstance(record, dict)]
        if isinstance(batches, list)
        else []
    )
    requests = (
        [record for record in requests if isinstance(record, dict)]
        if isinstance(requests, list)
        else []
    )
    records = [
        {
            key: value
            for key, value in record.items()
            if key not in REQUEST_SCOPED_METRICS
        }
        for record in batches
    ]
    records.extend(
        {
            key: value
            for key, value in record.items()
            if key in REQUEST_SCOPED_METRICS or key in ("variant", "task")
        }
        for record in requests
    )
    return records


def _output_token_task_group(name: str) -> str:
    """Collapse expanded leaves only for output-length presentation."""
    normalized = str(name).lower()
    for suite in (
        "nextqa",
        "mvbench",
        "motionbench",
        "vinoground",
        "intphys2",
        "vitatecs",
        "video_mmmu",
    ):
        if normalized == suite or normalized.startswith(f"{suite}_"):
            return suite
    return normalized


def add_measurements(row: dict[str, Any], result: dict[str, Any]) -> None:
    records = _measurement_records(result)
    if not records:
        return
    variants = {str(record.get("variant") or "unknown") for record in records}
    grouped = {}
    grouped_scopes = {}
    ignored = {
        "variant",
        "measurement_scope",
        "batch_index",
        "batch_id",
        "request_index",
        "request_id",
        "sample_count",
        "batch_size",
        "batch_size_effective",
        "request_indices",
        "batch_output_tokens",
        "doc_id",
        "task",
        "split",
        "hevc_cache_hit",
    }
    task_output_tokens: dict[str, list[float]] = {}
    for record in records:
        variant = str(record.get("variant") or "unknown")
        weight = max(1, int(record.get("sample_count", 1)))
        for name, value in record.items():
            if name in ignored or isinstance(value, bool):
                continue
            values = finite_values(value)
            if len(values) != 1:
                continue
            if name == "output_tokens" and record.get("task"):
                task_output_tokens.setdefault(
                    _output_token_task_group(str(record["task"])), []
                ).extend(values * weight)
            grouped.setdefault((variant, str(name)), []).extend(values * weight)
            grouped_scopes.setdefault(
                (variant, str(name)),
                str(record.get("measurement_scope") or "unscoped"),
            )

    if task_output_tokens:
        row[PER_TASK_OUTPUT_TOKENS_COLUMN] = json.dumps(
            {
                task: statistics.fmean(values)
                for task, values in sorted(task_output_tokens.items())
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    memory_names = {"peak_mem_mb", "delta_peak_mem_mb"}
    for (variant, name), values in grouped.items():
        section = "memory" if name in memory_names else "timing"
        variant_part = f"{safe_name(variant)}__" if len(variants) > 1 else ""
        prefix = f"{section}__{variant_part}{safe_name(name)}"
        row[f"{prefix}__mean"] = statistics.fmean(values)
        row.setdefault("_measurement_values", {})[prefix] = list(values)
        row.setdefault("_measurement_scopes", {})[prefix] = grouped_scopes[
            (variant, name)
        ]
        if section == "memory":
            row[f"{prefix}__max"] = max(values)


def metric_column(task: str, metric: str) -> str:
    metric_name, separator, filter_name = metric.partition(",")
    column = f"metric__{safe_name(task)}__{safe_name(metric_name)}"
    if separator and filter_name and filter_name != "none":
        column += f"__{safe_name(filter_name)}"
    return column


LMMS_TABLE_SUFFIXES = (
    "_stderr",
    "_stderr_clt",
    "_stderr_clustered",
    "_expected_accuracy",
    "_consensus_accuracy",
    "_internal_variance",
    "_consistency_rate",
)


def add_lmms_metrics(row: dict[str, Any], result_dir: Path, stem: str) -> None:
    """Add numeric cells from lmms-eval's task table, but no artifact metadata."""
    artifact_dir = result_dir / "lmms_artifacts" / stem
    native_result_path, payload = load_native_result(artifact_dir)
    if native_result_path is None:
        return
    task_results = payload["results"]

    for task, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue
        for raw_name, value in metrics.items():
            name, separator, filter_name = str(raw_name).partition(",")
            if (
                str(task) == "vinoground"
                and name == "vinoground_score"
                and isinstance(value, (list, tuple))
                and len(value) == 3
            ):
                for component, component_value in zip(
                    ("text", "video", "group"),
                    value,
                ):
                    if (
                        isinstance(component_value, bool)
                        or not isinstance(component_value, (int, float))
                    ):
                        continue
                    component_metric = f"{name}_{component}"
                    if separator:
                        component_metric += f",{filter_name}"
                    row[metric_column(str(task), component_metric)] = component_value
                continue
            if name.startswith("paired_") or name.endswith(LMMS_TABLE_SUFFIXES):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue

            row[metric_column(str(task), str(raw_name))] = value


def build_row(path: Path, results_root: Path) -> dict[str, Any]:
    result = load_torch_result(path)
    params = result_params(result)
    result_type = detect_result_type(result, params)
    row: dict[str, Any] = {
        "experiment": path.stem,
        "result_type": result_type,
        "source_path": str(path.relative_to(results_root)),
    }
    add_common_params(row, params)
    add_accuracy(row, result)
    add_measurements(row, result)
    if result_type == "lmms_eval":
        add_lmms_metrics(row, path.parent, path.stem)
    add_average_score(row)
    return row


def add_average_score(row: dict[str, Any]) -> None:
    """Add explicit, normalized primary scores and a coverage-safe average."""

    derived_columns = {
        AVG_SCORE_COLUMN,
        MEAN6_SCORE_COLUMN,
        *SCORE_COVERAGE_COLUMNS,
    }
    for column in list(row):
        if column in derived_columns or column.startswith(
            ("task_score__", "suite_score__")
        ):
            row.pop(column, None)

    task_scores, invalid = normalized_primary_task_scores(row)
    suite_scores = average_suite_scores(task_scores)
    for task, score in sorted(task_scores.items()):
        row[f"task_score__{task}"] = score
    for suite, score in sorted(suite_scores.items()):
        row[f"suite_score__{suite}"] = score

    # Declared suites catch missing artifacts; observed suites are included too.
    expected = expected_average_suites(row.get("task")) | set(suite_scores)
    missing = sorted(expected.difference(suite_scores))
    row["n_suites"] = len(suite_scores)
    row["expected_n_suites"] = len(expected)
    if missing:
        row["missing_suites"] = "+".join(missing)
    if invalid:
        row["invalid_primary_scores"] = "+".join(invalid)
    if expected and not missing:
        row[AVG_SCORE_COLUMN] = statistics.fmean(
            suite_scores[suite] for suite in sorted(expected)
        )
    if CORE_SIX_SUITES.issubset(suite_scores):
        row[MEAN6_SCORE_COLUMN] = statistics.fmean(
            suite_scores[suite] for suite in sorted(CORE_SIX_SUITES)
        )

    # Classification top-1 is already a fraction; the scale is never inferred.
    if task_scores or suite_scores:
        return
    accuracy = [
        float(value)
        for key, value in row.items()
        if key.startswith("accuracy__")
        and key.endswith("__top1")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    ]
    if accuracy:
        score = statistics.fmean(accuracy)
        row["task_score__classification"] = score
        row["suite_score__classification"] = score
        row["n_suites"] = 1
        row["expected_n_suites"] = 1
        row[AVG_SCORE_COLUMN] = score


def _task_case_name(row):
    experiment = str(row.get("experiment", ""))
    separators = tuple(f"_{model_id}_" for model_id in MODEL_NAME_PREFIXES)
    for separator in separators:
        task_prefix, sep, suffix = experiment.partition(separator)
        if sep and "+" not in task_prefix:
            return f"{sep[1:]}{suffix}"
    return experiment


def merge_task_rows(rows):
    """Merge task-first lmms leaves into one sample-weighted model-case row."""
    groups = {}
    output = []
    for row in rows:
        if row.get("result_type") != "lmms_eval":
            clean = dict(row)
            clean.pop("_measurement_values", None)
            output.append(clean)
            continue
        case_name = _task_case_name(row)
        key = (
            case_name,
            row.get("model_name"),
            row.get("model_path"),
            row.get("run_mode"),
            row.get("batch_size"),
        )
        groups.setdefault(key, []).append(row)

    for (case_name, *_identity), leaves in groups.items():
        merged = dict(leaves[0])
        merged["experiment"] = case_name
        merged["task"] = "+".join(sorted(str(row.get("task", "")) for row in leaves))
        merged["source_path"] = json.dumps(
            [row.get("source_path") for row in leaves], ensure_ascii=False
        )
        distributions = {}
        merged_per_task_output_tokens = {}
        for row in leaves:
            for prefix, values in row.get("_measurement_values", {}).items():
                distributions.setdefault(prefix, []).extend(values)
            per_task = row.get(PER_TASK_OUTPUT_TOKENS_COLUMN)
            if isinstance(per_task, str) and per_task:
                for task, mean in json.loads(per_task).items():
                    merged_per_task_output_tokens[task] = mean
        merged.pop("_measurement_values", None)
        merged_scopes = {}
        for row in leaves:
            merged_scopes.update(row.get("_measurement_scopes", {}))
        if merged_scopes:
            merged["_measurement_scopes"] = merged_scopes
        if merged_per_task_output_tokens:
            merged[PER_TASK_OUTPUT_TOKENS_COLUMN] = json.dumps(
                merged_per_task_output_tokens,
                ensure_ascii=False,
                sort_keys=True,
            )
        all_columns = {column for row in leaves for column in row}
        for column in all_columns:
            if column.startswith("metric__"):
                for row in leaves:
                    if column in row:
                        merged[column] = row[column]
                        break
            elif column.endswith("__mean") and column.startswith(("timing__", "memory__")):
                values = [row[column] for row in leaves if column in row]
                if values:
                    merged[column] = statistics.fmean(values)
            elif column.endswith("__max") and column.startswith("memory__"):
                merged[column] = max(row[column] for row in leaves if column in row)
        for prefix, values in distributions.items():
            merged[f"{prefix}__mean"] = statistics.fmean(values)
            if prefix.startswith("memory__"):
                merged[f"{prefix}__max"] = max(values)
        add_average_score(merged)
        output.append(merged)
    return output


def _headline_timing_columns(all_columns, metric: str) -> list[str]:
    """Timing mean columns for ``metric``, with or without a variant prefix."""

    matches = []
    for column in all_columns:
        if not column.startswith("timing__"):
            continue
        if column.endswith("__batch__mean"):
            middle = column[len("timing__") : -len("__batch__mean")]
        elif column.endswith("__mean"):
            middle = column[len("timing__") : -len("__mean")]
        else:
            continue
        if middle == safe_name(metric) or middle.endswith(f"__{safe_name(metric)}"):
            matches.append(column)
    return matches


def ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Put scientific configuration first, followed by results and provenance."""

    all_columns = {
        column
        for row in rows
        for column in row
        if not column.startswith("_")
    }
    columns: list[str] = []

    # 1. Configuration needed to interpret and compare each experiment.
    selected = [column for column in CONFIG_COLUMNS if column in all_columns]
    columns.extend(selected)
    all_columns.difference_update(selected)

    # 2. Explicitly normalized aggregate and its completeness diagnostics.
    selected = [
        column
        for column in (
            AVG_SCORE_COLUMN,
            MEAN6_SCORE_COLUMN,
            *SCORE_COVERAGE_COLUMNS,
        )
        if column in all_columns
    ]
    columns.extend(selected)
    all_columns.difference_update(selected)

    # 3. Per-task output-token means.
    if PER_TASK_OUTPUT_TOKENS_COLUMN in all_columns:
        columns.append(PER_TASK_OUTPUT_TOKENS_COLUMN)
        all_columns.difference_update({PER_TASK_OUTPUT_TOKENS_COLUMN})

    # 4. Headline timing means.
    selected = [
        column
        for metric in HEADLINE_TIMING_METRICS
        for column in sorted(_headline_timing_columns(all_columns, metric))
    ]
    columns.extend(selected)
    all_columns.difference_update(selected)

    # 5. Memory and remaining timing measurements.
    for prefix in ("timing__", "memory__"):
        selected = sorted(column for column in all_columns if column.startswith(prefix))
        columns.extend(selected)
        all_columns.difference_update(selected)

    # 6. Normalized suite/task scores, followed by native detail metrics.
    for prefix in ("suite_score__", "task_score__", "accuracy__", "metric__"):
        selected = sorted(column for column in all_columns if column.startswith(prefix))
        columns.extend(selected)
        all_columns.difference_update(selected)

    # 7. Remaining parameters, excluding trailing provenance identifiers.
    tail = set(TAIL_COLUMNS)
    columns.extend(sorted(all_columns.difference(tail)))

    # 8. Provenance identifiers at the end.
    columns.extend(column for column in TAIL_COLUMNS if column in all_columns)
    return columns


def _run_labels(rows: list[dict[str, Any]]) -> list[str]:
    """One unique column header per run for the transposed layout."""

    labels = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        label = str(row.get("experiment") or "").strip() or f"run_{index}"
        count = seen.get(label, 0)
        seen[label] = count + 1
        labels.append(label if count == 0 else f"{label}#{count + 1}")
    return labels


def _csv_rows_with_measurement_scopes(rows: list[dict[str, Any]]):
    """Project batch-scoped timing/memory columns with an explicit suffix."""

    projected = []
    for row in rows:
        scopes = row.get("_measurement_scopes", {})
        output = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for base, scope in scopes.items():
            if scope != "batch":
                continue
            mean_key = f"{base}__mean"
            max_key = f"{base}__max"
            if mean_key in output:
                output[f"{base}__batch__mean"] = output.pop(mean_key)
            if max_key in output:
                output[f"{base}__batch__max"] = output.pop(max_key)
        projected.append(output)
    return projected


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    transpose: bool = True,
) -> None:
    """Write the summary table, transposed by default (one row per field)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _csv_rows_with_measurement_scopes(rows)
    columns = ordered_columns(rows)
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            if not transpose:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=columns,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
                return
            writer = csv.writer(handle)
            writer.writerow(["field", *_run_labels(rows)])
            for column in columns:
                writer.writerow(
                    [column, *(row.get(column, "") for row in rows)]
                )


def main() -> int:
    args = parse_args()
    root = args.results_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / DEFAULT_CSV_NAME
    )
    if not root.is_dir():
        raise NotADirectoryError(f"results root does not exist: {root}")

    rows: list[dict[str, Any]] = []
    excluded_directories = {"profiler", "profiler_logs", "hevc_tmp"}
    for path in sorted(root.rglob("*.pt")):
        if set(path.relative_to(root).parts[:-1]).intersection(excluded_directories):
            continue
        try:
            row = build_row(path, root)
            if row["result_type"] != "unsupported":
                rows.append(row)
        except Exception as exc:
            message = f"Could not summarize {path}: {type(exc).__name__}: {exc}"
            if args.strict:
                raise RuntimeError(message) from exc
            print(f"warning: {message}", file=sys.stderr)

    rows = merge_task_rows(rows)
    rows.sort(
        key=lambda row: (
            str(row.get("result_type", "")),
            str(row.get("model_type", "")),
            str(row.get("task", "")),
            str(row.get("experiment", "")),
        )
    )
    if not rows:
        print(f"warning: no readable .pt results found under {root}", file=sys.stderr)
    write_csv(output, rows, transpose=not args.no_transpose)
    print(f"Saved result summary: {output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
