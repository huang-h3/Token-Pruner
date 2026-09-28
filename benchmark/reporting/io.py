"""Shared readers for saved benchmark results; no model or plotting imports."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

from benchmark.reporting.scores import primary_metric

RESPONSE_SUFFIX = ".responses.jsonl"


def response_files(root: Path) -> list[Path]:
    root = Path(root)
    return [root] if root.is_file() else sorted(root.rglob(f"*{RESPONSE_SUFFIX}"))


def iter_jsonl(path: Path, *, allow_truncated_tail: bool = False):
    """Stream JSON records; report corruption instead of silently skipping it."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if allow_truncated_tail and not any(rest.strip() for rest in handle):
                    warnings.warn(f"Ignoring incomplete final record in {path}:{line_number}",
                                  stacklevel=2)
                    return
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            yield record


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def document_key(record: dict) -> tuple:
    key = tuple(record.get(name) for name in ("task", "split", "doc_id"))
    if any(value is None for value in key):
        raise ValueError("Response is missing task, split, or doc_id")
    return key


def iter_responses(path: Path, *, stats: dict | None = None):
    """Yield each document once; optional stats records the duplicate count."""
    seen = set()
    if stats is not None:
        stats["duplicates"] = 0
    for record in iter_jsonl(path, allow_truncated_tail=True):
        key = document_key(record)
        if key in seen:
            if stats is not None:
                stats["duplicates"] += 1
            continue
        seen.add(key)
        yield {**record, **dict(zip(("task", "split", "doc_id"), key))}


def artifact_dir(result_path: Path) -> Path:
    path = Path(result_path)
    stem = path.name.removesuffix(RESPONSE_SUFFIX) if path.name.endswith(RESPONSE_SUFFIX) else path.stem
    return path.parent / "lmms_artifacts" / stem


def load_native_result(directory: Path) -> tuple[Path | None, dict]:
    paths = list(Path(directory).rglob("*_results.json"))
    if not paths:
        return None, {}
    path = max(paths, key=lambda p: (p.stat().st_mtime_ns, str(p)))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"Invalid lmms-eval result table: {path}")
    return path, payload


def primary_scores(payload: dict) -> dict[str, dict]:
    """Primary scores in percent, with native values and sample counts intact."""
    counts = payload.get("n-samples") or {}
    scores = {}
    for task, values in (payload.get("results") or {}).items():
        metric = primary_metric(task)
        if metric is None or not isinstance(values, dict):
            continue
        raw = values.get(f"{metric.metric},none")
        normalized = metric.normalize(raw)
        if normalized is not None:
            scores[task] = {
                "metric": metric.metric, "raw": float(raw),
                "score": 100.0 * normalized,
                "effective": int((counts.get(task) or {}).get("effective") or 0),
            }
    return scores


def arm_accuracy(responses_path: Path) -> dict[str, float]:
    """Primary scores of one arm, read from its native lmms-eval result table."""

    _, payload = load_native_result(artifact_dir(responses_path))
    return {task: value["score"] for task, value in primary_scores(payload).items()}


def arm_identity(record: dict) -> tuple[str, float]:
    """Return (selector, keep rate); the selector is scope plus the I/P modes."""

    if str(record.get("run_mode")) == "full":
        return "full", 1.0
    rate = float(record.get("k_keep_rate") or 0.0)
    p_mode = record.get("p_mode")
    if p_mode in ("random", "uniform"):
        return f"{record.get('prune_mode')}_{p_mode}", rate
    if p_mode == "global_folder":
        return "global_folder", rate
    if p_mode == "shared_folding":
        return f"shared-{record.get('folding_mode', 'unknown')}", rate
    return f"{record.get('prune_mode')}_i-{record.get('i_mode')}_p-{p_mode}", rate


def result_params(result):
    params = result.get("params")
    return params if isinstance(params, dict) else {}


def load_torch_result(path):
    import torch

    path = Path(path)
    result = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(result, dict):
        raise TypeError(f"expected a dictionary, got {type(result).__name__}")
    return result
