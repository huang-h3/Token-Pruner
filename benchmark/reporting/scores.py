"""Explicit primary-score contracts for lmms-eval benchmark summaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class PrimaryMetric:
    """The native primary metric and normalization contract for one task."""

    suite: str
    metric: str
    native_maximum: float
    include_in_average: bool = True

    def normalize(self, value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        tolerance = 1e-9 * max(1.0, self.native_maximum)
        if number < -tolerance or number > self.native_maximum + tolerance:
            return None
        return min(self.native_maximum, max(0.0, number)) / self.native_maximum


# Exact task names take precedence over prefixes.
_EXACT: dict[str, PrimaryMetric] = {
    "motionbench_full": PrimaryMetric("motionbench", "motionbench_acc", 1.0),
    "nextqa_mc_test": PrimaryMetric("nextqa", "exact_match", 1.0),
    # WUPS is each open-ended task's primary metric and gets its own task_score.
    "nextqa_oe_test": PrimaryMetric(
        "nextqa_oe", "WUPS", 100.0, include_in_average=False
    ),
    "nextqa_oe_val": PrimaryMetric(
        "nextqa_oe", "WUPS", 100.0, include_in_average=False
    ),
    "vinoground": PrimaryMetric("vinoground", "vinoground_score_group", 100.0),
}

_PREFIXES: tuple[tuple[str, PrimaryMetric], ...] = (
    ("intphys2", PrimaryMetric("intphys2", "intphys2_accuracy", 1.0)),
    ("motionbench_", PrimaryMetric("motionbench", "motionbench_acc", 1.0)),
    ("mvbench_", PrimaryMetric("mvbench", "mvbench_accuracy", 100.0)),
    ("video_mmmu_", PrimaryMetric("video_mmmu", "mmmu_acc", 1.0)),
    ("vitatecs_", PrimaryMetric("vitatecs", "accuracy", 100.0)),
)

# Equal-weight aggregate; emitted only when all six suites are available.
CORE_SIX_SUITES = frozenset(
    {"intphys2", "motionbench", "mvbench", "nextqa", "video_mmmu", "vitatecs"}
)

#: Group names that lmms-eval expands to leaf tasks of one suite.
_SUITE_NAMES = CORE_SIX_SUITES | {"vinoground"}


def primary_metric(task: str) -> PrimaryMetric | None:
    """Return the explicit primary-score contract for an lmms task name."""

    normalized = str(task).strip().lower()
    exact = _EXACT.get(normalized)
    if exact is not None:
        return exact
    for prefix, spec in _PREFIXES:
        if normalized.startswith(prefix):
            return spec
    return None


def expected_average_suites(task_value: Any) -> set[str]:
    """Infer requested, average-eligible suites from the saved task parameter."""

    if task_value is None:
        return set()
    if isinstance(task_value, (list, tuple, set)):
        task_names = [str(item) for item in task_value]
    else:
        task_names = re.split(r"[+,]", str(task_value))

    suites: set[str] = set()
    for raw_name in task_names:
        name = raw_name.strip().lower()
        if not name:
            continue
        if name in _SUITE_NAMES:
            suites.add(name)
            continue
        spec = primary_metric(name)
        if spec is not None and spec.include_in_average:
            suites.add(spec.suite)
    return suites


def normalized_primary_task_scores(
    row: Mapping[str, Any],
) -> tuple[dict[str, float], list[str]]:
    """Extract normalized primary scores from ``metric__TASK__METRIC`` columns."""

    scores: dict[str, float] = {}
    invalid: list[str] = []
    for column, value in row.items():
        if not column.startswith("metric__"):
            continue
        parts = column.split("__")
        if len(parts) != 3:
            continue
        _, task, metric = parts
        spec = primary_metric(task)
        if spec is None or metric.lower() != spec.metric.lower():
            continue
        normalized = spec.normalize(value)
        if normalized is None:
            invalid.append(column)
        else:
            scores[task] = normalized
    return scores, sorted(invalid)


def average_suite_scores(task_scores: Mapping[str, float]) -> dict[str, float]:
    """Give every benchmark suite equal weight, regardless of leaf count."""

    grouped: dict[str, list[float]] = {}
    for task, score in task_scores.items():
        spec = primary_metric(task)
        if spec is None or not spec.include_in_average:
            continue
        grouped.setdefault(spec.suite, []).append(score)
    return {
        suite: sum(values) / len(values) for suite, values in sorted(grouped.items())
    }
