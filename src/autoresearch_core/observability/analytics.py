"""Pure analytics helpers for loop and multi-loop Mission Control views."""

from __future__ import annotations

from collections import Counter
from typing import Any, cast

from autoresearch_core.contracts import AnalyticsSnapshot


def compute_loop_analytics(
    *,
    progress: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
) -> AnalyticsSnapshot:
    """Compute high-signal per-loop aggregate stats from read-model payloads."""
    source_counts = Counter(str(item.get("proposal_source") or "unknown") for item in decisions)
    status_counts = Counter(str(item.get("status") or "unknown") for item in decisions)
    mutation_group_counts = Counter(
        str(group)
        for item in decisions
        for group in (item.get("groups") or [])
    )
    model_family_wins = Counter(
        str(item.get("model") or "unknown")
        for item in leaderboard
        if item.get("decision_status") in {"keep", "baseline", "completed"}
    )
    best_iteration = None
    total_improvement = None
    metric_values: list[float] = [
        float(cast(int | float, item.get("metric_value")))
        for item in progress
        if isinstance(item.get("metric_value"), (int, float))
    ]
    if metric_values:
        best_metric = min(float(value) for value in metric_values)
        first_metric = float(metric_values[0])
        total_improvement = first_metric - best_metric
        for item in progress:
            if item.get("metric_value") == best_metric:
                iteration = item.get("iteration")
                if isinstance(iteration, int):
                    best_iteration = iteration
                    break
    return AnalyticsSnapshot(
        loop_count=1,
        total_runs=len(progress),
        keep_count=status_counts.get("keep", 0),
        discard_count=status_counts.get("discard", 0),
        crash_count=status_counts.get("crash", 0),
        source_counts=dict(source_counts),
        mutation_group_counts=dict(mutation_group_counts),
        model_family_wins=dict(model_family_wins),
        time_to_best_iteration=best_iteration,
        total_improvement=total_improvement,
    )


def compute_multi_loop_analytics(loop_summaries: list[dict[str, Any]]) -> AnalyticsSnapshot:
    """Compute aggregate stats across multiple historical loops."""
    source_counts: Counter[str] = Counter()
    model_family_wins: Counter[str] = Counter()
    total_runs = 0
    keep_count = 0
    discard_count = 0
    crash_count = 0

    for summary in loop_summaries:
        analytics = summary.get("analytics") or {}
        source_counts.update(analytics.get("source_counts") or {})
        model_family_wins.update(analytics.get("model_family_wins") or {})
        total_runs += int(analytics.get("total_runs") or 0)
        keep_count += int(analytics.get("keep_count") or 0)
        discard_count += int(analytics.get("discard_count") or 0)
        crash_count += int(analytics.get("crash_count") or 0)

    return AnalyticsSnapshot(
        loop_count=len(loop_summaries),
        total_runs=total_runs,
        keep_count=keep_count,
        discard_count=discard_count,
        crash_count=crash_count,
        source_counts=dict(source_counts),
        model_family_wins=dict(model_family_wins),
    )
