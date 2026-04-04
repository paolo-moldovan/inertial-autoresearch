"""Mission Control summary and loop-event helpers for IMU observability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autoresearch_core.observability.analytics import (
    compute_loop_analytics,
    compute_multi_loop_analytics,
)
from autoresearch_core.observability.queries import _loads
from autoresearch_core.observability.read_models import build_mission_control_summary_payload
from imu_denoise.observability.control import (
    LOOP_PAUSED,
    LOOP_RESUMED,
    LOOP_STOP_REQUESTED,
    LOOP_STOPPED,
    LOOP_TERMINATE_REQUESTED,
    LOOP_TERMINATED,
    QUEUE_APPLIED,
    QUEUE_CLAIMED,
    QUEUE_ENQUEUED,
)

if TYPE_CHECKING:
    from imu_denoise.observability.queries import MissionControlQueries


def get_mission_control_summary(
    queries: MissionControlQueries,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    loop_state = queries.get_current_loop_state()
    comparison_regime_fingerprint: str | None = None
    comparison_metric_key = "val_rmse"
    if loop_state is not None:
        loop_run_id = str(loop_state["loop_run_id"])
        comparison_metric_key = queries.get_run_objective_metric(loop_run_id) or "val_rmse"
        if str(loop_state.get("status")) in {"running", "paused", "terminating", "completed"}:
            comparison_regime_fingerprint = queries.get_run_regime_fingerprint(loop_run_id)
    leaderboard = queries.list_leaderboard(
        limit=limit,
        metric_key=comparison_metric_key,
        regime_fingerprint=comparison_regime_fingerprint,
    )
    best_result = leaderboard[0] if leaderboard else None
    progress: list[dict[str, Any]] = []
    queued: list[dict[str, Any]] = []
    recent_decisions = queries.list_recent_decisions(limit=20)
    recent_llm_calls = queries.list_recent_llm_calls(limit=20)
    recent_loop_events = list_recent_loop_events(queries, limit=20)
    mutation_regime_fingerprint = comparison_regime_fingerprint
    if loop_state is not None:
        loop_run_id = str(loop_state["loop_run_id"])
        progress = queries.list_loop_iteration_metrics(loop_run_id)
        queued = queries.list_queued_proposals(loop_run_id)
        recent_decisions = queries.list_recent_decisions(limit=20, loop_run_id=loop_run_id)
        recent_llm_calls = queries.list_recent_llm_calls(limit=20, loop_run_id=loop_run_id)
        recent_loop_events = list_recent_loop_events(queries, limit=20, loop_run_id=loop_run_id)
    mutation_leaderboard = queries.list_mutation_leaderboard(
        limit=10,
        regime_fingerprint=mutation_regime_fingerprint,
    )
    recent_mutation_lessons = queries.list_recent_mutation_lessons(
        limit=10,
        regime_fingerprint=mutation_regime_fingerprint,
    )
    current_run = queries.get_current_run_summary(
        None if loop_state is None else str(loop_state["loop_run_id"])
    )
    hermes_runtime = queries.get_hermes_runtime_summary(
        loop_run_id=None if loop_state is None else str(loop_state["loop_run_id"])
    )
    analytics = compute_loop_analytics(
        progress=progress,
        decisions=recent_decisions,
        leaderboard=leaderboard,
    )
    loop_summaries = []
    for loop_run in queries.list_loop_runs(limit=10):
        loop_run_id = str(loop_run["id"])
        loop_progress = queries.list_loop_iteration_metrics(loop_run_id)
        loop_decisions = queries.list_recent_decisions(limit=100, loop_run_id=loop_run_id)
        loop_leaderboard = queries.list_leaderboard(
            limit=10,
            metric_key=queries.get_run_objective_metric(loop_run_id) or "val_rmse",
            regime_fingerprint=queries.get_run_regime_fingerprint(loop_run_id),
        )
        loop_summaries.append(
            {
                "loop_run_id": loop_run_id,
                "analytics": compute_loop_analytics(
                    progress=loop_progress,
                    decisions=loop_decisions,
                    leaderboard=loop_leaderboard,
                ).__dict__,
            }
        )
    multi_loop_analytics = compute_multi_loop_analytics(loop_summaries)
    return build_mission_control_summary_payload(
        loop_state=loop_state,
        current_run=current_run,
        best_result=best_result,
        leaderboard=leaderboard,
        progress=progress,
        queued_proposals=queued,
        recent_loop_events=recent_loop_events,
        recent_decisions=recent_decisions,
        recent_llm_calls=recent_llm_calls,
        regime_fingerprint=comparison_regime_fingerprint,
        comparison_metric_key=comparison_metric_key,
        mutation_leaderboard=mutation_leaderboard,
        recent_mutation_lessons=recent_mutation_lessons,
        hermes_runtime=hermes_runtime,
        analytics=analytics.__dict__,
        multi_loop_analytics=multi_loop_analytics.__dict__,
    )


def list_recent_loop_events(
    queries: MissionControlQueries,
    *,
    limit: int = 50,
    loop_run_id: str | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = [
        LOOP_PAUSED,
        LOOP_RESUMED,
        LOOP_STOP_REQUESTED,
        LOOP_TERMINATE_REQUESTED,
        LOOP_STOPPED,
        LOOP_TERMINATED,
        QUEUE_ENQUEUED,
        QUEUE_CLAIMED,
        QUEUE_APPLIED,
    ]
    where_suffix = ""
    if loop_run_id is not None:
        where_suffix = " AND e.run_id = ?"
        params.append(loop_run_id)
    rows = queries.store.fetch_all(
        f"""
        SELECT
            e.*,
            r.name AS run_name
        FROM events e
        LEFT JOIN runs r ON r.id = e.run_id
        WHERE e.event_type IN (?, ?, ?, ?, ?, ?, ?, ?, ?)
        {where_suffix}
        ORDER BY e.created_at DESC
        LIMIT ?
        """,
        (*params, limit),
    )
    for row in rows:
        row["payload"] = _loads(row.pop("payload_json"))
    return rows
