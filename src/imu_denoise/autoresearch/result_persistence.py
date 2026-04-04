"""Result-persistence helpers for autoresearch loop iterations."""

from __future__ import annotations

from typing import Any

from . import artifacts as artifact_helpers


def persist_iteration_result(
    *,
    results_file: Any,
    result: Any,
    observability: Any,
    loop_run_id: str,
    results: list[Any],
    loop_controller: Any,
    loop_state: dict[str, Any],
    total_scheduled_runs: int,
    requested_batch_size: int | None,
    best_metric: float | None,
    best_run_id: str | None,
) -> None:
    artifact_helpers.append_result(results_file, result)
    observability.register_artifact(
        run_id=loop_run_id,
        path=results_file,
        artifact_type="autoresearch_results",
        label="results_tsv",
        source="runtime",
    )
    results.append(result)
    loop_controller.heartbeat(
        loop_run_id=loop_run_id,
        current_iteration=len(results),
        max_iterations=total_scheduled_runs,
        batch_size=requested_batch_size,
        pause_after_iteration=loop_state.get("pause_after_iteration"),
        pause_requested=False,
        stop_requested=bool(loop_state.get("stop_requested")),
        terminate_requested=bool(loop_state.get("terminate_requested")),
        best_metric=best_metric,
        best_run_id=best_run_id,
        active_child_run_id=None,
        status="running",
    )
