"""Result-recording helpers for executed autoresearch iterations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoresearch_core import build_run_result, candidate_payloads
from imu_denoise.training import TrainingInterrupted

from . import artifacts as artifact_helpers


@dataclass(frozen=True)
class IterationOutcome:
    """Resolved outcome state for one executed loop iteration."""

    result: Any
    best_metric: float | None
    best_run_id: str | None
    terminal_status: str | None = None
    terminal_message: str | None = None


def record_success_outcome(
    *,
    iteration: int,
    config: Any,
    summary: Any,
    proposal: Any,
    proposal_source: str,
    run_overrides: list[str],
    metric_key: str,
    metric_direction: str,
    best_metric: float | None,
    best_run_id: str | None,
    experiment_run_id: str,
    loop_run_id: str,
    selected_incumbent_run_id: str | None,
    selected_incumbent_metric: float | None,
    baseline_included: bool,
    candidate_pool: list[Any] | None,
    queue_row: dict[str, Any] | None,
    llm_call_id: str | None,
    hermes_reason: str | None,
    resolved_config: Any,
    change_set: dict[str, Any],
    observability: Any,
    loop_controller: Any,
    experiment_run_paths: Any,
    metric_from_summary_fn: Any,
    is_better_fn: Any,
) -> IterationOutcome:
    from imu_denoise.observability.lineage import data_regime_fingerprint

    metric_value = metric_from_summary_fn(summary, metric_key)
    next_best_metric = best_metric
    next_best_run_id = best_run_id
    if iteration == 0 and baseline_included:
        status = "baseline"
        next_best_metric = metric_value
        next_best_run_id = experiment_run_id
    elif is_better_fn(metric_value, best_metric, metric_direction):
        status = "keep"
        next_best_metric = metric_value
        next_best_run_id = experiment_run_id
    else:
        status = "discard"

    result = build_run_result(
        iteration=iteration,
        run_name=config.name,
        status=status,
        proposal_source=proposal_source,
        metric_key=metric_key,
        metric_value=metric_value,
        model_name=config.model.name,
        description=proposal.description,
        overrides=run_overrides,
        metrics_path=summary.artifacts.metrics_path,
    )
    observability.record_decision(
        run_id=experiment_run_id,
        iteration=iteration,
        proposal_source=proposal_source,
        description=proposal.description,
        status=status,
        metric_key=metric_key,
        metric_value=metric_value,
        overrides=run_overrides,
        candidates=candidate_payloads(candidate_pool),
        reason=hermes_reason,
        llm_call_id=llm_call_id,
        source="runtime",
    )
    observability.record_mutation_outcome(
        run_id=experiment_run_id,
        loop_run_id=loop_run_id,
        regime_fingerprint=data_regime_fingerprint(resolved_config),
        proposal_source=proposal_source,
        description=proposal.description,
        change_items=list(change_set["change_items"]),
        status=status,
        metric_key=metric_key,
        metric_value=metric_value,
        incumbent_metric=selected_incumbent_metric,
        direction=metric_direction,
        source="runtime",
    )
    if queue_row is not None:
        loop_controller.mark_queue_applied(
            proposal_id=int(queue_row["id"]),
            loop_run_id=loop_run_id,
            applied_run_id=experiment_run_id,
        )
    artifact_helpers.safe_update_run_manifest(
        experiment_run_paths,
        {
            "result": {
                "status": status,
                "metric_key": metric_key,
                "metric_value": metric_value,
                "compared_against_run_id": selected_incumbent_run_id,
                "new_incumbent_run_id": next_best_run_id,
            }
        },
    )
    return IterationOutcome(
        result=result,
        best_metric=next_best_metric,
        best_run_id=next_best_run_id,
    )


def handle_interrupted_outcome(
    *,
    exc: TrainingInterrupted,
    iteration: int,
    proposal: Any,
    proposal_source: str,
    run_overrides: list[str],
    metric_key: str,
    candidate_pool: list[Any] | None,
    llm_call_id: str | None,
    loop_run_id: str,
    queue_row: dict[str, Any] | None,
    observability: Any,
    loop_controller: Any,
    experiment_run_paths: Any,
    best_metric: float | None,
    best_run_id: str | None,
) -> IterationOutcome:
    result = build_run_result(
        iteration=iteration,
        run_name=f"autoresearch_{iteration:03d}",
        status=exc.status,
        proposal_source=proposal_source,
        metric_key=metric_key,
        metric_value=None,
        model_name="unknown",
        description=f"{proposal.description}: {exc}",
        overrides=run_overrides,
        metrics_path=None,
    )
    observability.record_decision(
        run_id=loop_run_id,
        iteration=iteration,
        proposal_source=proposal_source,
        description=proposal.description,
        status=exc.status,
        metric_key=metric_key,
        metric_value=None,
        overrides=run_overrides,
        candidates=candidate_payloads(candidate_pool),
        reason=str(exc),
        llm_call_id=llm_call_id,
        source="runtime",
    )
    artifact_helpers.safe_update_run_manifest(
        experiment_run_paths,
        {
            "result": {
                "status": exc.status,
                "metric_key": metric_key,
                "metric_value": None,
                "message": str(exc),
            }
        },
    )
    if queue_row is not None:
        loop_controller.mark_queue_failed(
            proposal_id=int(queue_row["id"]),
            notes=str(exc),
        )
    return IterationOutcome(
        result=result,
        best_metric=best_metric,
        best_run_id=best_run_id,
        terminal_status=exc.status,
        terminal_message=str(exc),
    )


def handle_crash_outcome(
    *,
    exc: Exception,
    iteration: int,
    proposal: Any,
    proposal_source: str,
    run_overrides: list[str],
    metric_key: str,
    metric_direction: str,
    candidate_pool: list[Any] | None,
    llm_call_id: str | None,
    loop_run_id: str,
    experiment_run_id: str,
    queue_row: dict[str, Any] | None,
    observability: Any,
    loop_controller: Any,
    experiment_run_paths: Any,
    resolved_config: Any,
    change_set: dict[str, Any],
    selected_incumbent_metric: float | None,
    best_metric: float | None,
    best_run_id: str | None,
) -> IterationOutcome:
    from imu_denoise.observability.lineage import data_regime_fingerprint

    result = build_run_result(
        iteration=iteration,
        run_name=f"autoresearch_{iteration:03d}",
        status="crash",
        proposal_source=proposal_source,
        metric_key=metric_key,
        metric_value=None,
        model_name="unknown",
        description=f"{proposal.description}: {exc}",
        overrides=run_overrides,
        metrics_path=None,
    )
    observability.record_decision(
        run_id=loop_run_id,
        iteration=iteration,
        proposal_source=proposal_source,
        description=proposal.description,
        status="crash",
        metric_key=metric_key,
        metric_value=None,
        overrides=run_overrides,
        candidates=candidate_payloads(candidate_pool),
        reason=str(exc),
        llm_call_id=llm_call_id,
        source="runtime",
    )
    observability.record_mutation_outcome(
        run_id=experiment_run_id,
        loop_run_id=loop_run_id,
        regime_fingerprint=data_regime_fingerprint(resolved_config),
        proposal_source=proposal_source,
        description=proposal.description,
        change_items=list(change_set["change_items"]),
        status="crash",
        metric_key=metric_key,
        metric_value=None,
        incumbent_metric=selected_incumbent_metric,
        direction=metric_direction,
        source="runtime",
    )
    artifact_helpers.safe_update_run_manifest(
        experiment_run_paths,
        {
            "result": {
                "status": "crash",
                "metric_key": metric_key,
                "metric_value": None,
                "message": str(exc),
            }
        },
    )
    if queue_row is not None:
        loop_controller.mark_queue_failed(
            proposal_id=int(queue_row["id"]),
            notes=str(exc),
        )
    return IterationOutcome(
        result=result,
        best_metric=best_metric,
        best_run_id=best_run_id,
    )
