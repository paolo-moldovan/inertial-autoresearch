"""Run-preparation helpers for one autoresearch iteration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from imu_denoise.autoresearch.selection_state import IterationSelection
    from imu_denoise.config import ExperimentConfig


@dataclass(frozen=True)
class PreparedIterationRun:
    """Prepared execution state emitted before the actual training run starts."""

    experiment_run_id: str
    run_overrides: list[str]
    resolved_config: Any
    selected_incumbent_run_id: str | None
    selected_incumbent_metric: float | None
    incumbent_config: dict[str, Any] | None
    selection_event: dict[str, Any]
    change_set: dict[str, Any]
    experiment_run_paths: Any
    llm_call_id: str | None


def build_run_overrides(
    *,
    iteration: int,
    proposal: Any,
    base_overrides: list[str],
    base_config: ExperimentConfig,
) -> list[str]:
    overrides = list(base_overrides)
    overrides.extend(proposal.overrides)
    overrides.append(f"name=autoresearch_{iteration:03d}")
    if base_config.autoresearch.time_budget_sec > 0:
        overrides.append(f"training.time_budget_sec={base_config.autoresearch.time_budget_sec}")
    return overrides


def prepare_iteration_run(
    *,
    observability: Any,
    loop_controller: Any,
    queries: Any,
    adapter: Any,
    import_hermes_state_fn: Any,
    base_config: ExperimentConfig,
    base_overrides: list[str],
    loop_run_id: str,
    loop_state: dict[str, Any],
    total_scheduled_runs: int,
    requested_batch_size: int | None,
    iteration: int,
    results: list[Any],
    best_metric: float | None,
    best_run_id: str | None,
    selection: IterationSelection,
) -> PreparedIterationRun:
    from imu_denoise.autoresearch.artifacts import safe_update_run_manifest
    from imu_denoise.observability.lineage import data_regime_fingerprint
    from imu_denoise.utils.paths import build_run_paths

    observability.update_status(
        run_id=loop_run_id,
        phase="autoresearch_loop",
        epoch=len(results),
        best_metric=best_metric,
        last_metric=results[-1].metric_value if results else None,
        message=f"iteration {len(results)}",
        source="runtime",
    )
    llm_call_id: str | None = None
    if selection.hermes_trace is not None:
        llm_call_id = observability.record_llm_call(
            run_id=loop_run_id,
            provider=base_config.autoresearch.hermes.provider,
            model=base_config.autoresearch.hermes.model,
            base_url=base_config.autoresearch.hermes.base_url,
            status=selection.hermes_trace.status,
            latency_ms=selection.hermes_trace.latency_ms,
            prompt=selection.hermes_trace.prompt,
            response=selection.hermes_trace.stdout,
            stdout_text=selection.hermes_trace.stdout,
            stderr_text=selection.hermes_trace.stderr,
            parsed_payload=selection.hermes_trace.parsed_payload,
            command=selection.hermes_trace.command,
            session_id=selection.hermes_trace.session_id,
            reason=selection.hermes_trace.reason,
            source="runtime",
        )
        if base_config.observability.import_hermes_state:
            import_hermes_state_fn(
                writer=observability,
                hermes_home=Path(base_config.autoresearch.hermes.home_dir),
            )
    run_overrides = build_run_overrides(
        iteration=iteration,
        proposal=selection.proposal,
        base_overrides=base_overrides,
        base_config=base_config,
    )
    observability.append_event(
        run_id=loop_run_id,
        event_type="candidate_generation",
        level="INFO",
        title=f"iteration {iteration} proposal selected",
        payload={
            "proposal_source": selection.proposal_source,
            "description": selection.proposal.description,
            "overrides": run_overrides,
            "candidate_count": (
                len(selection.candidate_pool) if selection.candidate_pool is not None else 0
            ),
            "policy_mode": (
                None if selection.policy_decision is None else selection.policy_decision.mode
            ),
            "hermes_reason": (
                None if selection.hermes_trace is None else selection.hermes_trace.reason
            ),
            "hermes_status": (
                None if selection.hermes_trace is None else selection.hermes_trace.status
            ),
            "blocked_candidates": selection.blocked_candidates,
        },
        source="runtime",
    )
    experiment_run_id = f"training-{iteration:03d}-{uuid4().hex}"
    selected_incumbent_run_id = best_run_id
    selected_incumbent_metric = best_metric
    incumbent_config = selection.incumbent_config_for_policy
    resolved_config = adapter.resolve_iteration_config(
        base_config=base_config,
        base_overrides=base_overrides,
        proposal_overrides=list(selection.proposal.overrides),
        incumbent_config=incumbent_config,
        extra_overrides=[
            f"name=autoresearch_{iteration:03d}",
            *(
                [f"training.time_budget_sec={base_config.autoresearch.time_budget_sec}"]
                if base_config.autoresearch.time_budget_sec > 0
                else []
            ),
        ],
    )
    reference_kind = "incumbent" if incumbent_config is not None else "base"
    selection_rationale = (
        f"queued proposal #{selection.queue_row['id']}"
        if selection.queue_row is not None
        else (
            (
                f"{selection.policy_decision.mode} policy selected {selection.proposal.description}"
                + (
                    " over Hermes preference"
                    if (
                        selection.policy_decision is not None
                        and selection.preferred_candidate_index is not None
                        and selection.candidate_pool is not None
                        and 0
                        <= selection.preferred_candidate_index
                        < len(selection.candidate_pool)
                        and selection.candidate_pool[
                            selection.preferred_candidate_index
                        ].description
                        != selection.proposal.description
                    )
                    else ""
                )
            )
            if selection.policy_decision is not None
            else (
                selection.hermes_trace.reason
                if selection.hermes_trace is not None and selection.hermes_trace.reason
                else f"{selection.proposal_source} proposal selected"
            )
        )
    )
    candidate_count = (
        len(selection.candidate_pool)
        if selection.candidate_pool is not None
        else (
            1
            if selection.queue_row is not None
            or selection.proposal_source.startswith("static")
            else None
        )
    )
    selection_event = observability.record_selection_event(
        run_id=experiment_run_id,
        loop_run_id=loop_run_id,
        iteration=iteration,
        proposal_source=selection.proposal_source,
        description=selection.proposal.description,
        incumbent_run_id=selected_incumbent_run_id,
        candidate_count=candidate_count,
        rationale=selection_rationale,
        policy_state={
            "baseline_mode": base_config.autoresearch.baseline.mode,
            "best_metric": best_metric,
            "best_run_id": best_run_id,
            "strategy": asdict(base_config.autoresearch.strategy),
            "policy_mode": (
                None if selection.policy_decision is None else selection.policy_decision.mode
            ),
            "policy_stagnating": (
                None if selection.policy_decision is None else selection.policy_decision.stagnating
            ),
            "policy_explore_probability": (
                None
                if selection.policy_decision is None
                else selection.policy_decision.explore_probability
            ),
            "selected_candidate_index": (
                None
                if selection.policy_decision is None
                else selection.policy_decision.selected_index
            ),
            "preferred_candidate_index": selection.preferred_candidate_index,
            "preferred_candidate_description": (
                None
                if selection.preferred_candidate_index is None or selection.candidate_pool is None
                else selection.candidate_pool[selection.preferred_candidate_index].description
            ),
            "blocked_candidates": selection.blocked_candidates,
            "hermes_status": (
                None if selection.hermes_trace is None else selection.hermes_trace.status
            ),
            "hermes_reason": (
                None if selection.hermes_trace is None else selection.hermes_trace.reason
            ),
            "hermes_session_id": (
                None if selection.hermes_trace is None else selection.hermes_trace.session_id
            ),
            "policy_candidates": (
                None
                if selection.policy_decision is None or selection.policy_candidate_payloads is None
                else [
                    {
                        **selection.policy_candidate_payloads[score.index],
                        "total_score": score.total_score,
                        "exploration_score": score.exploration_score,
                        "novelty_score": score.novelty_score,
                        "confidence": score.confidence,
                        "avg_metric_delta": score.avg_metric_delta,
                        "total_tries": score.total_tries,
                        "keep_count": score.keep_count,
                        "discard_count": score.discard_count,
                        "crash_count": score.crash_count,
                        "reasons": score.reasons,
                    }
                    for score in selection.policy_decision.scored_candidates[:8]
                ]
            ),
            "candidate_descriptions": (
                [candidate.description for candidate in selection.candidate_pool]
                if selection.candidate_pool is not None
                else None
            ),
            "queued_proposal_id": (
                None if selection.queue_row is None else int(selection.queue_row["id"])
            ),
        },
        source="runtime",
    )
    change_set = observability.record_change_set(
        run_id=experiment_run_id,
        loop_run_id=loop_run_id,
        parent_run_id=selected_incumbent_run_id,
        incumbent_run_id=selected_incumbent_run_id,
        reference_kind=reference_kind,
        proposal_source=selection.proposal_source,
        description=selection.proposal.description,
        overrides=list(selection.proposal.overrides),
        current_config=resolved_config,
        reference_config=incumbent_config if incumbent_config is not None else base_config,
        source="runtime",
    )
    experiment_run_paths = build_run_paths(
        base_config.output_dir,
        run_name=resolved_config.name,
        run_id=experiment_run_id,
    )
    safe_update_run_manifest(
        experiment_run_paths,
        {
            "regime_fingerprint": data_regime_fingerprint(resolved_config),
            "resolved_config": observability.config_payload(resolved_config),
            "selection_event": selection_event,
            "change_set": change_set,
        },
    )
    loop_controller.heartbeat(
        loop_run_id=loop_run_id,
        current_iteration=len(results),
        max_iterations=total_scheduled_runs,
        batch_size=requested_batch_size,
        pause_after_iteration=loop_state.get("pause_after_iteration"),
        pause_requested=bool(loop_state.get("pause_requested")),
        stop_requested=bool(loop_state.get("stop_requested")),
        terminate_requested=bool(loop_state.get("terminate_requested")),
        best_metric=best_metric,
        best_run_id=best_run_id,
        active_child_run_id=experiment_run_id,
        status="running",
    )
    return PreparedIterationRun(
        experiment_run_id=experiment_run_id,
        run_overrides=run_overrides,
        resolved_config=resolved_config,
        selected_incumbent_run_id=selected_incumbent_run_id,
        selected_incumbent_metric=selected_incumbent_metric,
        incumbent_config=incumbent_config,
        selection_event=selection_event,
        change_set=change_set,
        experiment_run_paths=experiment_run_paths,
        llm_call_id=llm_call_id,
    )
