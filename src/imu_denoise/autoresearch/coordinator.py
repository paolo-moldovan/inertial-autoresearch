"""Top-level loop coordination for the IMU autoresearch runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from autoresearch_core.engine import (
    LoopControlResolution,
    LoopProgressState,
    initialize_progress_state,
    run_loop_schedule,
)
from imu_denoise.training import TrainingInterrupted

from . import iteration as iteration_helpers
from . import session as session_helpers


def run_loop(
    *,
    context: session_helpers.LoopRuntimeContext,
    pause_enabled: bool,
    metric_from_summary_fn: Any,
    is_better_fn: Any,
    wait_while_paused_fn: Any,
    finish_loop_with_status_fn: Any,
    import_hermes_state_fn: Any,
    outcome_helpers: Any,
) -> list[Any]:
    """Run the configured autoresearch loop using the prepared session context."""
    from imu_denoise.observability.control import LOOP_PAUSED

    state = initialize_progress_state(
        baseline_metric=context.baseline_reference.metric_value,
        baseline_run_id=context.baseline_reference.run_id,
    )
    session_helpers.start_loop_session(
        context=context,
        current_iteration=len(state.results),
        best_metric=state.best_metric,
        best_run_id=state.best_run_id,
        pause_enabled=pause_enabled,
    )

    def _wait_while_paused(progress_state: LoopProgressState) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            wait_while_paused_fn(
                loop_controller=context.loop_controller,
                loop_run_id=context.loop_run_id,
                total_iterations=context.total_scheduled_runs,
                batch_size=context.requested_batch_size,
                current_iteration=len(progress_state.results),
                best_metric=progress_state.best_metric,
                best_run_id=progress_state.best_run_id,
            ),
        )

    def _apply_pause(
        control: LoopControlResolution,
        loop_state: Mapping[str, Any],
        progress_state: LoopProgressState,
    ) -> Mapping[str, Any]:
        pause_reason = control.pause_reason or "manual"
        pause_after_iteration = loop_state.get("pause_after_iteration")
        context.observability.append_event(
            run_id=context.loop_run_id,
            event_type=LOOP_PAUSED,
            level="INFO",
            title="loop paused",
            payload={"reason": pause_reason, "current_iteration": len(progress_state.results)},
            source="runtime",
        )
        context.loop_controller.heartbeat(
            loop_run_id=context.loop_run_id,
            current_iteration=len(progress_state.results),
            max_iterations=context.total_scheduled_runs,
            batch_size=context.requested_batch_size,
            pause_after_iteration=pause_after_iteration,
            pause_requested=False,
            stop_requested=bool(loop_state.get("stop_requested")),
            terminate_requested=bool(loop_state.get("terminate_requested")),
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
            active_child_run_id=None,
            status="paused",
        )
        return _wait_while_paused(progress_state)

    def _handle_terminal(
        control: LoopControlResolution,
        progress_state: LoopProgressState,
    ) -> None:
        finish_loop_with_status_fn(
            observability=context.observability,
            loop_controller=context.loop_controller,
            loop_run_id=context.loop_run_id,
            current_iteration=len(progress_state.results),
            max_iterations=context.total_scheduled_runs,
            batch_size=context.requested_batch_size,
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
            status=control.terminal_status,
            message=control.terminal_message or control.terminal_status,
        )

    def _select_iteration(
        iteration: int,
        fallback_proposal: Any,
        progress_state: LoopProgressState,
    ) -> Any:
        return iteration_helpers.resolve_iteration_selection(
            loop_controller=context.loop_controller,
            loop_run_id=context.loop_run_id,
            iteration=iteration,
            base_config=context.base_config,
            base_overrides=context.base_overrides,
            mutation_catalog=context.mutation_catalog,
            adapter=context.adapter,
            queries=context.queries,
            rng=context.rng,
            results=progress_state.results,
            fallback_proposal=fallback_proposal,
            hermes_used_descriptions=progress_state.provider_used_descriptions,
            best_run_id=progress_state.best_run_id,
        )

    def _prepare_iteration(
        iteration: int,
        selection: Any,
        loop_state: Mapping[str, Any],
        progress_state: LoopProgressState,
    ) -> Any:
        return iteration_helpers.prepare_iteration_run(
            observability=context.observability,
            loop_controller=context.loop_controller,
            queries=context.queries,
            adapter=context.adapter,
            import_hermes_state_fn=import_hermes_state_fn,
            base_config=context.base_config,
            base_overrides=context.base_overrides,
            loop_run_id=context.loop_run_id,
            loop_state=dict(loop_state),
            total_scheduled_runs=context.total_scheduled_runs,
            requested_batch_size=context.requested_batch_size,
            iteration=iteration,
            results=progress_state.results,
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
            selection=selection,
        )

    def _execute_iteration(
        iteration: int,
        selection: Any,
        prepared_run: Any,
        progress_state: LoopProgressState,
    ) -> tuple[Any, float | None, str | None]:
        proposal = selection.proposal
        proposal_source = selection.proposal_source
        hermes_trace = selection.hermes_trace
        candidate_pool = selection.candidate_pool

        config, summary, experiment_run_id = context.adapter.execute_training_run(
            config=prepared_run.resolved_config,
            overrides=prepared_run.run_overrides,
            metric_key=context.base_config.autoresearch.metric_key,
            parent_run_id=context.loop_run_id,
            iteration=iteration,
            run_id=prepared_run.experiment_run_id,
        )
        outcome = outcome_helpers.record_success_outcome(
            iteration=iteration,
            config=config,
            summary=summary,
            proposal=proposal,
            proposal_source=proposal_source,
            run_overrides=prepared_run.run_overrides,
            metric_key=context.base_config.autoresearch.metric_key,
            metric_direction=context.base_config.autoresearch.metric_direction,
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
            experiment_run_id=experiment_run_id,
            loop_run_id=context.loop_run_id,
            selected_incumbent_run_id=prepared_run.selected_incumbent_run_id,
            selected_incumbent_metric=prepared_run.selected_incumbent_metric,
            baseline_included=(
                iteration == 0 and context.baseline_reference.include_baseline_run
            ),
            candidate_pool=candidate_pool,
            queue_row=selection.queue_row,
            llm_call_id=prepared_run.llm_call_id,
            hermes_reason=None if hermes_trace is None else hermes_trace.reason,
            resolved_config=prepared_run.resolved_config,
            change_set=prepared_run.change_set,
            observability=context.observability,
            loop_controller=context.loop_controller,
            experiment_run_paths=prepared_run.experiment_run_paths,
            metric_from_summary_fn=metric_from_summary_fn,
            is_better_fn=is_better_fn,
        )
        return outcome.result, outcome.best_metric, outcome.best_run_id

    def _handle_interrupted(
        exc: BaseException,
        iteration: int,
        selection: Any,
        prepared_run: Any,
        loop_state: Mapping[str, Any],
        progress_state: LoopProgressState,
    ) -> None:
        assert isinstance(exc, TrainingInterrupted)
        outcome = outcome_helpers.handle_interrupted_outcome(
            exc=exc,
            iteration=iteration,
            proposal=selection.proposal,
            proposal_source=selection.proposal_source,
            run_overrides=prepared_run.run_overrides,
            metric_key=context.base_config.autoresearch.metric_key,
            candidate_pool=selection.candidate_pool,
            llm_call_id=prepared_run.llm_call_id,
            loop_run_id=context.loop_run_id,
            queue_row=selection.queue_row,
            observability=context.observability,
            loop_controller=context.loop_controller,
            experiment_run_paths=prepared_run.experiment_run_paths,
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
        )
        outcome_helpers.persist_iteration_result(
            results_file=context.results_file,
            result=outcome.result,
            observability=context.observability,
            loop_run_id=context.loop_run_id,
            results=progress_state.results,
            loop_controller=context.loop_controller,
            loop_state=dict(loop_state),
            total_scheduled_runs=context.total_scheduled_runs,
            requested_batch_size=context.requested_batch_size,
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
        )
        finish_loop_with_status_fn(
            observability=context.observability,
            loop_controller=context.loop_controller,
            loop_run_id=context.loop_run_id,
            current_iteration=len(progress_state.results),
            max_iterations=context.total_scheduled_runs,
            batch_size=context.requested_batch_size,
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
            status=outcome.terminal_status or exc.status,
            message=outcome.terminal_message or str(exc),
        )

    def _handle_crash(
        exc: Exception,
        iteration: int,
        selection: Any,
        prepared_run: Any,
        loop_state: Mapping[str, Any],
        progress_state: LoopProgressState,
    ) -> Any:
        del loop_state
        outcome = outcome_helpers.handle_crash_outcome(
            exc=exc,
            iteration=iteration,
            proposal=selection.proposal,
            proposal_source=selection.proposal_source,
            run_overrides=prepared_run.run_overrides,
            metric_key=context.base_config.autoresearch.metric_key,
            metric_direction=context.base_config.autoresearch.metric_direction,
            candidate_pool=selection.candidate_pool,
            llm_call_id=prepared_run.llm_call_id,
            loop_run_id=context.loop_run_id,
            experiment_run_id=prepared_run.experiment_run_id,
            queue_row=selection.queue_row,
            observability=context.observability,
            loop_controller=context.loop_controller,
            experiment_run_paths=prepared_run.experiment_run_paths,
            resolved_config=prepared_run.resolved_config,
            change_set=prepared_run.change_set,
            selected_incumbent_metric=prepared_run.selected_incumbent_metric,
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
        )
        return outcome.result

    def _persist_iteration(
        result: Any,
        loop_state: Mapping[str, Any],
        progress_state: LoopProgressState,
    ) -> None:
        outcome_helpers.persist_iteration_result(
            results_file=context.results_file,
            result=result,
            observability=context.observability,
            loop_run_id=context.loop_run_id,
            results=progress_state.results,
            loop_controller=context.loop_controller,
            loop_state=dict(loop_state),
            total_scheduled_runs=context.total_scheduled_runs,
            requested_batch_size=context.requested_batch_size,
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
        )
        if hasattr(result, "description"):
            progress_state.provider_used_descriptions.add(result.description)

    def _finish_completed(progress_state: LoopProgressState) -> None:
        finish_loop_with_status_fn(
            observability=context.observability,
            loop_controller=context.loop_controller,
            loop_run_id=context.loop_run_id,
            current_iteration=len(progress_state.results),
            max_iterations=context.total_scheduled_runs,
            batch_size=context.requested_batch_size,
            best_metric=progress_state.best_metric,
            best_run_id=progress_state.best_run_id,
            status="completed",
            message=f"completed {len(progress_state.results)} iterations",
        )

    return run_loop_schedule(
        schedule=context.schedule,
        progress_state=state,
        fetch_loop_state=lambda: context.loop_controller.get_loop_state(context.loop_run_id),
        wait_while_paused=_wait_while_paused,
        apply_pause=_apply_pause,
        handle_terminal=_handle_terminal,
        select_iteration=_select_iteration,
        prepare_iteration=_prepare_iteration,
        execute_iteration=_execute_iteration,
        persist_iteration=_persist_iteration,
        handle_interrupted=_handle_interrupted,
        handle_crash=_handle_crash,
        finish_completed=_finish_completed,
        interrupted_exception_type=TrainingInterrupted,
    )
