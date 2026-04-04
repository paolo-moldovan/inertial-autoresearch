"""Reusable loop-engine helpers for provider selection and loop-state flow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any

from autoresearch_core.contracts import CandidateProposal, RunResult, SupportsRunResult


@dataclass(frozen=True)
class ProviderSelectionResult:
    """Resolved provider-facing candidate pool and chosen index."""

    candidates: list[CandidateProposal]
    blocked_candidates: dict[str, list[str]] = field(default_factory=dict)
    preferred_candidate_index: int = 0
    proposal_source: str = "static"
    provider_trace: Any | None = None


@dataclass
class LoopProgressState:
    """Mutable loop progress state tracked across iterations."""

    results: list[Any]
    best_metric: float | None
    best_run_id: str | None
    provider_used_descriptions: set[str]


@dataclass(frozen=True)
class LoopControlResolution:
    """Resolved generic control-plane action for the current loop heartbeat."""

    terminal_status: str | None = None
    terminal_message: str | None = None
    should_pause: bool = False
    pause_reason: str | None = None


def initialize_progress_state(
    *,
    baseline_metric: float | None,
    baseline_run_id: str | None,
) -> LoopProgressState:
    """Build the initial loop progress state from the resolved baseline policy."""
    return LoopProgressState(
        results=[],
        best_metric=baseline_metric,
        best_run_id=baseline_run_id,
        provider_used_descriptions=set(),
    )


def result_snapshot(result: SupportsRunResult) -> dict[str, object]:
    """Convert a run result into the compact provider-facing history payload."""
    return {
        "iteration": result.iteration,
        "run_name": result.run_name,
        "status": result.status,
        "proposal_source": result.proposal_source,
        "metric_key": result.metric_key,
        "metric_value": result.metric_value,
        "model_name": result.model_name,
        "description": result.description,
        "overrides": result.overrides,
    }


def recent_policy_results(results: Sequence[SupportsRunResult]) -> list[dict[str, Any]]:
    """Reduce loop history to the fields used by the local policy layer."""
    return [
        {
            "iteration": result.iteration,
            "status": result.status,
            "proposal_source": result.proposal_source,
            "metric_value": result.metric_value,
            "description": result.description,
        }
        for result in results
    ]


def candidate_payloads(
    candidates: Sequence[CandidateProposal] | None,
) -> list[dict[str, Any]] | None:
    """Convert candidate proposals into a persistence/UI-friendly payload."""
    if candidates is None:
        return None
    return [
        {
            "description": candidate.description,
            "overrides": list(candidate.overrides),
        }
        for candidate in candidates
    ]


def build_run_result(
    *,
    iteration: int,
    run_name: str,
    status: str,
    proposal_source: str,
    metric_key: str,
    metric_value: float | None,
    model_name: str,
    description: str,
    overrides: list[str],
    metrics_path: Any,
) -> RunResult:
    """Build the shared run-result contract used across loop orchestration."""
    return RunResult(
        iteration=iteration,
        run_name=run_name,
        status=status,
        proposal_source=proposal_source,
        metric_key=metric_key,
        metric_value=metric_value,
        model_name=model_name,
        description=description,
        overrides=list(overrides),
        metrics_path=metrics_path,
    )


def resolve_loop_control(
    *,
    loop_state: Mapping[str, Any],
    completed_iterations: int,
) -> LoopControlResolution:
    """Interpret a loop-state row into generic pause/stop/terminate actions."""
    if bool(loop_state.get("terminate_requested")):
        return LoopControlResolution(
            terminal_status="terminated",
            terminal_message=f"terminated after {completed_iterations} iterations",
        )
    if bool(loop_state.get("stop_requested")):
        return LoopControlResolution(
            terminal_status="stopped",
            terminal_message=f"stopped after {completed_iterations} iterations",
        )

    pause_reason = "manual"
    should_pause = bool(loop_state.get("pause_requested"))
    pause_after_iteration = loop_state.get("pause_after_iteration")
    if (
        isinstance(pause_after_iteration, int)
        and pause_after_iteration > 0
        and completed_iterations >= pause_after_iteration
    ):
        should_pause = True
        pause_reason = "batch"
    return LoopControlResolution(
        should_pause=should_pause,
        pause_reason=pause_reason if should_pause else None,
    )


def resolve_provider_selection(
    *,
    iteration: int,
    orchestrator: str,
    fallback_proposal: CandidateProposal,
    candidate_pool: list[CandidateProposal],
    blocked_candidates: dict[str, list[str]] | None = None,
    used_descriptions: set[str] | None = None,
    rng: Random | None = None,
    provider_ready: Callable[[], bool] | None = None,
    provider_select: (
        Callable[[list[CandidateProposal]], tuple[CandidateProposal, Any]] | None
    ) = None,
    provider_error: type[Exception] = Exception,
) -> ProviderSelectionResult:
    """Resolve the provider-visible pool and selected candidate for one iteration."""
    if iteration == 0:
        return ProviderSelectionResult(
            candidates=[fallback_proposal],
            blocked_candidates=dict(blocked_candidates or {}),
            preferred_candidate_index=0,
            proposal_source="static",
            provider_trace=None,
        )

    available_candidates = _available_candidates(
        candidate_pool=candidate_pool,
        used_descriptions=used_descriptions or set(),
        rng=rng,
    )
    fallback_index = _candidate_index(
        candidates=available_candidates,
        selected=fallback_proposal,
    )

    if orchestrator != "hermes":
        return ProviderSelectionResult(
            candidates=available_candidates,
            blocked_candidates=dict(blocked_candidates or {}),
            preferred_candidate_index=fallback_index,
            proposal_source="static",
            provider_trace=None,
        )

    if provider_ready is None or provider_select is None or not provider_ready():
        return ProviderSelectionResult(
            candidates=available_candidates,
            blocked_candidates=dict(blocked_candidates or {}),
            preferred_candidate_index=fallback_index,
            proposal_source="static-fallback",
            provider_trace=None,
        )

    try:
        proposal, trace = provider_select(available_candidates)
    except provider_error as exc:
        trace = getattr(exc, "trace", None)
        return ProviderSelectionResult(
            candidates=available_candidates,
            blocked_candidates=dict(blocked_candidates or {}),
            preferred_candidate_index=fallback_index,
            proposal_source="static-fallback",
            provider_trace=trace,
        )

    return ProviderSelectionResult(
        candidates=available_candidates,
        blocked_candidates=dict(blocked_candidates or {}),
        preferred_candidate_index=_candidate_index(
            candidates=available_candidates,
            selected=proposal,
        ),
        proposal_source="hermes",
        provider_trace=trace,
    )


def _available_candidates(
    *,
    candidate_pool: list[CandidateProposal],
    used_descriptions: set[str],
    rng: Random | None,
) -> list[CandidateProposal]:
    available = [
        proposal
        for proposal in candidate_pool
        if proposal.description not in used_descriptions
    ]
    if available:
        return available
    available = list(candidate_pool)
    if rng is not None:
        rng.shuffle(available)
    return available


def _candidate_index(
    *,
    candidates: list[CandidateProposal],
    selected: CandidateProposal,
) -> int:
    for index, candidate in enumerate(candidates):
        if (
            candidate.description == selected.description
            and candidate.overrides == selected.overrides
        ):
            return index
    return 0


def run_loop_schedule(
    *,
    schedule: Sequence[Any],
    progress_state: LoopProgressState,
    fetch_loop_state: Callable[[], Mapping[str, Any] | None],
    wait_while_paused: Callable[[LoopProgressState], Mapping[str, Any]],
    apply_pause: Callable[
        [LoopControlResolution, Mapping[str, Any], LoopProgressState],
        Mapping[str, Any],
    ],
    handle_terminal: Callable[[LoopControlResolution, LoopProgressState], None],
    select_iteration: Callable[[int, Any, LoopProgressState], Any],
    prepare_iteration: Callable[[int, Any, Mapping[str, Any], LoopProgressState], Any],
    execute_iteration: Callable[
        [int, Any, Any, LoopProgressState],
        tuple[Any, float | None, str | None],
    ],
    persist_iteration: Callable[[Any, Mapping[str, Any], LoopProgressState], None],
    handle_interrupted: Callable[
        [BaseException, int, Any, Any, Mapping[str, Any], LoopProgressState],
        None,
    ],
    handle_crash: Callable[
        [Exception, int, Any, Any, Mapping[str, Any], LoopProgressState],
        Any,
    ],
    finish_completed: Callable[[LoopProgressState], None],
    interrupted_exception_type: type[BaseException],
) -> list[Any]:
    """Run the generic loop schedule while delegating domain actions through callbacks."""
    for iteration, fallback_entry in enumerate(
        schedule[len(progress_state.results) :],
        start=len(progress_state.results),
    ):
        loop_state = fetch_loop_state()
        if loop_state is None:
            raise RuntimeError("Missing loop state for autoresearch loop.")
        if loop_state.get("status") == "paused":
            loop_state = wait_while_paused(progress_state)

        control = resolve_loop_control(
            loop_state=loop_state,
            completed_iterations=len(progress_state.results),
        )
        if control.terminal_status is not None:
            handle_terminal(control, progress_state)
            return progress_state.results
        if control.should_pause:
            loop_state = apply_pause(control, loop_state, progress_state)

        selection = select_iteration(iteration, fallback_entry, progress_state)
        prepared = prepare_iteration(iteration, selection, loop_state, progress_state)

        try:
            result, best_metric, best_run_id = execute_iteration(
                iteration,
                selection,
                prepared,
                progress_state,
            )
            progress_state.best_metric = best_metric
            progress_state.best_run_id = best_run_id
        except interrupted_exception_type as exc:
            handle_interrupted(
                exc,
                iteration,
                selection,
                prepared,
                loop_state,
                progress_state,
            )
            return progress_state.results
        except Exception as exc:
            result = handle_crash(
                exc,
                iteration,
                selection,
                prepared,
                loop_state,
                progress_state,
            )

        persist_iteration(result, loop_state, progress_state)

    finish_completed(progress_state)
    return progress_state.results
