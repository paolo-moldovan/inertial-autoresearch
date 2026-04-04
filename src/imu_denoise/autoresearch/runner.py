"""Local config-first autoresearch loop for IMU denoising."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

from autoresearch_core import RunResult, recent_policy_results, result_snapshot

from . import artifacts as artifact_helpers
from . import coordinator as coordinator_helpers
from . import execution as execution_helpers
from . import iteration as iteration_helpers
from . import lifecycle as lifecycle_helpers
from . import outcomes as outcome_helpers
from . import selection as selection_helpers
from . import session as session_helpers

if TYPE_CHECKING:
    from autoresearch_core.providers.hermes import HermesQueryTrace
    from imu_denoise.autoresearch.mutations import MutationProposal
    from imu_denoise.config import ExperimentConfig
    from imu_denoise.observability import LoopController, MissionControlQueries

ROOT = Path(__file__).resolve().parents[3]


def _metric_from_summary(summary: Any, metric_key: str) -> float:
    return execution_helpers.metric_from_summary(summary, metric_key)


def _resolve_iteration_config(
    *,
    base_config: ExperimentConfig,
    base_overrides: list[str],
    proposal_overrides: list[str],
    incumbent_config: dict[str, Any] | None = None,
    extra_overrides: list[str] | None = None,
) -> ExperimentConfig:
    return execution_helpers.resolve_iteration_config(
        base_config=base_config,
        base_overrides=base_overrides,
        proposal_overrides=proposal_overrides,
        incumbent_config=incumbent_config,
        extra_overrides=extra_overrides,
    )


def _run_single_experiment(
    *,
    config: ExperimentConfig,
    overrides: list[str],
    metric_key: str,
    parent_run_id: str | None = None,
    iteration: int | None = None,
    run_id: str | None = None,
) -> tuple[Any, Any, str]:
    return execution_helpers.run_single_experiment(
        config=config,
        overrides=overrides,
        metric_key=metric_key,
        parent_run_id=parent_run_id,
        iteration=iteration,
        run_id=run_id,
    )


LoopResult = RunResult


BaselineReference = lifecycle_helpers.BaselineReference


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the local autoresearch loop."""
    parser = argparse.ArgumentParser(description="Run local config-first autoresearch iterations.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional config path(s) to merge after the shared defaults.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Base override applied to every run in dotted.key=value form.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override autoresearch.max_iterations.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Pause after this many completed runs when --pause is enabled.",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Enable batch pause/review mode.",
    )
    return parser


def _ensure_results_file(path: Path) -> None:
    artifact_helpers.ensure_results_file(path)


def _append_result(path: Path, result: LoopResult) -> None:
    artifact_helpers.append_result(path, result)


def _is_better(candidate: float, incumbent: float | None, direction: str) -> bool:
    if incumbent is None:
        return True
    if direction == "maximize":
        return candidate > incumbent
    return candidate < incumbent


def _build_run_overrides(
    *,
    iteration: int,
    proposal: MutationProposal,
    base_overrides: list[str],
    base_config: ExperimentConfig,
) -> list[str]:
    return iteration_helpers.build_run_overrides(
        iteration=iteration,
        proposal=proposal,
        base_overrides=base_overrides,
        base_config=base_config,
    )


def _result_snapshot(result: LoopResult) -> dict[str, object]:
    return result_snapshot(result)


def _recent_policy_results(results: list[LoopResult]) -> list[dict[str, Any]]:
    return recent_policy_results(results)


def _select_mutation_proposal(
    *,
    iteration: int,
    base_config: ExperimentConfig,
    base_overrides: list[str],
    mutation_catalog: list[MutationProposal] | None = None,
    resolve_iteration_config_fn: Callable[..., ExperimentConfig] | None = None,
    rng: Random,
    results: list[LoopResult],
    fallback_proposal: MutationProposal,
    hermes_used_descriptions: set[str],
    incumbent_summary: dict[str, object] | None,
    incumbent_config: dict[str, Any] | None,
    mutation_lessons: list[dict[str, object]] | None,
) -> tuple[
    list[MutationProposal],
    dict[str, list[str]],
    int | None,
    str,
    HermesQueryTrace | None,
]:
    config_resolver = resolve_iteration_config_fn or _resolve_iteration_config
    return selection_helpers.select_mutation_proposal(
        root=ROOT,
        iteration=iteration,
        base_config=base_config,
        base_overrides=base_overrides,
        mutation_catalog=mutation_catalog,
        resolve_iteration_config_fn=config_resolver,
        rng=rng,
        results=results,
        fallback_proposal=fallback_proposal,
        hermes_used_descriptions=hermes_used_descriptions,
        incumbent_summary=incumbent_summary,
        incumbent_config=incumbent_config,
        mutation_lessons=mutation_lessons,
    )


def _best_metric_from_results(results: list[LoopResult], direction: str) -> float | None:
    return lifecycle_helpers.best_metric_from_results(results, direction)


def _resolve_baseline_reference(
    base_config: ExperimentConfig,
    queries: MissionControlQueries,
) -> BaselineReference:
    return lifecycle_helpers.resolve_baseline_reference(
        base_config=base_config,
        queries=queries,
    )


def _wait_while_paused(
    *,
    loop_controller: LoopController,
    loop_run_id: str,
    total_iterations: int,
    batch_size: int | None,
    current_iteration: int,
    best_metric: float | None,
    best_run_id: str | None,
) -> dict[str, Any]:
    return lifecycle_helpers.wait_while_paused(
        loop_controller=loop_controller,
        loop_run_id=loop_run_id,
        total_iterations=total_iterations,
        batch_size=batch_size,
        current_iteration=current_iteration,
        best_metric=best_metric,
        best_run_id=best_run_id,
    )


def _finish_loop_with_status(
    *,
    observability: Any,
    loop_controller: LoopController,
    loop_run_id: str,
    current_iteration: int,
    max_iterations: int,
    batch_size: int | None,
    best_metric: float | None,
    best_run_id: str | None,
    status: str,
    message: str,
) -> None:
    lifecycle_helpers.finish_loop_with_status(
        observability=observability,
        loop_controller=loop_controller,
        loop_run_id=loop_run_id,
        current_iteration=current_iteration,
        max_iterations=max_iterations,
        batch_size=batch_size,
        best_metric=best_metric,
        best_run_id=best_run_id,
        status=status,
        message=message,
    )


def _resolve_results_file(
    *,
    base_config: ExperimentConfig,
    loop_run_id: str,
    loop_name: str,
) -> Path:
    return artifact_helpers.resolve_results_file(
        base_config=base_config,
        loop_run_id=loop_run_id,
        loop_name=loop_name,
    )


def _safe_update_run_manifest(run_paths: Any, payload: dict[str, Any]) -> None:
    artifact_helpers.safe_update_run_manifest(run_paths, payload)


def run_autoresearch(
    *,
    config_paths: list[str],
    base_overrides: list[str] | None = None,
    max_iterations: int | None = None,
    batch_size: int | None = None,
    pause_enabled: bool = False,
) -> list[LoopResult]:
    """Run the baseline plus config mutations and record results to TSV."""
    base_overrides = list(base_overrides or [])
    context = session_helpers.initialize_runtime_context(
        config_paths=config_paths,
        base_overrides=base_overrides,
        max_iterations=max_iterations,
        batch_size=batch_size,
        pause_enabled=pause_enabled,
    )
    from imu_denoise.observability import import_hermes_state

    return coordinator_helpers.run_loop(
        context=context,
        pause_enabled=pause_enabled,
        metric_from_summary_fn=_metric_from_summary,
        is_better_fn=_is_better,
        wait_while_paused_fn=_wait_while_paused,
        finish_loop_with_status_fn=_finish_loop_with_status,
        import_hermes_state_fn=import_hermes_state,
        outcome_helpers=outcome_helpers,
    )


def main() -> int:
    """CLI entrypoint for the local autoresearch loop."""
    args = build_parser().parse_args()
    config_paths = list(args.config) or ["configs/training/quick.yaml"]
    results = run_autoresearch(
        config_paths=config_paths,
        base_overrides=args.overrides,
        max_iterations=args.max_iterations,
        batch_size=args.batch or None,
        pause_enabled=args.pause,
    )
    print(f"Completed {len(results)} autoresearch runs.")
    if results:
        best_completed = [result for result in results if result.metric_value is not None]
        if best_completed:
            best = min(best_completed, key=lambda result: result.metric_value or float("inf"))
            print(
                f"Best run: {best.run_name} {best.metric_key}={best.metric_value:.6f} "
                f"status={best.status}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
