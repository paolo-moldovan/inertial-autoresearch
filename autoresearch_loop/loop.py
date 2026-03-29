"""Local config-first autoresearch loop for IMU denoising."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from autoresearch_loop.hermes import HermesQueryTrace
    from autoresearch_loop.mutations import MutationProposal
    from imu_denoise.config import ExperimentConfig
    from imu_denoise.observability import LoopController, MissionControlQueries

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imu_denoise.training import TrainingInterrupted  # noqa: E402

RESULTS_HEADER = [
    "timestamp",
    "iteration",
    "run_name",
    "status",
    "proposal_source",
    "metric_key",
    "metric_value",
    "model_name",
    "description",
    "overrides",
    "metrics_path",
]


@dataclass(frozen=True)
class LoopResult:
    """Single experiment outcome recorded by the autoresearch loop."""

    iteration: int
    run_name: str
    status: str
    proposal_source: str
    metric_key: str
    metric_value: float | None
    model_name: str
    description: str
    overrides: list[str]
    metrics_path: Path | None


@dataclass(frozen=True)
class BaselineReference:
    """Resolved baseline policy for a loop run."""

    include_baseline_run: bool
    metric_value: float | None
    run_id: str | None
    description: str


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


def _sanitize_tsv_field(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").strip()


def _ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(RESULTS_HEADER) + "\n")


def _append_result(path: Path, result: LoopResult) -> None:
    row = [
        datetime.now(tz=UTC).isoformat(),
        str(result.iteration),
        result.run_name,
        result.status,
        result.proposal_source,
        result.metric_key,
        "" if result.metric_value is None else f"{result.metric_value:.6f}",
        result.model_name,
        _sanitize_tsv_field(result.description),
        _sanitize_tsv_field(json.dumps(result.overrides, separators=(",", ":"))),
        "" if result.metrics_path is None else str(result.metrics_path),
    ]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def _is_better(candidate: float, incumbent: float | None, direction: str) -> bool:
    if incumbent is None:
        return True
    if direction == "maximize":
        return candidate > incumbent
    return candidate < incumbent


def _metric_from_summary(
    summary: Any,
    metric_key: str,
) -> float:
    if metric_key == "val_rmse":
        return float(summary.best_val_rmse)
    if metric_key == "final_val_loss":
        return float(summary.final_val_loss)
    raise ValueError(f"Unsupported autoresearch metric_key: {metric_key}")


def _build_run_overrides(
    *,
    iteration: int,
    proposal: MutationProposal,
    base_overrides: list[str],
    base_config: ExperimentConfig,
) -> list[str]:
    overrides = list(base_overrides)
    overrides.extend(proposal.overrides)
    overrides.append(f"name=autoresearch_{iteration:03d}")
    if base_config.autoresearch.time_budget_sec > 0:
        overrides.append(f"training.time_budget_sec={base_config.autoresearch.time_budget_sec}")
    return overrides


def _result_snapshot(result: LoopResult) -> dict[str, object]:
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


def _select_mutation_proposal(
    *,
    iteration: int,
    base_config: ExperimentConfig,
    rng: Random,
    results: list[LoopResult],
    fallback_proposal: MutationProposal,
    hermes_used_descriptions: set[str],
) -> tuple[MutationProposal, str, HermesQueryTrace | None, list[MutationProposal] | None]:
    from autoresearch_loop.mutations import default_mutation_pool

    if iteration == 0 or base_config.autoresearch.orchestrator != "hermes":
        return fallback_proposal, "static", None, None

    from autoresearch_loop.hermes import (
        HermesProposalError,
        choose_mutation_proposal_with_trace,
        hermes_backend_ready,
    )

    if not hermes_backend_ready(base_config.autoresearch.hermes, root=ROOT):
        return fallback_proposal, "static-fallback", None, None

    candidate_pool = default_mutation_pool()[1:]
    available_candidates = [
        proposal
        for proposal in candidate_pool
        if proposal.description not in hermes_used_descriptions
    ]
    if not available_candidates:
        hermes_used_descriptions.clear()
        available_candidates = candidate_pool[:]
        rng.shuffle(available_candidates)

    try:
        proposal, trace = choose_mutation_proposal_with_trace(
            config=base_config.autoresearch.hermes,
            iteration=iteration,
            metric_key=base_config.autoresearch.metric_key,
            metric_direction=base_config.autoresearch.metric_direction,
            history=[_result_snapshot(result) for result in results],
            candidates=available_candidates,
            root=ROOT,
        )
    except HermesProposalError as exc:
        return fallback_proposal, "static-fallback", exc.trace, available_candidates

    hermes_used_descriptions.add(proposal.description)
    return proposal, "hermes", trace, available_candidates


def _run_single_experiment(
    *,
    base_config: ExperimentConfig,
    overrides: list[str],
    metric_key: str,
    parent_run_id: str | None = None,
    iteration: int | None = None,
    run_id: str | None = None,
) -> tuple[Any, Any, str]:
    from imu_denoise.cli.common import build_model
    from imu_denoise.config import load_config_from_dict
    from imu_denoise.data.datamodule import create_dataloaders
    from imu_denoise.device import DeviceContext
    from imu_denoise.observability import ObservabilityWriter
    from imu_denoise.training import (
        Trainer,
        TrainingInterrupted,
        build_loss,
        build_optimizer_and_scheduler,
        seed_everything,
    )

    config = load_config_from_dict(asdict(base_config), overrides=overrides)
    observability = ObservabilityWriter.from_experiment_config(config)
    run_id = observability.start_run(
        name=config.name,
        phase="training",
        dataset=config.data.dataset,
        model=config.model.name,
        device=config.device.preferred,
        parent_run_id=parent_run_id,
        iteration=iteration,
        config=config,
        overrides=overrides,
        objective_metric=metric_key,
        objective_direction="minimize",
        source="runtime",
        run_id=run_id,
    )
    seed_everything(config.training.seed)
    device_ctx = DeviceContext.from_config(config.device)
    model = build_model(config)
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            config.data,
            config.training,
            device_ctx,
        )
        optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), config.training)
        trainer = Trainer(
            model=model,
            config=config,
            device_ctx=device_ctx,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=build_loss(config.training.loss),
            observability=observability,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        summary = trainer.fit(train_loader, val_loader, test_loader)
        _metric_from_summary(summary, metric_key)
        return config, summary, run_id
    except TrainingInterrupted:
        raise
    except Exception as exc:
        observability.finish_run(
            run_id=run_id,
            status="failed",
            summary={"message": str(exc)},
            source="runtime",
        )
        raise


def _best_metric_from_results(results: list[LoopResult], direction: str) -> float | None:
    valid = [result.metric_value for result in results if result.metric_value is not None]
    if not valid:
        return None
    return max(valid) if direction == "maximize" else min(valid)


def _resolve_baseline_reference(
    base_config: ExperimentConfig,
    queries: MissionControlQueries,
) -> BaselineReference:
    policy = base_config.autoresearch.baseline.mode

    if policy == "per_loop":
        return BaselineReference(
            include_baseline_run=True,
            metric_value=None,
            run_id=None,
            description="per-loop baseline",
        )

    if policy == "global":
        baseline = queries.find_best_global_incumbent(
            metric_key=base_config.autoresearch.metric_key,
            dataset=base_config.data.dataset,
            direction=base_config.autoresearch.metric_direction,
            reference_config=base_config,
        )
        if baseline is None:
            return BaselineReference(
                include_baseline_run=True,
                metric_value=None,
                run_id=None,
                description="global incumbent not found; falling back to per-loop baseline",
            )
        return BaselineReference(
            include_baseline_run=False,
            metric_value=float(baseline["metric_value"]),
            run_id=str(baseline["run_id"]),
            description=f"global incumbent {str(baseline['run_id'])[:8]}",
        )

    if policy == "manual":
        configured_run_id = base_config.autoresearch.baseline.run_id.strip()
        if not configured_run_id:
            raise ValueError("autoresearch.baseline.run_id is required when mode=manual")
        match = queries.resolve_id_fragment(configured_run_id)
        if match is None or match["entity_type"] != "run":
            run_id = configured_run_id
        else:
            run_id = str(match["id"])
        metric_value = queries.get_run_metric(
            run_id,
            metric_key=base_config.autoresearch.metric_key,
        )
        if metric_value is None:
            raise ValueError(
                f"Could not resolve baseline metric for manual baseline run: {configured_run_id}"
            )
        return BaselineReference(
            include_baseline_run=False,
            metric_value=metric_value,
            run_id=run_id,
            description=f"manual baseline {run_id[:8]}",
        )

    raise ValueError(f"Unsupported autoresearch baseline mode: {policy}")


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
    while True:
        loop_state = loop_controller.get_loop_state(loop_run_id)
        if loop_state is None:
            raise RuntimeError("Loop state disappeared while waiting for resume.")
        if bool(loop_state.get("stop_requested")) or bool(loop_state.get("terminate_requested")):
            return loop_state
        if loop_state["status"] != "paused":
            return loop_state
        loop_controller.heartbeat(
            loop_run_id=loop_run_id,
            current_iteration=current_iteration,
            max_iterations=total_iterations,
            batch_size=batch_size,
            pause_after_iteration=loop_state.get("pause_after_iteration"),
            pause_requested=bool(loop_state.get("pause_requested")),
            stop_requested=bool(loop_state.get("stop_requested")),
            terminate_requested=bool(loop_state.get("terminate_requested")),
            best_metric=best_metric,
            best_run_id=best_run_id,
            active_child_run_id=None,
            status="paused",
        )
        time.sleep(0.25)


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
    observability.finish_run(
        run_id=loop_run_id,
        status=status,
        summary={"message": message},
        source="runtime",
    )
    loop_controller.complete_loop(
        loop_run_id=loop_run_id,
        current_iteration=current_iteration,
        max_iterations=max_iterations,
        batch_size=batch_size,
        best_metric=best_metric,
        best_run_id=best_run_id,
        status=status,
    )


def _resolve_results_file(
    *,
    base_config: ExperimentConfig,
    loop_run_id: str,
    loop_name: str,
) -> Path:
    from imu_denoise.config import AutoResearchConfig
    from imu_denoise.utils.paths import build_run_paths

    configured = Path(base_config.autoresearch.results_file)
    default_path = Path(AutoResearchConfig().results_file)
    if configured == default_path:
        return build_run_paths(
            base_config.output_dir,
            run_name=loop_name,
            run_id=loop_run_id,
        ).loop_results_path
    return configured


def _safe_update_run_manifest(run_paths: Any, payload: dict[str, Any]) -> None:
    try:
        from imu_denoise.utils.paths import update_run_manifest

        update_run_manifest(run_paths, payload)
    except Exception:
        return


def run_autoresearch(
    *,
    config_paths: list[str],
    base_overrides: list[str] | None = None,
    max_iterations: int | None = None,
    batch_size: int | None = None,
    pause_enabled: bool = False,
) -> list[LoopResult]:
    """Run the baseline plus config mutations and record results to TSV."""
    from autoresearch_loop.mutations import MutationProposal, build_mutation_schedule
    from imu_denoise.cli.common import resolve_config
    from imu_denoise.config import load_config_from_dict
    from imu_denoise.observability import (
        LoopAlreadyRunningError,
        LoopController,
        MissionControlQueries,
        ObservabilityWriter,
        import_hermes_state,
    )
    from imu_denoise.observability.control import LOOP_PAUSED
    from imu_denoise.utils.paths import build_run_paths, write_run_manifest

    base_overrides = list(base_overrides or [])
    base_config = resolve_config(config_paths, base_overrides)
    observability = ObservabilityWriter.from_experiment_config(base_config)
    queries = MissionControlQueries(
        db_path=Path(base_config.observability.db_path),
        blob_dir=Path(base_config.observability.blob_dir),
    )
    loop_name = f"{base_config.name}-autoresearch"
    loop_run_id = observability.make_run_id(name=loop_name, phase="autoresearch_loop")
    loop_controller = LoopController.from_experiment_config(base_config, writer=observability)
    loop_run_id = observability.start_run(
        name=loop_name,
        phase="autoresearch_loop",
        dataset=base_config.data.dataset,
        model=base_config.model.name,
        device=base_config.device.preferred,
        config=base_config,
        overrides=base_overrides,
        objective_metric=base_config.autoresearch.metric_key,
        objective_direction=base_config.autoresearch.metric_direction,
        source="runtime",
        run_id=loop_run_id,
    )
    write_run_manifest(
        build_run_paths(base_config.output_dir, run_name=loop_name, run_id=loop_run_id),
        {
            "run_id": loop_run_id,
            "name": loop_name,
            "phase": "autoresearch_loop",
        },
    )
    total_iterations = (
        max_iterations if max_iterations is not None else base_config.autoresearch.max_iterations
    )
    results_file = _resolve_results_file(
        base_config=base_config,
        loop_run_id=loop_run_id,
        loop_name=loop_name,
    )
    _ensure_results_file(results_file)
    requested_batch_size = batch_size if pause_enabled and batch_size and batch_size > 0 else None

    rng = Random(base_config.training.seed)
    baseline_reference = _resolve_baseline_reference(base_config, queries)
    schedule = build_mutation_schedule(
        total_iterations,
        rng,
        include_baseline=baseline_reference.include_baseline_run,
    )
    total_scheduled_runs = len(schedule)
    results: list[LoopResult] = []
    best_metric = baseline_reference.metric_value
    if best_metric is None:
        best_metric = _best_metric_from_results(results, base_config.autoresearch.metric_direction)
    best_run_id: str | None = baseline_reference.run_id
    hermes_used_descriptions: set[str] = {
        result.description for result in results if result.proposal_source == "hermes"
    }
    try:
        loop_controller.initialize_loop(
            loop_run_id=loop_run_id,
            max_iterations=total_scheduled_runs,
            batch_size=requested_batch_size,
            pause_enabled=pause_enabled,
            current_iteration=len(results),
            best_metric=best_metric,
            best_run_id=best_run_id,
        )
    except LoopAlreadyRunningError:
        observability.finish_run(
            run_id=loop_run_id,
            status="failed",
            summary={"message": "another loop is already active"},
            source="runtime",
        )
        raise

    observability.append_event(
        run_id=loop_run_id,
        event_type="baseline_reference",
        level="INFO",
        title="baseline policy resolved",
        payload={
            "mode": base_config.autoresearch.baseline.mode,
            "include_baseline_run": baseline_reference.include_baseline_run,
            "baseline_run_id": baseline_reference.run_id,
            "baseline_metric_value": baseline_reference.metric_value,
            "description": baseline_reference.description,
        },
        source="runtime",
    )

    for iteration, fallback_proposal in enumerate(schedule[len(results) :], start=len(results)):
        loop_state = loop_controller.get_loop_state(loop_run_id)
        if loop_state is None:
            raise RuntimeError("Missing loop state for autoresearch loop.")
        if loop_state["status"] == "paused":
            loop_state = _wait_while_paused(
                loop_controller=loop_controller,
                loop_run_id=loop_run_id,
                total_iterations=total_scheduled_runs,
                batch_size=requested_batch_size,
                current_iteration=len(results),
                best_metric=best_metric,
                best_run_id=best_run_id,
            )

        if bool(loop_state.get("terminate_requested")):
            _finish_loop_with_status(
                observability=observability,
                loop_controller=loop_controller,
                loop_run_id=loop_run_id,
                current_iteration=len(results),
                max_iterations=total_scheduled_runs,
                batch_size=requested_batch_size,
                best_metric=best_metric,
                best_run_id=best_run_id,
                status="terminated",
                message=f"terminated after {len(results)} iterations",
            )
            return results
        if bool(loop_state.get("stop_requested")):
            _finish_loop_with_status(
                observability=observability,
                loop_controller=loop_controller,
                loop_run_id=loop_run_id,
                current_iteration=len(results),
                max_iterations=total_scheduled_runs,
                batch_size=requested_batch_size,
                best_metric=best_metric,
                best_run_id=best_run_id,
                status="stopped",
                message=f"stopped after {len(results)} iterations",
            )
            return results

        should_pause = bool(loop_state.get("pause_requested"))
        pause_reason = "manual"
        pause_after_iteration = loop_state.get("pause_after_iteration")
        if (
            isinstance(pause_after_iteration, int)
            and pause_after_iteration > 0
            and len(results) >= pause_after_iteration
        ):
            should_pause = True
            pause_reason = "batch"
        if should_pause:
            observability.append_event(
                run_id=loop_run_id,
                event_type=LOOP_PAUSED,
                level="INFO",
                title="loop paused",
                payload={"reason": pause_reason, "current_iteration": len(results)},
                source="runtime",
            )
            loop_controller.heartbeat(
                loop_run_id=loop_run_id,
                current_iteration=len(results),
                max_iterations=total_scheduled_runs,
                batch_size=requested_batch_size,
                pause_after_iteration=pause_after_iteration,
                pause_requested=False,
                stop_requested=bool(loop_state.get("stop_requested")),
                terminate_requested=bool(loop_state.get("terminate_requested")),
                best_metric=best_metric,
                best_run_id=best_run_id,
                active_child_run_id=None,
                status="paused",
            )
            loop_state = _wait_while_paused(
                loop_controller=loop_controller,
                loop_run_id=loop_run_id,
                total_iterations=total_scheduled_runs,
                batch_size=requested_batch_size,
                current_iteration=len(results),
                best_metric=best_metric,
                best_run_id=best_run_id,
            )

        queue_row = None
        if iteration > 0:
            queue_row = loop_controller.claim_next_queued_proposal(loop_run_id=loop_run_id)
        if queue_row is not None:
            proposal = MutationProposal(
                description=str(queue_row["description"]),
                overrides=list(queue_row["overrides"]),
            )
            proposal_source = "human-queued"
            hermes_trace = None
            candidate_pool = None
        else:
            proposal, proposal_source, hermes_trace, candidate_pool = _select_mutation_proposal(
                iteration=iteration,
                base_config=base_config,
                rng=rng,
                results=results,
                fallback_proposal=fallback_proposal,
                hermes_used_descriptions=hermes_used_descriptions,
            )
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
        if hermes_trace is not None:
            llm_call_id = observability.record_llm_call(
                run_id=loop_run_id,
                provider=base_config.autoresearch.hermes.provider,
                model=base_config.autoresearch.hermes.model,
                base_url=base_config.autoresearch.hermes.base_url,
                status=hermes_trace.status,
                latency_ms=hermes_trace.latency_ms,
                prompt=hermes_trace.prompt,
                response=hermes_trace.stdout,
                stdout_text=hermes_trace.stdout,
                stderr_text=hermes_trace.stderr,
                parsed_payload=hermes_trace.parsed_payload,
                command=hermes_trace.command,
                session_id=hermes_trace.session_id,
                reason=hermes_trace.reason,
                source="runtime",
            )
            if base_config.observability.import_hermes_state:
                import_hermes_state(
                    writer=observability,
                    hermes_home=Path(base_config.autoresearch.hermes.home_dir),
                )
        run_overrides = _build_run_overrides(
            iteration=iteration,
            proposal=proposal,
            base_overrides=base_overrides,
            base_config=base_config,
        )
        observability.append_event(
            run_id=loop_run_id,
            event_type="candidate_generation",
            level="INFO",
            title=f"iteration {iteration} proposal selected",
            payload={
                "proposal_source": proposal_source,
                "description": proposal.description,
                "overrides": run_overrides,
                "candidate_count": len(candidate_pool) if candidate_pool is not None else 0,
            },
            source="runtime",
        )
        experiment_run_id = f"training-{iteration:03d}-{uuid4().hex}"
        resolved_config = load_config_from_dict(asdict(base_config), overrides=run_overrides)
        selected_incumbent_run_id = best_run_id
        incumbent_config = (
            queries.get_run_config_payload(selected_incumbent_run_id)
            if selected_incumbent_run_id is not None
            else None
        )
        reference_kind = "incumbent" if incumbent_config is not None else "base"
        selection_rationale = (
            f"queued proposal #{queue_row['id']}"
            if queue_row is not None
            else (
                hermes_trace.reason
                if hermes_trace is not None and hermes_trace.reason
                else f"{proposal_source} proposal selected"
            )
        )
        candidate_count = (
            len(candidate_pool)
            if candidate_pool is not None
            else (1 if queue_row is not None or proposal_source.startswith("static") else None)
        )
        selection_event = observability.record_selection_event(
            run_id=experiment_run_id,
            loop_run_id=loop_run_id,
            iteration=iteration,
            proposal_source=proposal_source,
            description=proposal.description,
            incumbent_run_id=selected_incumbent_run_id,
            candidate_count=candidate_count,
            rationale=selection_rationale,
            policy_state={
                "baseline_mode": base_config.autoresearch.baseline.mode,
                "best_metric": best_metric,
                "best_run_id": best_run_id,
                "candidate_descriptions": (
                    [candidate.description for candidate in candidate_pool]
                    if candidate_pool is not None
                    else None
                ),
                "queued_proposal_id": None if queue_row is None else int(queue_row["id"]),
            },
            source="runtime",
        )
        change_set = observability.record_change_set(
            run_id=experiment_run_id,
            loop_run_id=loop_run_id,
            parent_run_id=selected_incumbent_run_id,
            incumbent_run_id=selected_incumbent_run_id,
            reference_kind=reference_kind,
            proposal_source=proposal_source,
            description=proposal.description,
            overrides=run_overrides,
            current_config=resolved_config,
            reference_config=incumbent_config if incumbent_config is not None else base_config,
            source="runtime",
        )
        experiment_run_paths = build_run_paths(
            base_config.output_dir,
            run_name=resolved_config.name,
            run_id=experiment_run_id,
        )
        _safe_update_run_manifest(
            experiment_run_paths,
            {
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

        try:
            config, summary, experiment_run_id = _run_single_experiment(
                base_config=base_config,
                overrides=run_overrides,
                metric_key=base_config.autoresearch.metric_key,
                parent_run_id=loop_run_id,
                iteration=iteration,
                run_id=experiment_run_id,
            )
            metric_value = _metric_from_summary(summary, base_config.autoresearch.metric_key)
            if iteration == 0 and baseline_reference.include_baseline_run:
                status = "baseline"
                best_metric = metric_value
                best_run_id = experiment_run_id
            elif _is_better(
                metric_value,
                best_metric,
                base_config.autoresearch.metric_direction,
            ):
                status = "keep"
                best_metric = metric_value
                best_run_id = experiment_run_id
            else:
                status = "discard"

            result = LoopResult(
                iteration=iteration,
                run_name=config.name,
                status=status,
                proposal_source=proposal_source,
                metric_key=base_config.autoresearch.metric_key,
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
                metric_key=base_config.autoresearch.metric_key,
                metric_value=metric_value,
                overrides=run_overrides,
                candidates=(
                    [
                        {
                            "description": candidate.description,
                            "overrides": candidate.overrides,
                        }
                        for candidate in candidate_pool
                    ]
                    if candidate_pool is not None
                    else None
                ),
                reason=hermes_trace.reason if hermes_trace is not None else None,
                llm_call_id=llm_call_id,
                source="runtime",
            )
            if queue_row is not None:
                loop_controller.mark_queue_applied(
                    proposal_id=int(queue_row["id"]),
                    loop_run_id=loop_run_id,
                    applied_run_id=experiment_run_id,
                )
            _safe_update_run_manifest(
                experiment_run_paths,
                {
                    "result": {
                        "status": status,
                        "metric_key": base_config.autoresearch.metric_key,
                        "metric_value": metric_value,
                        "compared_against_run_id": selected_incumbent_run_id,
                        "new_incumbent_run_id": best_run_id,
                    }
                },
            )
        except TrainingInterrupted as exc:
            result = LoopResult(
                iteration=iteration,
                run_name=f"autoresearch_{iteration:03d}",
                status=exc.status,
                proposal_source=proposal_source,
                metric_key=base_config.autoresearch.metric_key,
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
                metric_key=base_config.autoresearch.metric_key,
                metric_value=None,
                overrides=run_overrides,
                candidates=(
                    [
                        {
                            "description": candidate.description,
                            "overrides": candidate.overrides,
                        }
                        for candidate in candidate_pool
                    ]
                    if candidate_pool is not None
                    else None
                ),
                reason=str(exc),
                llm_call_id=llm_call_id,
                source="runtime",
            )
            _safe_update_run_manifest(
                experiment_run_paths,
                {
                    "result": {
                        "status": exc.status,
                        "metric_key": base_config.autoresearch.metric_key,
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
            _append_result(results_file, result)
            observability.register_artifact(
                run_id=loop_run_id,
                path=results_file,
                artifact_type="autoresearch_results",
                label="results_tsv",
                source="runtime",
            )
            _finish_loop_with_status(
                observability=observability,
                loop_controller=loop_controller,
                loop_run_id=loop_run_id,
                current_iteration=len(results),
                max_iterations=total_scheduled_runs,
                batch_size=requested_batch_size,
                best_metric=best_metric,
                best_run_id=best_run_id,
                status=exc.status,
                message=str(exc),
            )
            return results
        except Exception as exc:
            result = LoopResult(
                iteration=iteration,
                run_name=f"autoresearch_{iteration:03d}",
                status="crash",
                proposal_source=proposal_source,
                metric_key=base_config.autoresearch.metric_key,
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
                metric_key=base_config.autoresearch.metric_key,
                metric_value=None,
                overrides=run_overrides,
                candidates=(
                    [
                        {
                            "description": candidate.description,
                            "overrides": candidate.overrides,
                        }
                        for candidate in candidate_pool
                    ]
                    if candidate_pool is not None
                    else None
                ),
                reason=str(exc),
                llm_call_id=llm_call_id,
                source="runtime",
            )
            _safe_update_run_manifest(
                experiment_run_paths,
                {
                    "result": {
                        "status": "crash",
                        "metric_key": base_config.autoresearch.metric_key,
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

        _append_result(results_file, result)
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

    _finish_loop_with_status(
        observability=observability,
        loop_controller=loop_controller,
        loop_run_id=loop_run_id,
        current_iteration=len(results),
        max_iterations=total_scheduled_runs,
        batch_size=requested_batch_size,
        best_metric=best_metric,
        best_run_id=best_run_id,
        status="completed",
        message=f"completed {len(results)} iterations",
    )
    return results


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
