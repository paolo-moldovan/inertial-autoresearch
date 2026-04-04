"""Loop-session setup helpers for the IMU autoresearch runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from imu_denoise.autoresearch.adapter import IMUProjectAdapter

from . import artifacts as artifact_helpers
from . import lifecycle as lifecycle_helpers


@dataclass(frozen=True)
class LoopRuntimeContext:
    """Shared runtime resources and startup state for one autoresearch loop."""

    adapter: IMUProjectAdapter
    base_config: Any
    base_overrides: list[str]
    observability: Any
    queries: Any
    loop_controller: Any
    loop_name: str
    loop_run_id: str
    results_file: Path
    requested_batch_size: int | None
    rng: Random
    baseline_reference: lifecycle_helpers.BaselineReference
    mutation_catalog: list[Any]
    schedule: list[Any]
    total_scheduled_runs: int


def initialize_runtime_context(
    *,
    config_paths: list[str],
    base_overrides: list[str],
    max_iterations: int | None,
    batch_size: int | None,
    pause_enabled: bool,
) -> LoopRuntimeContext:
    from imu_denoise.autoresearch.mutations import MutationProposal, build_mutation_schedule
    from imu_denoise.observability import LoopController, MissionControlQueries, ObservabilityWriter
    from imu_denoise.observability.lineage import data_regime_fingerprint
    from imu_denoise.utils.paths import build_run_paths, write_run_manifest

    adapter = IMUProjectAdapter()
    base_config = adapter.resolve_base_config(
        config_paths=config_paths,
        base_overrides=base_overrides,
    )
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
            "regime_fingerprint": data_regime_fingerprint(base_config),
        },
    )
    total_iterations = (
        max_iterations if max_iterations is not None else base_config.autoresearch.max_iterations
    )
    results_file = artifact_helpers.resolve_results_file(
        base_config=base_config,
        loop_run_id=loop_run_id,
        loop_name=loop_name,
    )
    artifact_helpers.ensure_results_file(results_file)
    requested_batch_size = batch_size if pause_enabled and batch_size and batch_size > 0 else None
    rng = Random(base_config.training.seed)
    baseline_reference = lifecycle_helpers.resolve_baseline_reference(
        base_config=base_config,
        queries=queries,
    )
    mutation_catalog = [
        MutationProposal(
            description=proposal.description,
            overrides=list(proposal.overrides),
            groups=tuple(proposal.groups),
            architecture_change=proposal.architecture_change,
        )
        for proposal in adapter.get_mutation_catalog()
    ]
    schedule = build_mutation_schedule(
        total_iterations,
        rng,
        include_baseline=baseline_reference.include_baseline_run,
        catalog=mutation_catalog,
    )
    return LoopRuntimeContext(
        adapter=adapter,
        base_config=base_config,
        base_overrides=base_overrides,
        observability=observability,
        queries=queries,
        loop_controller=loop_controller,
        loop_name=loop_name,
        loop_run_id=loop_run_id,
        results_file=results_file,
        requested_batch_size=requested_batch_size,
        rng=rng,
        baseline_reference=baseline_reference,
        mutation_catalog=mutation_catalog,
        schedule=schedule,
        total_scheduled_runs=len(schedule),
    )


def start_loop_session(
    *,
    context: LoopRuntimeContext,
    current_iteration: int,
    best_metric: float | None,
    best_run_id: str | None,
    pause_enabled: bool,
) -> None:
    from imu_denoise.observability import LoopAlreadyRunningError

    try:
        context.loop_controller.initialize_loop(
            loop_run_id=context.loop_run_id,
            max_iterations=context.total_scheduled_runs,
            batch_size=context.requested_batch_size,
            pause_enabled=pause_enabled,
            current_iteration=current_iteration,
            best_metric=best_metric,
            best_run_id=best_run_id,
        )
    except LoopAlreadyRunningError:
        context.observability.finish_run(
            run_id=context.loop_run_id,
            status="failed",
            summary={"message": "another loop is already active"},
            source="runtime",
        )
        raise

    context.observability.append_event(
        run_id=context.loop_run_id,
        event_type="baseline_reference",
        level="INFO",
        title="baseline policy resolved",
        payload={
            "mode": context.base_config.autoresearch.baseline.mode,
            "include_baseline_run": context.baseline_reference.include_baseline_run,
            "baseline_run_id": context.baseline_reference.run_id,
            "baseline_metric_value": context.baseline_reference.metric_value,
            "description": context.baseline_reference.description,
        },
        source="runtime",
    )
