"""Local config-first autoresearch loop for IMU denoising."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autoresearch_loop.hermes import HermesQueryTrace
    from autoresearch_loop.mutations import MutationProposal
    from imu_denoise.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the local autoresearch loop."""
    parser = argparse.ArgumentParser(description="Run local config-first autoresearch iterations.")
    parser.add_argument(
        "--config",
        action="append",
        default=[
            "configs/base.yaml",
            "configs/device.yaml",
            "configs/models/lstm.yaml",
            "configs/training/quick.yaml",
            "configs/autoresearch.yaml",
        ],
        help="Config path(s) to merge. Defaults to a synthetic quick autoresearch stack.",
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
    config_paths: list[str],
    overrides: list[str],
    metric_key: str,
    parent_run_id: str | None = None,
    iteration: int | None = None,
) -> tuple[Any, Any, str]:
    from imu_denoise.cli.common import build_model
    from imu_denoise.config import load_config
    from imu_denoise.data.datamodule import create_dataloaders
    from imu_denoise.device import DeviceContext
    from imu_denoise.observability import ObservabilityWriter
    from imu_denoise.training import (
        Trainer,
        build_loss,
        build_optimizer_and_scheduler,
        seed_everything,
    )

    config = load_config(*config_paths, overrides=overrides)
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
    except Exception as exc:
        observability.finish_run(
            run_id=run_id,
            status="failed",
            summary={"message": str(exc)},
            source="runtime",
        )
        raise


def run_autoresearch(
    *,
    config_paths: list[str],
    base_overrides: list[str] | None = None,
    max_iterations: int | None = None,
) -> list[LoopResult]:
    """Run the baseline plus config mutations and record results to TSV."""
    from autoresearch_loop.mutations import build_mutation_schedule
    from imu_denoise.config import load_config
    from imu_denoise.observability import ObservabilityWriter, import_hermes_state

    base_overrides = list(base_overrides or [])
    base_config = load_config(*config_paths, overrides=base_overrides)
    observability = ObservabilityWriter.from_experiment_config(base_config)
    loop_run_id = observability.start_run(
        name=f"{base_config.name}-autoresearch",
        phase="autoresearch_loop",
        dataset=base_config.data.dataset,
        model=base_config.model.name,
        device=base_config.device.preferred,
        config=base_config,
        overrides=base_overrides,
        objective_metric=base_config.autoresearch.metric_key,
        objective_direction=base_config.autoresearch.metric_direction,
        source="runtime",
    )
    total_iterations = (
        max_iterations if max_iterations is not None else base_config.autoresearch.max_iterations
    )
    results_file = Path(base_config.autoresearch.results_file)
    _ensure_results_file(results_file)

    rng = Random(base_config.training.seed)
    schedule = build_mutation_schedule(total_iterations, rng)
    best_metric: float | None = None
    results: list[LoopResult] = []
    hermes_used_descriptions: set[str] = set()

    for iteration, fallback_proposal in enumerate(schedule):
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
            epoch=iteration,
            best_metric=best_metric,
            last_metric=results[-1].metric_value if results else None,
            message=f"iteration {iteration}",
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

        try:
            config, summary, experiment_run_id = _run_single_experiment(
                config_paths=config_paths,
                overrides=run_overrides,
                metric_key=base_config.autoresearch.metric_key,
                parent_run_id=loop_run_id,
                iteration=iteration,
            )
            metric_value = _metric_from_summary(summary, base_config.autoresearch.metric_key)
            if iteration == 0:
                status = "baseline"
                best_metric = metric_value
            elif _is_better(
                metric_value,
                best_metric,
                base_config.autoresearch.metric_direction,
            ):
                status = "keep"
                best_metric = metric_value
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

        _append_result(results_file, result)
        observability.register_artifact(
            run_id=loop_run_id,
            path=results_file,
            artifact_type="autoresearch_results",
            label="results_tsv",
            source="runtime",
        )
        results.append(result)

    observability.finish_run(
        run_id=loop_run_id,
        status="completed",
        summary={"message": f"completed {len(results)} iterations"},
        source="runtime",
    )
    return results


def main() -> int:
    """CLI entrypoint for the local autoresearch loop."""
    args = build_parser().parse_args()
    results = run_autoresearch(
        config_paths=args.config,
        base_overrides=args.overrides,
        max_iterations=args.max_iterations,
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
