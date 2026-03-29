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
) -> tuple[MutationProposal, str]:
    from autoresearch_loop.mutations import default_mutation_pool

    if iteration == 0 or base_config.autoresearch.orchestrator != "hermes":
        return fallback_proposal, "static"

    from autoresearch_loop.hermes import (
        HermesProposalError,
        choose_mutation_proposal,
        hermes_backend_ready,
    )

    if not hermes_backend_ready(base_config.autoresearch.hermes, root=ROOT):
        return fallback_proposal, "static-fallback"

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
        proposal = choose_mutation_proposal(
            config=base_config.autoresearch.hermes,
            iteration=iteration,
            metric_key=base_config.autoresearch.metric_key,
            metric_direction=base_config.autoresearch.metric_direction,
            history=[_result_snapshot(result) for result in results],
            candidates=available_candidates,
            root=ROOT,
        )
    except HermesProposalError:
        return fallback_proposal, "static-fallback"

    hermes_used_descriptions.add(proposal.description)
    return proposal, "hermes"


def _run_single_experiment(
    *,
    config_paths: list[str],
    overrides: list[str],
    metric_key: str,
) -> tuple[Any, Any]:
    from imu_denoise.cli.common import build_model
    from imu_denoise.config import load_config
    from imu_denoise.data.datamodule import create_dataloaders
    from imu_denoise.device import DeviceContext
    from imu_denoise.training import (
        Trainer,
        build_loss,
        build_optimizer_and_scheduler,
        seed_everything,
    )

    config = load_config(*config_paths, overrides=overrides)
    seed_everything(config.training.seed)
    device_ctx = DeviceContext.from_config(config.device)
    model = build_model(config)

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
    )
    summary = trainer.fit(train_loader, val_loader, test_loader)
    _metric_from_summary(summary, metric_key)
    return config, summary


def run_autoresearch(
    *,
    config_paths: list[str],
    base_overrides: list[str] | None = None,
    max_iterations: int | None = None,
) -> list[LoopResult]:
    """Run the baseline plus config mutations and record results to TSV."""
    from autoresearch_loop.mutations import build_mutation_schedule
    from imu_denoise.config import load_config

    base_overrides = list(base_overrides or [])
    base_config = load_config(*config_paths, overrides=base_overrides)
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
        proposal, proposal_source = _select_mutation_proposal(
            iteration=iteration,
            base_config=base_config,
            rng=rng,
            results=results,
            fallback_proposal=fallback_proposal,
            hermes_used_descriptions=hermes_used_descriptions,
        )
        run_overrides = _build_run_overrides(
            iteration=iteration,
            proposal=proposal,
            base_overrides=base_overrides,
            base_config=base_config,
        )

        try:
            config, summary = _run_single_experiment(
                config_paths=config_paths,
                overrides=run_overrides,
                metric_key=base_config.autoresearch.metric_key,
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

        _append_result(results_file, result)
        results.append(result)

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
