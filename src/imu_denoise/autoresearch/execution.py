"""IMU-domain execution and config-resolution helpers for autoresearch."""

from __future__ import annotations

import json
from typing import Any, cast

from imu_denoise.config import ExperimentConfig
from imu_denoise.training import TrainingInterrupted


def metric_from_summary(
    summary: Any,
    metric_key: str,
) -> float:
    """Extract the configured objective metric from a training summary."""
    if getattr(summary, "best_metric_key", None) == metric_key:
        return float(summary.best_metric_value)
    best_eval_metrics = getattr(summary, "best_eval_metrics", None)
    if isinstance(best_eval_metrics, dict) and metric_key in best_eval_metrics:
        return float(best_eval_metrics[metric_key])
    if metric_key == "val_rmse":
        return float(summary.best_val_rmse)
    if metric_key == "final_val_loss":
        return float(summary.final_val_loss)
    raise ValueError(f"Unsupported autoresearch metric_key: {metric_key}")


def resolve_reference_config_payload(
    *,
    base_config: ExperimentConfig,
    incumbent_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the reference payload used when composing the next config."""
    if incumbent_config is not None:
        return cast(dict[str, Any], json.loads(json.dumps(incumbent_config)))
    return cast(dict[str, Any], json.loads(json.dumps(_asdict_config(base_config))))


def resolve_iteration_config(
    *,
    base_config: ExperimentConfig,
    base_overrides: list[str],
    proposal_overrides: list[str],
    incumbent_config: dict[str, Any] | None = None,
    extra_overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Compose the iteration config from base, incumbent, and proposal overrides."""
    from imu_denoise.config import load_config_from_dict

    reference_payload = resolve_reference_config_payload(
        base_config=base_config,
        incumbent_config=incumbent_config,
    )
    overrides = [*base_overrides, *proposal_overrides, *(extra_overrides or [])]
    return load_config_from_dict(reference_payload, overrides=overrides)


def run_single_experiment(
    *,
    config: ExperimentConfig,
    overrides: list[str],
    metric_key: str,
    parent_run_id: str | None = None,
    iteration: int | None = None,
    run_id: str | None = None,
) -> tuple[Any, Any, str]:
    """Run one IMU training experiment and return resolved config, summary, and run id."""
    from imu_denoise.cli.common import build_model
    from imu_denoise.data.datamodule import create_dataloaders
    from imu_denoise.device import DeviceContext
    from imu_denoise.observability import ObservabilityWriter
    from imu_denoise.observability.training_hooks import (
        build_training_control,
        build_training_hooks,
    )
    from imu_denoise.training import (
        Trainer,
        build_loss,
        build_optimizer_and_scheduler,
        seed_everything,
    )

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
        data_bundle = create_dataloaders(
            config.data,
            config.training,
            device_ctx,
        )
        optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), config.training)
        training_hooks = build_training_hooks(config=config, observability=observability)
        training_control = build_training_control(
            parent_run_id=parent_run_id,
            run_id=run_id,
            observability=observability,
        )
        trainer = Trainer(
            model=model,
            config=config,
            device_ctx=device_ctx,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=build_loss(config.training),
            training_hooks=training_hooks,
            training_control=training_control,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        summary = trainer.fit(data_bundle)
        metric_from_summary(summary, metric_key)
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


def _asdict_config(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize the config via stdlib dataclass handling without importing observability."""
    from dataclasses import asdict

    return asdict(config)
