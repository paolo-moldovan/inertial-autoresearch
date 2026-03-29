"""Training CLI preflight."""

from __future__ import annotations

import argparse
from typing import Any

import torch

from imu_denoise.cli.common import add_common_config_arguments, build_model, resolve_config
from imu_denoise.data.datamodule import create_dataloaders
from imu_denoise.device import DeviceContext
from imu_denoise.observability import ObservabilityWriter
from imu_denoise.training import (
    Trainer,
    build_loss,
    build_optimizer_and_scheduler,
    seed_everything,
)
from imu_denoise.utils.paths import build_run_paths, update_run_manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the train CLI."""
    parser = argparse.ArgumentParser(description="IMU denoising training preflight.")
    add_common_config_arguments(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and model construction without starting training.",
    )
    return parser


def run_command(args: Any) -> int:
    """Train a denoising model or run a training preflight."""
    config = resolve_config(args.config, args.overrides)
    seed_everything(config.training.seed)
    device_ctx = DeviceContext.from_config(config.device)
    model = build_model(config)
    observability = ObservabilityWriter.from_experiment_config(config)
    run_id = observability.make_run_id(name=config.name, phase="training")
    run_paths = build_run_paths(config.output_dir, run_name=config.name, run_id=run_id)

    if config.device.compile and device_ctx.supports_compile:
        model = torch.compile(model)

    checkpoint_dir = run_paths.checkpoints_dir
    figures_dir = run_paths.figures_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Resolved experiment configuration:")
    print(f"  name: {config.name}")
    print(f"  dataset: {config.data.dataset}")
    print(f"  model: {config.model.name}")
    print(f"  device: {device_ctx.device.type}")
    print(f"  dtype: {device_ctx.dtype}")
    print(f"  amp_enabled: {device_ctx.amp_enabled}")
    print(f"  pin_memory: {device_ctx.pin_memory}")
    print(f"  run_dir: {run_paths.root}")
    print(f"  checkpoint_dir: {checkpoint_dir}")
    print(f"  log_dir: {run_paths.logs_dir}")
    print(f"  figures_dir: {figures_dir}")
    print(f"  model_class: {model.__class__.__name__}")

    if args.dry_run:
        return 0

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
    )
    summary = trainer.fit(train_loader, val_loader, test_loader)
    selection_event = observability.record_selection_event(
        run_id=summary.run_id,
        loop_run_id=None,
        iteration=None,
        proposal_source="manual",
        description="manual run",
        incumbent_run_id=None,
        candidate_count=1,
        rationale="launched manually from the CLI",
        policy_state={"mode": "manual"},
        source="runtime",
    )
    change_set = observability.record_change_set(
        run_id=summary.run_id,
        loop_run_id=None,
        parent_run_id=None,
        incumbent_run_id=None,
        reference_kind="manual",
        proposal_source="manual",
        description="manual run",
        overrides=list(args.overrides),
        current_config=config,
        reference_config=None,
        source="runtime",
    )
    observability.record_decision(
        run_id=summary.run_id,
        iteration=None,
        proposal_source="manual",
        description="manual run",
        status="completed",
        metric_key="val_rmse",
        metric_value=summary.best_val_rmse,
        overrides=list(args.overrides),
        source="runtime",
    )
    update_run_manifest(
        run_paths,
        {
            "resolved_config": observability.config_payload(config),
            "selection_event": selection_event,
            "change_set": change_set,
        },
    )

    print("Training complete:")
    print(f"  best_epoch: {summary.best_epoch}")
    print(f"  best_val_rmse: {summary.best_val_rmse:.6f}")
    print(f"  final_train_loss: {summary.final_train_loss:.6f}")
    print(f"  final_val_loss: {summary.final_val_loss:.6f}")
    print(f"  training_seconds: {summary.training_seconds:.2f}")
    print(f"  best_checkpoint: {summary.artifacts.best_checkpoint}")
    print(f"  metrics_path: {summary.artifacts.metrics_path}")
    return 0


def main() -> int:
    """CLI entrypoint."""
    return run_command(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
