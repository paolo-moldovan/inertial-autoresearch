"""Tests for the Phase 3 training stack."""

from __future__ import annotations

from pathlib import Path

from imu_denoise.cli.common import build_model
from imu_denoise.config import (
    DataConfig,
    DeviceConfig,
    ExperimentConfig,
    ObservabilityConfig,
    TrainingConfig,
)
from imu_denoise.data.datamodule import create_dataloaders
from imu_denoise.device import DeviceContext
from imu_denoise.observability import MissionControlQueries
from imu_denoise.training import (
    Trainer,
    build_loss,
    build_optimizer_and_scheduler,
    seed_everything,
)


def test_trainer_runs_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    """Trainer should complete a tiny synthetic run and write artifacts."""
    config = ExperimentConfig(
        name="unit-train",
        output_dir=str(tmp_path / "artifacts"),
        log_dir=str(tmp_path / "artifacts" / "logs"),
        data=DataConfig(
            dataset="synthetic",
            window_size=16,
            stride=8,
            normalize=True,
            augment=False,
        ),
        training=TrainingConfig(
            epochs=2,
            batch_size=4,
            num_workers=0,
            seed=123,
            early_stop_patience=5,
        ),
        device=DeviceConfig(preferred="cpu"),
        observability=ObservabilityConfig(
            enabled=True,
            db_path=str(tmp_path / "artifacts" / "observability" / "mission_control.db"),
            blob_dir=str(tmp_path / "artifacts" / "observability" / "blobs"),
        ),
    )
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

    assert summary.best_epoch >= 1
    assert summary.best_val_rmse >= 0.0
    assert summary.artifacts.run_dir.parent.name == "runs"
    assert summary.artifacts.best_checkpoint.exists()
    assert summary.artifacts.last_checkpoint.exists()
    assert summary.artifacts.metrics_path.exists()
    assert summary.artifacts.history_path.exists()
    assert summary.artifacts.runtime_log_path.exists()
    assert (summary.artifacts.run_dir / "figures" / "training_curves.png").exists()
    assert (summary.artifacts.run_dir / "figures" / "denoising_comparison.png").exists()
    assert (summary.artifacts.run_dir / "figures" / "psd.png").exists()
    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )
    runs = queries.list_runs(limit=10)
    assert any(run["name"] == "unit-train" and run["phase"] == "training" for run in runs)
    artifacts = queries.list_artifacts()
    assert any(artifact["artifact_type"] == "training_metrics" for artifact in artifacts)
    assert any(
        artifact["artifact_type"] == "figure" and artifact["label"] == "training_curves"
        for artifact in artifacts
    )
