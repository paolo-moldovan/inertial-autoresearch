"""Tests for the Phase 3 training stack."""

from __future__ import annotations

import json
from pathlib import Path

from imu_denoise.cli.common import build_model
from imu_denoise.config import (
    DataConfig,
    DeviceConfig,
    EvaluationConfig,
    ExperimentConfig,
    ObservabilityConfig,
    TrainingConfig,
)
from imu_denoise.data.datamodule import create_dataloaders
from imu_denoise.device import DeviceContext
from imu_denoise.observability import MissionControlQueries
from imu_denoise.observability.training_hooks import build_training_hooks
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
    data_bundle = create_dataloaders(
        config.data,
        config.training,
        device_ctx,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), config.training)
    training_hooks = build_training_hooks(config=config)

    trainer = Trainer(
        model=model,
        config=config,
        device_ctx=device_ctx,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=build_loss(config.training),
        training_hooks=training_hooks,
    )
    summary = trainer.fit(data_bundle)

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


def test_trainer_honors_evaluation_frequency_and_preserves_history(tmp_path: Path) -> None:
    """Skipped evaluation epochs should still be logged while expensive eval runs less often."""
    config = ExperimentConfig(
        name="eval-frequency",
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
            epochs=3,
            batch_size=4,
            num_workers=0,
            seed=123,
            early_stop_patience=5,
        ),
        evaluation=EvaluationConfig(frequency_epochs=2, metrics=["rmse"]),
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
    data_bundle = create_dataloaders(config.data, config.training, device_ctx)
    optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), config.training)
    training_hooks = build_training_hooks(config=config)

    trainer = Trainer(
        model=model,
        config=config,
        device_ctx=device_ctx,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=build_loss(config.training),
        training_hooks=training_hooks,
    )
    summary = trainer.fit(data_bundle)

    history = [
        json.loads(line)
        for line in summary.artifacts.history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["evaluated"] for row in history] == [False, True, True]
    assert history[0]["val_rmse"] is None
    assert history[1]["val_rmse"] is not None


def test_weighted_loss_supports_sensor_type_and_channel_weights() -> None:
    """Loss weighting should be deterministic for mixed accel/gyro errors."""
    import torch

    pred = torch.tensor([[[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]]], dtype=torch.float32)
    target = torch.zeros_like(pred)

    sensor_weighted = build_loss(
        TrainingConfig(loss="mse", accel_loss_weight=1.0, gyro_loss_weight=2.0)
    )
    channel_weighted = build_loss(
        TrainingConfig(loss="mse", channel_loss_weights=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    )

    assert float(sensor_weighted(pred, target)) == float(channel_weighted(pred, target))


def test_trainer_can_track_sequence_metric_as_objective(tmp_path: Path) -> None:
    """Sequence-aware metrics should be usable as the trainer's best-run objective."""
    config = ExperimentConfig(
        name="sequence-objective",
        output_dir=str(tmp_path / "artifacts"),
        log_dir=str(tmp_path / "artifacts" / "logs"),
        data=DataConfig(
            dataset="synthetic",
            window_size=16,
            stride=8,
            normalize=True,
            augment=False,
            dataset_kwargs={
                "duration_sec": 1.0,
                "rate_hz": 20.0,
                "num_sequences": 4,
                "seed": 11,
            },
        ),
        training=TrainingConfig(
            epochs=2,
            batch_size=4,
            num_workers=0,
            seed=123,
            early_stop_patience=5,
        ),
        evaluation=EvaluationConfig(
            frequency_epochs=1,
            metrics=["rmse", "sequence_rmse"],
            reconstruction="hann",
        ),
        autoresearch=ExperimentConfig().autoresearch.__class__(metric_key="sequence_rmse"),
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
    data_bundle = create_dataloaders(config.data, config.training, device_ctx)
    optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), config.training)
    training_hooks = build_training_hooks(config=config)

    trainer = Trainer(
        model=model,
        config=config,
        device_ctx=device_ctx,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=build_loss(config.training),
        training_hooks=training_hooks,
    )
    summary = trainer.fit(data_bundle)

    assert summary.best_metric_key == "sequence_rmse"
    assert summary.best_metric_value >= 0.0
    metrics = json.loads(summary.artifacts.metrics_path.read_text(encoding="utf-8"))
    assert metrics["best_metric_key"] == "sequence_rmse"
