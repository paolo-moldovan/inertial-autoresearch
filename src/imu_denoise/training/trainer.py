"""Training loop for IMU denoising models."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from autoresearch_core.training import (
    NoOpTrainingControl,
    NoOpTrainingHooks,
    TrainingControl,
    TrainingHooks,
)
from imu_denoise.config.schema import ExperimentConfig
from imu_denoise.data.datamodule import DataBundle
from imu_denoise.device.context import DeviceContext
from imu_denoise.evaluation.evaluator import Evaluator
from imu_denoise.models.base import BaseDenoiser
from imu_denoise.training.callbacks import CheckpointManager, EarlyStopping
from imu_denoise.training.losses import LossFn
from imu_denoise.utils.io import save_metrics
from imu_denoise.utils.logging import setup_logger
from imu_denoise.utils.paths import build_run_paths, write_run_manifest


class TrainingInterrupted(RuntimeError):
    """Raised when an external control-plane signal interrupts training."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class TrainingArtifacts:
    """Paths produced by a training run."""

    run_dir: Path
    checkpoint_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metrics_path: Path
    history_path: Path
    runtime_log_path: Path


@dataclass(frozen=True)
class TrainingSummary:
    """Compact result of a completed training run."""

    run_id: str
    best_epoch: int
    best_val_rmse: float
    best_metric_key: str
    best_metric_value: float
    best_eval_metrics: dict[str, float]
    final_train_loss: float
    final_val_loss: float
    training_seconds: float
    artifacts: TrainingArtifacts


class Trainer:
    """Train and validate an IMU denoising model."""

    def __init__(
        self,
        *,
        model: BaseDenoiser,
        config: ExperimentConfig,
        device_ctx: DeviceContext,
        optimizer: Optimizer,
        scheduler: LRScheduler | ReduceLROnPlateau | None,
        loss_fn: LossFn,
        training_hooks: TrainingHooks | None = None,
        training_control: TrainingControl | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        self.model = device_ctx.to_device(model)
        self.config = config
        self.device_ctx = device_ctx
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = device_ctx.create_scaler()
        self.training_hooks = training_hooks or NoOpTrainingHooks()
        self.run_id = run_id or self.training_hooks.make_run_id(name=config.name, phase="training")
        self.training_control = training_control or NoOpTrainingControl()
        self.run_paths = build_run_paths(
            config.output_dir,
            run_name=config.name,
            run_id=self.run_id,
        )
        self.logger = setup_logger(
            f"{config.name}.{self.run_paths.root.name}",
            log_dir=str(self.run_paths.logs_dir),
            log_filename="runtime",
        )
        self.checkpoints = CheckpointManager(self.run_paths.checkpoints_dir)
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stop_patience,
            mode="max" if config.autoresearch.metric_direction == "maximize" else "min",
        )
        self.checkpoints.mode = (
            "max" if config.autoresearch.metric_direction == "maximize" else "min"
        )
        self.history_path = self.run_paths.history_path
        self.parent_run_id = parent_run_id

    def fit(
        self,
        data_bundle: DataBundle,
    ) -> TrainingSummary:
        """Run the training loop and return a summary plus artifact paths."""
        train_loader = data_bundle.train_loader
        val_loader = data_bundle.val_loader
        test_loader = data_bundle.test_loader
        start_time = time.perf_counter()
        best_epoch = 0
        best_val_rmse = float("inf")
        best_metric_value = float("inf")
        best_eval_metrics: dict[str, float] = {}
        last_train_loss = float("nan")
        last_val_loss = float("nan")
        objective_metric_key = self.config.autoresearch.metric_key
        run_id = self.training_hooks.start_run(
            name=self.config.name,
            phase="training",
            dataset=self.config.data.dataset,
            model=self.config.model.name,
            device=self.device_ctx.device.type,
            parent_run_id=self.parent_run_id,
            config=self.config,
            objective_metric="val_rmse",
            objective_direction="minimize",
            source="runtime",
            run_id=self.run_id,
        )
        log_handler = self.training_hooks.create_log_handler(run_id)
        self.logger.addHandler(log_handler)
        write_run_manifest(
            self.run_paths,
            {
                "run_id": run_id,
                "name": self.config.name,
                "phase": "training",
                "parent_run_id": self.parent_run_id,
            },
        )
        self.training_hooks.update_status(
            run_id=run_id,
            phase="training",
            message=f"device={self.device_ctx.device.type} dtype={self.device_ctx.dtype}",
            source="runtime",
        )

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if self.history_path.exists():
            self.history_path.unlink()

        try:
            for epoch in range(1, self.config.training.epochs + 1):
                train_loss = self._run_epoch(train_loader, training=True)
                val_loss = self._run_epoch(val_loader, training=False)
                should_evaluate = self._should_run_full_evaluation(epoch)
                metrics: dict[str, float] | None = None
                val_rmse: float | None = None
                objective_metric_value: float | None = None
                if should_evaluate:
                    metrics = (
                        Evaluator(self.model, self.device_ctx)
                        .with_config(self.config.evaluation, logger=self.logger)
                        .evaluate(
                            val_loader,
                            fs=data_bundle.sampling_rate_hz,
                        )
                    )
                    rmse_metric = metrics.get("rmse")
                    if rmse_metric is None:
                        raise ValueError(
                            "training requires evaluation.metrics to include 'rmse' so "
                            "best_val_rmse can be tracked."
                        )
                    val_rmse = float(rmse_metric)
                    objective_metric_value = self._objective_metric_value(
                        metrics=metrics,
                        val_loss=val_loss,
                    )
                lr = self.optimizer.param_groups[0]["lr"]

                self._write_history(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_rmse": val_rmse,
                        "lr": lr,
                        "evaluated": should_evaluate,
                    }
                )
                if val_rmse is None:
                    self.logger.info(
                        "Epoch %d/%d train_loss=%.6f val_loss=%.6f val_rmse=skipped lr=%.6g",
                        epoch,
                        self.config.training.epochs,
                        train_loss,
                        val_loss,
                        lr,
                    )
                else:
                    self.logger.info(
                        "Epoch %d/%d train_loss=%.6f val_loss=%.6f val_rmse=%.6f lr=%.6g",
                        epoch,
                        self.config.training.epochs,
                        train_loss,
                        val_loss,
                        val_rmse,
                        lr,
                    )

                self._step_scheduler(val_loss)
                checkpoint_metric = (
                    objective_metric_value
                    if objective_metric_value is not None
                    else (
                        best_metric_value
                        if best_metric_value != float("inf")
                        else float("inf")
                    )
                )
                is_best = False
                checkpoint_extra = {"val_metrics": metrics} if metrics is not None else None
                if objective_metric_value is not None:
                    is_best = self.checkpoints.save(
                        epoch=epoch,
                        metric_value=objective_metric_value,
                        model=self.model,
                        optimizer=self.optimizer,
                        extra=checkpoint_extra,
                    )
                else:
                    self.checkpoints.save_last(
                        epoch=epoch,
                        metric_value=checkpoint_metric,
                        model=self.model,
                        optimizer=self.optimizer,
                        extra=checkpoint_extra,
                    )
                if is_best and objective_metric_value is not None:
                    best_epoch = epoch
                    best_metric_value = objective_metric_value
                    best_eval_metrics = dict(metrics or {})
                    if val_rmse is not None:
                        best_val_rmse = val_rmse
                    self.training_hooks.register_artifact(
                        run_id=run_id,
                        path=self.checkpoints.best_path,
                        artifact_type="checkpoint",
                        label="best",
                        metadata={
                            "epoch": epoch,
                            "val_rmse": val_rmse,
                            "objective_metric_key": objective_metric_key,
                            "objective_metric_value": objective_metric_value,
                            "evaluated": True,
                        },
                        source="runtime",
                    )

                self.training_hooks.register_artifact(
                    run_id=run_id,
                    path=self.checkpoints.last_path,
                    artifact_type="checkpoint",
                    label="last",
                    metadata={"epoch": epoch, "val_rmse": val_rmse, "evaluated": should_evaluate},
                    source="runtime",
                )

                last_train_loss = train_loss
                last_val_loss = val_loss
                self.training_hooks.record_epoch(
                    run_id=run_id,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_rmse=val_rmse,
                    lr=lr,
                    best_metric=best_val_rmse if best_val_rmse != float("inf") else None,
                    source="runtime",
                )
                heartbeat_metric = (
                    best_metric_value
                    if best_metric_value != float("inf")
                    else best_val_rmse
                )
                self.training_control.heartbeat(
                    best_metric=heartbeat_metric,
                    active_child_run_id=self.run_id,
                )

                if self.config.training.time_budget_sec > 0:
                    elapsed = time.perf_counter() - start_time
                    if elapsed >= self.config.training.time_budget_sec:
                        self.logger.warning("Stopping early because the time budget was reached.")
                        self.training_hooks.append_event(
                            run_id=run_id,
                            event_type="time_budget_reached",
                            level="WARNING",
                            title="training time budget reached",
                            payload={"elapsed_seconds": elapsed},
                            source="runtime",
                        )
                        break

                self.training_control.check_abort()

                if objective_metric_value is not None and self.early_stopping.update(
                    objective_metric_value
                ):
                    self.logger.info("Stopping early after %d epochs without improvement.", epoch)
                    self.training_hooks.append_event(
                        run_id=run_id,
                        event_type="early_stop",
                        level="INFO",
                        title="early stopping triggered",
                        payload={
                            "epoch": epoch,
                            "val_rmse": val_rmse,
                            "objective_metric_key": objective_metric_key,
                            "objective_metric_value": objective_metric_value,
                        },
                        source="runtime",
                    )
                    break

            elapsed = time.perf_counter() - start_time
            test_metrics = (
                Evaluator(self.model, self.device_ctx)
                .with_config(self.config.evaluation, logger=self.logger)
                .evaluate(
                    test_loader,
                    fs=data_bundle.sampling_rate_hz,
                )
                if test_loader is not None
                else {}
            )
            summary_metrics = {
                "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse,
                "best_metric_key": objective_metric_key,
                "best_metric_value": best_metric_value,
                "best_eval_metrics": best_eval_metrics,
                "final_train_loss": last_train_loss,
                "final_val_loss": last_val_loss,
                "training_seconds": elapsed,
                "test_metrics": test_metrics,
            }
            metrics_path = self.run_paths.metrics_path
            save_metrics(metrics_path, summary_metrics)
            self.training_hooks.register_artifact(
                run_id=run_id,
                path=metrics_path,
                artifact_type="training_metrics",
                label="training_metrics",
                metadata=summary_metrics,
                source="runtime",
            )
            self.training_hooks.register_artifact(
                run_id=run_id,
                path=self.history_path,
                artifact_type="history",
                label="history",
                source="runtime",
            )
            self._generate_summary_figures(
                run_id=run_id,
                val_loader=val_loader,
                test_loader=test_loader,
                sampling_rate_hz=data_bundle.sampling_rate_hz,
            )

            artifacts = TrainingArtifacts(
                run_dir=self.run_paths.root,
                checkpoint_dir=self.run_paths.checkpoints_dir,
                best_checkpoint=self.checkpoints.best_path,
                last_checkpoint=self.checkpoints.last_path,
                metrics_path=metrics_path,
                history_path=self.history_path,
                runtime_log_path=self.run_paths.runtime_log_path,
            )
            summary = TrainingSummary(
                run_id=run_id,
                best_epoch=best_epoch,
                best_val_rmse=best_val_rmse,
                best_metric_key=objective_metric_key,
                best_metric_value=best_metric_value,
                best_eval_metrics=best_eval_metrics,
                final_train_loss=last_train_loss,
                final_val_loss=last_val_loss,
                training_seconds=elapsed,
                artifacts=artifacts,
            )
            self.training_hooks.finish_run(
                run_id=run_id,
                status="completed",
                summary=summary_metrics,
                source="runtime",
            )
            return summary
        except Exception as exc:
            self.training_hooks.finish_run(
                run_id=run_id,
                status="failed",
                summary={"message": str(exc)},
                source="runtime",
            )
            raise
        finally:
            self.logger.removeHandler(log_handler)

    def _run_epoch(self, dataloader: DataLoader[Any], *, training: bool) -> float:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_batches = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in dataloader:
                self.training_control.check_abort()
                noisy, clean = self._unpack_batch(batch)
                noisy = noisy.to(self.device_ctx.device)
                clean = clean.to(self.device_ctx.device)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with self.device_ctx.autocast():
                    pred = self.model(noisy)
                    loss = self.loss_fn(pred, clean)

                if training:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        if self.config.training.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.training.gradient_clip,
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.autograd.backward(loss)
                        if self.config.training.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.training.gradient_clip,
                            )
                        self.optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def _objective_metric_value(
        self,
        *,
        metrics: dict[str, float],
        val_loss: float,
    ) -> float:
        metric_key = self.config.autoresearch.metric_key
        if metric_key == "final_val_loss":
            return float(val_loss)
        evaluation_metric_key = "rmse" if metric_key == "val_rmse" else metric_key
        metric_value = metrics.get(evaluation_metric_key)
        if metric_value is None:
            raise ValueError(
                "training requires evaluation.metrics to include "
                f"{evaluation_metric_key!r} when autoresearch.metric_key={metric_key!r}."
            )
        return float(metric_value)

    def _generate_summary_figures(
        self,
        *,
        run_id: str,
        val_loader: DataLoader[Any],
        test_loader: DataLoader[Any] | None,
        sampling_rate_hz: float,
    ) -> None:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        from imu_denoise.evaluation.visualization import (
            plot_denoising_comparison,
            plot_error_distribution,
            plot_psd,
            plot_training_curves,
        )

        self.run_paths.figures_dir.mkdir(parents=True, exist_ok=True)

        curves_path = self.run_paths.figures_dir / "training_curves.png"
        curves_fig = plot_training_curves(self.history_path, save_path=curves_path)
        plt.close(curves_fig)
        self.training_hooks.register_artifact(
            run_id=run_id,
            path=curves_path,
            artifact_type="figure",
            label="training_curves",
            source="runtime",
        )

        sample_batch = self._first_batch(test_loader) or self._first_batch(val_loader)
        if sample_batch is None:
            return

        noisy_tensor, clean_tensor = self._unpack_batch(sample_batch)
        timestamps_tensor = (
            sample_batch.get("timestamps") if isinstance(sample_batch, dict) else None
        )
        noisy = noisy_tensor.to(self.device_ctx.device)
        self.model.eval()
        with torch.no_grad():
            denoised = self.model(noisy).detach().cpu().float().numpy()

        noisy_np = noisy.detach().cpu().float().numpy()
        clean_np = clean_tensor.detach().cpu().float().numpy()
        timestamps = self._timestamps_for_batch(
            timestamps_tensor=timestamps_tensor,
            sample_count=noisy_np.shape[1],
            sampling_rate_hz=sampling_rate_hz,
        )
        sample_index = 0

        comparison_path = self.run_paths.figures_dir / "denoising_comparison.png"
        comparison_fig = plot_denoising_comparison(
            noisy=noisy_np[sample_index],
            denoised=denoised[sample_index],
            clean=clean_np[sample_index],
            timestamps=timestamps[sample_index],
            title=f"{self.config.name} denoising comparison",
            save_path=comparison_path,
        )
        plt.close(comparison_fig)
        self.training_hooks.register_artifact(
            run_id=run_id,
            path=comparison_path,
            artifact_type="figure",
            label="denoising_comparison",
            source="runtime",
        )

        psd_path = self.run_paths.figures_dir / "psd.png"
        psd_fig = plot_psd(
            signals={
                "noisy": noisy_np[sample_index],
                "denoised": denoised[sample_index],
                "clean": clean_np[sample_index],
            },
            fs=sampling_rate_hz,
            title=f"{self.config.name} PSD",
            save_path=psd_path,
        )
        plt.close(psd_fig)
        self.training_hooks.register_artifact(
            run_id=run_id,
            path=psd_path,
            artifact_type="figure",
            label="psd",
            source="runtime",
        )

        errors_path = self.run_paths.figures_dir / "error_distribution.png"
        errors_fig = plot_error_distribution(
            errors=(denoised[sample_index] - clean_np[sample_index]),
            title=f"{self.config.name} error distribution",
            save_path=errors_path,
        )
        plt.close(errors_fig)
        self.training_hooks.register_artifact(
            run_id=run_id,
            path=errors_path,
            artifact_type="figure",
            label="error_distribution",
            source="runtime",
        )

    @staticmethod
    def _first_batch(dataloader: DataLoader[Any] | None) -> dict[str, Any] | tuple[Any, ...] | None:
        if dataloader is None:
            return None
        iterator = iter(dataloader)
        try:
            return cast(dict[str, Any] | tuple[Any, ...], next(iterator))
        except StopIteration:
            return None

    def _timestamps_for_batch(
        self,
        *,
        timestamps_tensor: torch.Tensor | None,
        sample_count: int,
        sampling_rate_hz: float,
    ) -> np.ndarray:
        if timestamps_tensor is not None:
            return timestamps_tensor.detach().cpu().float().numpy()
        dt = 0.0 if sampling_rate_hz <= 0.0 else 1.0 / sampling_rate_hz
        return np.tile(np.arange(sample_count, dtype=np.float32) * dt, (1, 1))

    def _should_run_full_evaluation(self, epoch: int) -> bool:
        frequency = max(1, int(self.config.evaluation.frequency_epochs))
        return epoch == self.config.training.epochs or epoch % frequency == 0

    def _step_scheduler(self, val_loss: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
            return
        self.scheduler.step()

    def _write_history(self, record: dict[str, Any]) -> None:
        with open(self.history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    @staticmethod
    def _unpack_batch(batch: dict[str, Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch["noisy"], batch["clean"]
        return batch[0], batch[1]
