"""Full evaluation pipeline for IMU denoising models."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from imu_denoise.config.schema import EvaluationConfig
from imu_denoise.evaluation.metrics import (
    compute_selected_metrics,
    drift_endpoint_error,
    smoothness_error,
)
from imu_denoise.evaluation.reconstruction import reconstruct_window_predictions
from imu_denoise.utils.io import load_checkpoint

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from imu_denoise.config.schema import ModelConfig
    from imu_denoise.device.context import DeviceContext
    from imu_denoise.models.base import BaseDenoiser


class Evaluator:
    """Run evaluation on a denoising model and aggregate metrics.

    Args:
        model: A :class:`BaseDenoiser` instance.
        device_ctx: A :class:`DeviceContext` for device placement and AMP.
    """

    def __init__(self, model: BaseDenoiser, device_ctx: DeviceContext) -> None:
        self.evaluation = EvaluationConfig()
        self.logger = None
        self.model = model
        self.device_ctx = device_ctx
        self.model = device_ctx.to_device(self.model)

    def with_config(
        self,
        evaluation: EvaluationConfig,
        *,
        logger: Any | None = None,
    ) -> Evaluator:
        """Attach evaluation settings without changing existing call sites."""
        self.evaluation = evaluation
        self.logger = logger
        return self

    def evaluate(self, dataloader: DataLoader[Any], fs: float = 200.0) -> dict[str, float]:
        """Evaluate the model on an entire dataloader and compute aggregate metrics.

        Expects each batch to yield a dictionary (or tuple) with keys/positions:
        ``"noisy"`` (batch, seq_len, 6) and ``"clean"`` (batch, seq_len, 6).

        Args:
            dataloader: DataLoader yielding batches of noisy/clean IMU pairs.
            fs: Sampling frequency in Hz for spectral metrics.

        Returns:
            Dictionary of aggregated evaluation metrics.
        """
        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_timestamps: list[np.ndarray] = []
        sequence_ids: list[str] = []

        if self.evaluation.realtime_mode and not bool(getattr(self.model, "causal", False)):
            warnings.warn(
                (
                    f"Model {self.model.__class__.__name__} is non-causal but "
                    "evaluation.realtime_mode is enabled."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        with torch.no_grad():
            for batch in dataloader:
                noisy, clean = self._unpack_batch(batch)
                noisy = noisy.to(self.device_ctx.device)

                with self.device_ctx.autocast():
                    pred = self.model(noisy)

                all_preds.append(pred.cpu().float().numpy())
                all_targets.append(clean.numpy())
                if isinstance(batch, dict):
                    all_timestamps.append(batch["timestamps"].cpu().float().numpy())
                    sequence_ids.extend([str(item) for item in batch["sequence_id"]])

        pred_windows = np.concatenate(all_preds, axis=0)
        target_windows = np.concatenate(all_targets, axis=0)
        timestamps = np.concatenate(all_timestamps, axis=0) if all_timestamps else np.empty((0, 0))
        return evaluate_window_predictions(
            pred_windows=pred_windows,
            target_windows=target_windows,
            timestamps=timestamps,
            sequence_ids=sequence_ids,
            fs=fs,
            evaluation=self.evaluation,
        )

    def evaluate_from_checkpoint(
        self,
        checkpoint_path: Path,
        dataloader: DataLoader[Any],
        model_config: ModelConfig,
        fs: float = 200.0,
    ) -> dict[str, float]:
        """Load a checkpoint and evaluate it.

        Args:
            checkpoint_path: Path to the saved checkpoint.
            dataloader: DataLoader for evaluation data.
            model_config: Model configuration (unused here but available for
                reconstruction if needed by subclasses).
            fs: Sampling frequency in Hz.

        Returns:
            Dictionary of evaluation metrics.
        """
        load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            device=self.device_ctx.device,
        )
        return self.evaluate(dataloader, fs=fs)

    @staticmethod
    def _unpack_batch(
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract noisy and clean tensors from a batch.

        Supports both dict-style (``{"noisy": ..., "clean": ...}``) and
        tuple-style ``(noisy, clean)`` batches.
        """
        if isinstance(batch, dict):
            return batch["noisy"], batch["clean"]
        return batch[0], batch[1]


def evaluate_window_predictions(
    *,
    pred_windows: np.ndarray,
    target_windows: np.ndarray,
    timestamps: np.ndarray,
    sequence_ids: list[str],
    fs: float,
    evaluation: EvaluationConfig,
) -> dict[str, float]:
    """Evaluate window predictions with optional sequence reconstruction."""
    selected = set(evaluation.metrics)
    window_metric_names = {
        metric for metric in selected if metric in {"rmse", "mae", "spectral_divergence"}
    }
    sequence_metric_names = {
        metric.removeprefix("sequence_")
        for metric in selected
        if metric.startswith("sequence_")
        and metric.removeprefix("sequence_") in {"rmse", "mae", "spectral_divergence"}
    }
    results: dict[str, float] = {}

    flattened_pred = pred_windows.reshape(-1, pred_windows.shape[-1])
    flattened_target = target_windows.reshape(-1, target_windows.shape[-1])
    if window_metric_names:
        results.update(
            compute_selected_metrics(
                flattened_pred,
                flattened_target,
                fs=fs,
                metric_names=sorted(window_metric_names),
            )
        )

    needs_sequence = bool(sequence_metric_names or {"smoothness", "drift_error"} & selected)
    if not needs_sequence:
        return results
    if evaluation.reconstruction == "none":
        raise ValueError(
            "Sequence-level metrics require evaluation.reconstruction to be set to a supported "
            "overlap-add mode."
        )

    reconstructed = reconstruct_window_predictions(
        pred_windows=pred_windows,
        target_windows=target_windows,
        timestamps=timestamps,
        sequence_ids=sequence_ids,
        mode=evaluation.reconstruction,
    )
    if sequence_metric_names:
        ordered_pred = np.concatenate([item["pred"] for item in reconstructed.values()], axis=0)
        ordered_target = np.concatenate([item["target"] for item in reconstructed.values()], axis=0)
        results.update(
            compute_selected_metrics(
                ordered_pred,
                ordered_target,
                fs=fs,
                metric_names=sorted(sequence_metric_names),
                prefix="sequence_",
            )
        )
    if "smoothness" in selected:
        values = [
            smoothness_error(item["pred"], item["target"])
            for item in reconstructed.values()
        ]
        results["smoothness"] = float(np.mean(values)) if values else 0.0
    if "drift_error" in selected:
        values = [
            drift_endpoint_error(item["pred"], item["target"], item["timestamps"])
            for item in reconstructed.values()
        ]
        results["drift_error"] = float(np.mean(values)) if values else 0.0
    return results
