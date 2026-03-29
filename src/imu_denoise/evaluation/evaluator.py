"""Full evaluation pipeline for IMU denoising models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from imu_denoise.evaluation.metrics import compute_all_metrics
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
        self.model = model
        self.device_ctx = device_ctx
        self.model = device_ctx.to_device(self.model)

    def evaluate(self, dataloader: DataLoader, fs: float = 200.0) -> dict[str, float]:
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

        with torch.no_grad():
            for batch in dataloader:
                noisy, clean = self._unpack_batch(batch)
                noisy = noisy.to(self.device_ctx.device)

                with self.device_ctx.autocast():
                    pred = self.model(noisy)

                all_preds.append(pred.cpu().float().numpy())
                all_targets.append(clean.numpy())

        preds = np.concatenate(all_preds, axis=0).reshape(-1, all_preds[0].shape[-1])
        targets = np.concatenate(all_targets, axis=0).reshape(-1, all_targets[0].shape[-1])

        return compute_all_metrics(preds, targets, fs=fs)

    def evaluate_from_checkpoint(
        self,
        checkpoint_path: Path,
        dataloader: DataLoader,
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
    def _unpack_batch(batch: dict | tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract noisy and clean tensors from a batch.

        Supports both dict-style (``{"noisy": ..., "clean": ...}``) and
        tuple-style ``(noisy, clean)`` batches.
        """
        if isinstance(batch, dict):
            return batch["noisy"], batch["clean"]
        return batch[0], batch[1]
