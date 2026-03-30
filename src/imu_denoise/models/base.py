"""Abstract base class for all IMU denoising models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseDenoiser(nn.Module, ABC):
    """Base class for IMU denoising models.

    All denoisers take (batch, seq_len, 6) input — 3 accelerometer + 3 gyroscope
    channels — and produce (batch, seq_len, 6) denoised output.
    """
    causal: bool = False

    @abstractmethod
    def forward(self, noisy_imu: Tensor, timestamps: Tensor | None = None) -> Tensor:
        """Denoise an IMU signal.

        Args:
            noisy_imu: Noisy IMU readings of shape (batch, seq_len, 6).
            timestamps: Optional timestamps of shape (batch, seq_len).

        Returns:
            Denoised IMU signal of shape (batch, seq_len, 6).
        """

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute training loss. Override for custom losses.

        Args:
            pred: Predicted denoised signal of shape (batch, seq_len, 6).
            target: Ground truth clean signal of shape (batch, seq_len, 6).

        Returns:
            Scalar loss tensor.
        """
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def predict(self, noisy_imu: Tensor) -> Tensor:
        """Run inference in eval mode without gradient tracking.

        Args:
            noisy_imu: Noisy IMU readings of shape (batch, seq_len, 6).

        Returns:
            Denoised IMU signal of shape (batch, seq_len, 6).
        """
        self.eval()
        return self.forward(noisy_imu)
