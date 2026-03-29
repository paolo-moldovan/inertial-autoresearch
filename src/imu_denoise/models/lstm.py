"""Bidirectional LSTM denoiser with residual connection."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor

from imu_denoise.models.base import BaseDenoiser
from imu_denoise.models.registry import register_model

IMU_CHANNELS = 6


@register_model("lstm")
class LSTMDenoiser(BaseDenoiser):
    """LSTM-based IMU denoiser.

    Architecture:
        - Linear input projection (6 -> hidden_dim)
        - Bidirectional multi-layer LSTM
        - Linear output projection (hidden_dim*2 -> 6)
        - Residual connection: output = input + correction

    Args:
        hidden_dim: LSTM hidden state size.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers (only applied when num_layers > 1).
        bidirectional: Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(IMU_CHANNELS, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(output_dim, IMU_CHANNELS)

    def forward(self, noisy_imu: Tensor, timestamps: Tensor | None = None) -> Tensor:
        """Denoise IMU signal using LSTM with residual connection.

        Args:
            noisy_imu: Noisy IMU readings of shape (batch, seq_len, 6).
            timestamps: Unused. Accepted for interface compatibility.

        Returns:
            Denoised signal of shape (batch, seq_len, 6).
        """
        x = self.input_proj(noisy_imu)
        x, _ = self.lstm(x)
        correction = cast(Tensor, self.output_proj(x))
        return noisy_imu + correction
