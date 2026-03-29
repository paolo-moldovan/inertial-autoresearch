"""Dilated causal 1D CNN denoiser with residual connection."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor

from imu_denoise.models.base import BaseDenoiser
from imu_denoise.models.registry import register_model

IMU_CHANNELS = 6


class _CausalConvBlock(nn.Module):
    """Single causal dilated convolution block with GELU and LayerNorm."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on the left, 0 on the right
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We handle padding manually for causal behavior
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, channels, seq_len).

        Returns:
            Output of shape (batch, channels, seq_len).
        """
        # Causal padding: pad only on the left
        residual = x
        x = nn.functional.pad(x, (self.causal_pad, 0))
        x = self.conv(x)
        x = self.activation(x)
        # LayerNorm expects (batch, seq_len, channels), so transpose
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return x + residual


@register_model("conv1d")
class Conv1DDenoiser(BaseDenoiser):
    """Dilated causal 1D CNN denoiser.

    Architecture:
        - 1x1 Conv input projection (6 -> hidden_dim)
        - Stack of dilated causal conv blocks with exponentially growing dilation
        - 1x1 Conv output projection (hidden_dim -> 6)
        - Residual connection: output = input + correction

    Args:
        hidden_dim: Number of channels in hidden layers.
        num_layers: Number of dilated convolution blocks.
        kernel_size: Convolution kernel size.
        dilation_base: Base for exponential dilation growth (dilation = base^i).
        dropout: Dropout rate within conv blocks.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 6,
        kernel_size: int = 7,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(IMU_CHANNELS, hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                _CausalConvBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation_base**i,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.output_proj = nn.Conv1d(hidden_dim, IMU_CHANNELS, kernel_size=1)

    def forward(self, noisy_imu: Tensor, timestamps: Tensor | None = None) -> Tensor:
        """Denoise IMU signal using dilated causal CNN with residual connection.

        Args:
            noisy_imu: Noisy IMU readings of shape (batch, seq_len, 6).
            timestamps: Unused. Accepted for interface compatibility.

        Returns:
            Denoised signal of shape (batch, seq_len, 6).
        """
        # Conv1d expects (batch, channels, seq_len)
        x = noisy_imu.transpose(1, 2)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        correction = cast(Tensor, self.output_proj(x))
        # Back to (batch, seq_len, channels)
        correction = correction.transpose(1, 2)
        return noisy_imu + correction
