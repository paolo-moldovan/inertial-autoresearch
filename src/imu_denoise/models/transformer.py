"""Transformer encoder denoiser with sinusoidal positional encoding."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor

from imu_denoise.models.base import BaseDenoiser
from imu_denoise.models.components.positional import SinusoidalPositionalEncoding
from imu_denoise.models.registry import register_model

IMU_CHANNELS = 6


@register_model("transformer")
class TransformerDenoiser(BaseDenoiser):
    """Transformer encoder-based IMU denoiser.

    Architecture:
        - Linear input projection (6 -> hidden_dim)
        - Sinusoidal positional encoding
        - Stack of TransformerEncoderLayers
        - Linear output projection (hidden_dim -> 6)
        - Residual connection: output = input + correction

    Args:
        hidden_dim: Model embedding dimension.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        max_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(IMU_CHANNELS, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=hidden_dim,
            max_len=max_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_dim, IMU_CHANNELS)

    def forward(self, noisy_imu: Tensor, timestamps: Tensor | None = None) -> Tensor:
        """Denoise IMU signal using transformer encoder with residual connection.

        Args:
            noisy_imu: Noisy IMU readings of shape (batch, seq_len, 6).
            timestamps: Unused. Accepted for interface compatibility.

        Returns:
            Denoised signal of shape (batch, seq_len, 6).
        """
        x = self.input_proj(noisy_imu)
        x = self.pos_encoding(x)
        x = cast(Tensor, self.transformer_encoder(x))
        correction = cast(Tensor, self.output_proj(x))
        return noisy_imu + correction
