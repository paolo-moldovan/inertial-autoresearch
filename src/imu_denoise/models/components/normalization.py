"""RMSNorm implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs by their RMS value, without centering (no mean subtraction).
    Often faster and comparably effective to standard LayerNorm.

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor where the last dimension has size ``dim``.

        Returns:
            Normalized tensor, same shape as input.
        """
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
