"""Loss function factory for denoising training."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

LossFn = Callable[[Tensor, Tensor], Tensor]


def spectral_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute a frequency-domain reconstruction loss."""
    pred_fft = torch.fft.rfft(pred.float(), dim=1)
    target_fft = torch.fft.rfft(target.float(), dim=1)
    pred_power = pred_fft.abs().pow(2)
    target_power = target_fft.abs().pow(2)
    return F.l1_loss(torch.log1p(pred_power), torch.log1p(target_power))


def build_loss(name: str) -> LossFn:
    """Create a loss function from a training config name."""
    normalized = name.lower()
    if normalized == "mse":
        return cast(LossFn, F.mse_loss)
    if normalized == "huber":
        return cast(LossFn, F.huber_loss)
    if normalized == "spectral":
        return spectral_loss
    raise ValueError(f"Unsupported loss: {name}")
