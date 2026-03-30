"""Loss function factory for denoising training."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from imu_denoise.config.schema import TrainingConfig

LossFn = Callable[[Tensor, Tensor], Tensor]


def _resolved_channel_weights(config: TrainingConfig) -> Tensor | None:
    if config.channel_loss_weights:
        if len(config.channel_loss_weights) != 6:
            raise ValueError("training.channel_loss_weights must contain exactly 6 values.")
        return torch.tensor(config.channel_loss_weights, dtype=torch.float32)
    if config.accel_loss_weight != 1.0 or config.gyro_loss_weight != 1.0:
        return torch.tensor(
            [config.accel_loss_weight] * 3 + [config.gyro_loss_weight] * 3,
            dtype=torch.float32,
        )
    return None


def _apply_channel_weights(loss: Tensor, channel_weights: Tensor | None) -> Tensor:
    if channel_weights is None:
        return loss.mean()
    weights = channel_weights.to(loss.device, dtype=loss.dtype)
    weighted = loss * weights.view(*([1] * (loss.ndim - 1)), -1)
    return weighted.mean()


def spectral_loss(pred: Tensor, target: Tensor, *, channel_weights: Tensor | None = None) -> Tensor:
    """Compute a frequency-domain reconstruction loss."""
    pred_fft = torch.fft.rfft(pred.float(), dim=1)
    target_fft = torch.fft.rfft(target.float(), dim=1)
    pred_power = pred_fft.abs().pow(2)
    target_power = target_fft.abs().pow(2)
    loss = F.l1_loss(torch.log1p(pred_power), torch.log1p(target_power), reduction="none")
    return _apply_channel_weights(loss, channel_weights)


def build_loss(config_or_name: TrainingConfig | str) -> LossFn:
    """Create a loss function from a training config name."""
    if isinstance(config_or_name, TrainingConfig):
        normalized = config_or_name.loss.lower()
        channel_weights = _resolved_channel_weights(config_or_name)
    else:
        normalized = config_or_name.lower()
        channel_weights = None

    if normalized == "mse":
        return cast(
            LossFn,
            lambda pred, target: _apply_channel_weights(
                F.mse_loss(pred, target, reduction="none"),
                channel_weights,
            ),
        )
    if normalized == "huber":
        return cast(
            LossFn,
            lambda pred, target: _apply_channel_weights(
                F.huber_loss(pred, target, reduction="none"),
                channel_weights,
            ),
        )
    if normalized == "spectral":
        return cast(
            LossFn,
            lambda pred, target: spectral_loss(
                pred,
                target,
                channel_weights=channel_weights,
            ),
        )
    raise ValueError(f"Unsupported loss: {normalized}")
