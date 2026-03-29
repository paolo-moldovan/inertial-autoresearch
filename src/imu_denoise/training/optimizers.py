"""Optimizer and scheduler factories."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from imu_denoise.config.schema import TrainingConfig


def _build_optimizer(parameters: Any, config: TrainingConfig) -> Optimizer:
    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _build_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
) -> LRScheduler | ReduceLROnPlateau | None:
    name = config.scheduler.lower()
    kwargs = dict(config.scheduler_kwargs)

    if name == "none":
        return None
    if name == "cosine":
        t_max_raw = kwargs.pop("T_max", max(1, config.epochs))
        eta_min_raw = kwargs.pop("eta_min", 0.0)
        t_max = int(cast(Any, t_max_raw))
        eta_min = float(cast(Any, eta_min_raw))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
    if name == "step":
        step_size_raw = kwargs.pop("step_size", max(1, config.epochs // 3))
        gamma_raw = kwargs.pop("gamma", 0.1)
        step_size = int(cast(Any, step_size_raw))
        gamma = float(cast(Any, gamma_raw))
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    if name == "plateau":
        factor_raw = kwargs.pop("factor", 0.5)
        patience_raw = kwargs.pop("patience", max(1, config.early_stop_patience // 2))
        factor = float(cast(Any, factor_raw))
        patience = int(cast(Any, patience_raw))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=factor,
            patience=patience,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def build_optimizer_and_scheduler(
    parameters: Any,
    config: TrainingConfig,
) -> tuple[Optimizer, LRScheduler | ReduceLROnPlateau | None]:
    """Build the optimizer and optional scheduler from training config."""
    optimizer = _build_optimizer(parameters, config)
    scheduler = _build_scheduler(optimizer, config)
    return optimizer, scheduler
