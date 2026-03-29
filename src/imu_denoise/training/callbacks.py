"""Small callback-style helpers for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import Optimizer

from imu_denoise.utils.io import save_checkpoint


@dataclass
class EarlyStopping:
    """Track validation metric improvements and decide when to stop."""

    patience: int
    mode: str = "min"
    best_value: float | None = None
    num_bad_epochs: int = 0

    def update(self, value: float) -> bool:
        """Return ``True`` when training should stop."""
        if self.best_value is None or self._is_improvement(value):
            self.best_value = value
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

    def _is_improvement(self, value: float) -> bool:
        assert self.best_value is not None
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value


@dataclass
class CheckpointManager:
    """Save last and best checkpoints under an experiment directory."""

    checkpoint_dir: Path
    monitor: str = "val_rmse"
    mode: str = "min"
    best_value: float | None = None

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    def save(
        self,
        *,
        epoch: int,
        metric_value: float,
        model: nn.Module,
        optimizer: Optimizer,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Save checkpoints and return whether this epoch produced a new best."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        save_checkpoint(
            self.last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=metric_value,
            extra=extra,
        )

        is_best = self.best_value is None or self._is_improvement(metric_value)
        if is_best:
            self.best_value = metric_value
            save_checkpoint(
                self.best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=metric_value,
                extra=extra,
            )
        return is_best

    def _is_improvement(self, value: float) -> bool:
        assert self.best_value is not None
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value
