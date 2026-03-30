"""Compatibility wrapper for training hook builders."""

from __future__ import annotations

from imu_denoise.observability.training_hooks import (
    MissionControlTrainingControl,
    MissionControlTrainingHooks,
    build_training_control,
    build_training_hooks,
)

__all__ = [
    "MissionControlTrainingControl",
    "MissionControlTrainingHooks",
    "build_training_control",
    "build_training_hooks",
]
