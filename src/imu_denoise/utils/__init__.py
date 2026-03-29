"""Utility functions for IMU denoising experiments.

This package keeps imports lazy so callers can use light-weight helpers without
pulling unrelated submodules into every process.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "load_checkpoint",
    "load_metrics",
    "save_checkpoint",
    "save_metrics",
    "setup_logger",
    "get_gpu_memory_mb",
    "log_system_info",
    "timed",
    "quat_conjugate",
    "quat_multiply",
    "quat_to_angular_velocity",
    "quat_to_rotation_matrix",
    "rotate_vector",
]


def __getattr__(name: str) -> Any:
    if name in {"load_checkpoint", "load_metrics", "save_checkpoint", "save_metrics"}:
        from imu_denoise.utils.io import (
            load_checkpoint,
            load_metrics,
            save_checkpoint,
            save_metrics,
        )

        return {
            "load_checkpoint": load_checkpoint,
            "load_metrics": load_metrics,
            "save_checkpoint": save_checkpoint,
            "save_metrics": save_metrics,
        }[name]
    if name in {"setup_logger"}:
        from imu_denoise.utils.logging import setup_logger

        return setup_logger
    if name in {"get_gpu_memory_mb", "log_system_info", "timed"}:
        from imu_denoise.utils.profiling import get_gpu_memory_mb, log_system_info, timed

        return {
            "get_gpu_memory_mb": get_gpu_memory_mb,
            "log_system_info": log_system_info,
            "timed": timed,
        }[name]
    if name in {
        "quat_conjugate",
        "quat_multiply",
        "quat_to_angular_velocity",
        "quat_to_rotation_matrix",
        "rotate_vector",
    }:
        from imu_denoise.utils.quaternion import (
            quat_conjugate,
            quat_multiply,
            quat_to_angular_velocity,
            quat_to_rotation_matrix,
            rotate_vector,
        )

        return {
            "quat_conjugate": quat_conjugate,
            "quat_multiply": quat_multiply,
            "quat_to_angular_velocity": quat_to_angular_velocity,
            "quat_to_rotation_matrix": quat_to_rotation_matrix,
            "rotate_vector": rotate_vector,
        }[name]
    raise AttributeError(name)
