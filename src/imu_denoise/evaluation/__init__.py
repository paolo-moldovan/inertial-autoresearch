"""Evaluation utilities for IMU denoising models.

This package avoids eager imports so training paths do not pull in plotting
dependencies unless they are explicitly needed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Evaluator",
    "compute_all_metrics",
    "mae",
    "rmse",
    "rmse_per_axis",
    "spectral_divergence",
    "plot_denoising_comparison",
    "plot_error_distribution",
    "plot_psd",
    "plot_training_curves",
]


def __getattr__(name: str) -> Any:
    if name == "Evaluator":
        from imu_denoise.evaluation.evaluator import Evaluator

        return Evaluator
    if name in {"compute_all_metrics", "mae", "rmse", "rmse_per_axis", "spectral_divergence"}:
        from imu_denoise import evaluation as _evaluation  # pragma: no cover

        del _evaluation
        from imu_denoise.evaluation.metrics import (
            compute_all_metrics,
            mae,
            rmse,
            rmse_per_axis,
            spectral_divergence,
        )

        return {
            "compute_all_metrics": compute_all_metrics,
            "mae": mae,
            "rmse": rmse,
            "rmse_per_axis": rmse_per_axis,
            "spectral_divergence": spectral_divergence,
        }[name]
    if name in {
        "plot_denoising_comparison",
        "plot_error_distribution",
        "plot_psd",
        "plot_training_curves",
    }:
        from imu_denoise.evaluation.visualization import (
            plot_denoising_comparison,
            plot_error_distribution,
            plot_psd,
            plot_training_curves,
        )

        return {
            "plot_denoising_comparison": plot_denoising_comparison,
            "plot_error_distribution": plot_error_distribution,
            "plot_psd": plot_psd,
            "plot_training_curves": plot_training_curves,
        }[name]
    raise AttributeError(name)
