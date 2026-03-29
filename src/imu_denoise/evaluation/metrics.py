"""Evaluation metrics for IMU denoising quality assessment.

All metric functions accept arrays of shape ``(N, C)`` where ``C`` is typically
3 (single sensor) or 6 (accelerometer + gyroscope).
"""

from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error across all elements.

    Args:
        pred: Predicted signal of shape ``(N, C)``.
        target: Ground truth signal of shape ``(N, C)``.

    Returns:
        Scalar RMSE value.
    """
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error across all elements.

    Args:
        pred: Predicted signal of shape ``(N, C)``.
        target: Ground truth signal of shape ``(N, C)``.

    Returns:
        Scalar MAE value.
    """
    return float(np.mean(np.abs(pred - target)))


def rmse_per_axis(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Root mean squared error computed independently per channel.

    Args:
        pred: Predicted signal of shape ``(N, C)``.
        target: Ground truth signal of shape ``(N, C)``.

    Returns:
        Array of shape ``(C,)`` with per-axis RMSE.
    """
    return np.asarray(np.sqrt(np.mean((pred - target) ** 2, axis=0)))


def spectral_divergence(pred: np.ndarray, target: np.ndarray, fs: float) -> float:
    """Log spectral distance between predicted and target signals.

    Computes the average log-ratio of power spectral densities across all channels
    using the FFT. A lower value indicates better spectral fidelity.

    Args:
        pred: Predicted signal of shape ``(N, C)``.
        target: Ground truth signal of shape ``(N, C)``.
        fs: Sampling frequency in Hz.

    Returns:
        Mean log spectral distance (non-negative scalar).
    """
    n = pred.shape[0]
    eps = 1e-10

    # Compute one-sided power spectra
    pred_fft = np.fft.rfft(pred, axis=0)
    target_fft = np.fft.rfft(target, axis=0)

    pred_psd = np.abs(pred_fft) ** 2 / (fs * n)
    target_psd = np.abs(target_fft) ** 2 / (fs * n)

    # Log spectral distance: mean of |log(P_pred / P_target)|
    log_ratio = np.abs(np.log10(pred_psd + eps) - np.log10(target_psd + eps))
    return float(np.mean(log_ratio))


def compute_all_metrics(
    pred: np.ndarray, target: np.ndarray, fs: float = 200.0
) -> dict[str, float]:
    """Compute the full suite of evaluation metrics.

    Args:
        pred: Predicted signal of shape ``(N, C)``.
        target: Ground truth signal of shape ``(N, C)``.
        fs: Sampling frequency in Hz (default 200 for EuRoC).

    Returns:
        Dictionary mapping metric names to scalar values. Per-axis RMSE values
        are stored as ``rmse_axis_0``, ``rmse_axis_1``, etc.
    """
    per_axis = rmse_per_axis(pred, target)

    metrics: dict[str, float] = {
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "spectral_divergence": spectral_divergence(pred, target, fs),
    }

    for i, val in enumerate(per_axis):
        metrics[f"rmse_axis_{i}"] = float(val)

    return metrics
