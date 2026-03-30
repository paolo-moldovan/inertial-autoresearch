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


def smoothness_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Compare second-order temporal differences between prediction and target."""
    if len(pred) < 3 or len(target) < 3:
        return 0.0
    pred_second = np.diff(pred, n=2, axis=0)
    target_second = np.diff(target, n=2, axis=0)
    return float(np.sqrt(np.mean((pred_second - target_second) ** 2)))


def drift_endpoint_error(
    pred: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
) -> float:
    """Proxy downstream drift by comparing integrated endpoint error per channel."""
    if len(pred) < 2 or len(target) < 2 or len(timestamps) < 2:
        return 0.0
    dt = np.diff(timestamps).reshape(-1, 1)
    pred_integral = np.cumsum(0.5 * (pred[1:] + pred[:-1]) * dt, axis=0)
    target_integral = np.cumsum(0.5 * (target[1:] + target[:-1]) * dt, axis=0)
    endpoint_error = pred_integral[-1] - target_integral[-1]
    return float(np.sqrt(np.mean(endpoint_error**2)))


def compute_selected_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    fs: float = 200.0,
    metric_names: list[str] | tuple[str, ...] | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Compute a selected metric subset, optionally prefixed for scoped reporting."""
    requested = list(metric_names or ["rmse", "mae", "spectral_divergence"])
    results: dict[str, float] = {}

    if "rmse" in requested:
        results[f"{prefix}rmse"] = rmse(pred, target)
        per_axis = rmse_per_axis(pred, target)
        for index, value in enumerate(per_axis):
            results[f"{prefix}rmse_axis_{index}"] = float(value)
    if "mae" in requested:
        results[f"{prefix}mae"] = mae(pred, target)
    if "spectral_divergence" in requested:
        results[f"{prefix}spectral_divergence"] = spectral_divergence(pred, target, fs)

    return results


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
    return compute_selected_metrics(
        pred,
        target,
        fs=fs,
        metric_names=["rmse", "mae", "spectral_divergence"],
    )
