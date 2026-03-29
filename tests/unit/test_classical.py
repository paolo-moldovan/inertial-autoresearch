"""Tests for classical IMU denoising baselines."""

from __future__ import annotations

import numpy as np

from imu_denoise.classical import ComplementaryFilterBaseline, KalmanFilterBaseline


def _make_noisy_constant_batch() -> np.ndarray:
    rng = np.random.default_rng(7)
    clean = np.ones((2, 32, 6), dtype=np.float32)
    noise = rng.normal(0.0, 0.2, size=clean.shape).astype(np.float32)
    return clean + noise


def test_kalman_baseline_preserves_shape() -> None:
    """Kalman baseline should preserve batch/window/channel dimensions."""
    noisy = _make_noisy_constant_batch()
    baseline = KalmanFilterBaseline()

    denoised = baseline.denoise(noisy)

    assert denoised.shape == noisy.shape
    assert np.isfinite(denoised).all()


def test_complementary_baseline_preserves_shape() -> None:
    """Complementary baseline should preserve batch/window/channel dimensions."""
    noisy = _make_noisy_constant_batch()
    baseline = ComplementaryFilterBaseline()

    denoised = baseline.denoise(noisy)

    assert denoised.shape == noisy.shape
    assert np.isfinite(denoised).all()


def test_kalman_baseline_reduces_noise_energy() -> None:
    """Kalman smoothing should reduce MSE on a constant-signal toy problem."""
    noisy = _make_noisy_constant_batch()
    clean = np.ones_like(noisy)
    baseline = KalmanFilterBaseline()

    denoised = baseline.denoise(noisy)

    noisy_mse = np.mean((noisy - clean) ** 2)
    denoised_mse = np.mean((denoised - clean) ** 2)
    assert denoised_mse < noisy_mse
