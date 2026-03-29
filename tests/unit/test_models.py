"""Tests for model registration and basic forward-pass contracts."""

from __future__ import annotations

import torch

from imu_denoise.models import get_model, list_models


def test_builtin_models_are_registered() -> None:
    """The built-in model zoo should be discoverable through the registry."""
    assert list_models() == ["conv1d", "lstm", "transformer"]


def test_lstm_forward_shape(synthetic_batch: torch.Tensor) -> None:
    """LSTM denoiser should preserve the canonical IMU tensor shape."""
    model = get_model("lstm")
    output = model(synthetic_batch)
    assert output.shape == synthetic_batch.shape


def test_conv1d_forward_shape(synthetic_batch: torch.Tensor) -> None:
    """Conv1D denoiser should preserve the canonical IMU tensor shape."""
    model = get_model("conv1d")
    output = model(synthetic_batch)
    assert output.shape == synthetic_batch.shape


def test_transformer_forward_shape(synthetic_batch: torch.Tensor) -> None:
    """Transformer denoiser should preserve the canonical IMU tensor shape."""
    model = get_model("transformer")
    output = model(synthetic_batch)
    assert output.shape == synthetic_batch.shape
