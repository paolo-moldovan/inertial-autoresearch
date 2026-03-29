"""Tests for device selection and runtime context behavior."""

from __future__ import annotations

import pytest
import torch

from imu_denoise.config import DeviceConfig
from imu_denoise.device.context import DeviceContext
from imu_denoise.device.detect import detect_device


def test_detect_device_prefers_cpu_when_no_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto device detection should fall back to CPU when CUDA and MPS are unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert detect_device().type == "cpu"


def test_detect_device_rejects_unavailable_requested_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly requested unavailable backend should fail loudly."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        detect_device("cuda")


def test_device_context_disables_amp_for_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """MPS should downgrade unsupported dtypes and disable AMP."""
    monkeypatch.setattr(
        "imu_denoise.device.context.detect_device",
        lambda preferred: torch.device("mps"),
    )

    context = DeviceContext.from_config(DeviceConfig(preferred="auto", dtype="bfloat16"))

    assert context.device.type == "mps"
    assert context.dtype == torch.float32
    assert context.amp_enabled is False
    assert context.pin_memory is False
    assert context.supports_compile is False


def test_device_context_enables_amp_for_cuda_float16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA float16 should enable AMP and pinned-memory dataloaders."""
    monkeypatch.setattr(
        "imu_denoise.device.context.detect_device",
        lambda preferred: torch.device("cuda"),
    )

    context = DeviceContext.from_config(DeviceConfig(preferred="auto", dtype="float16"))

    assert context.device.type == "cuda"
    assert context.dtype == torch.float16
    assert context.amp_enabled is True
    assert context.pin_memory is True
    assert context.supports_compile is True
