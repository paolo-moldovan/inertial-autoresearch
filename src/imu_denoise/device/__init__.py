"""Device abstraction layer for CUDA/MPS/CPU."""

from imu_denoise.device.context import DeviceContext
from imu_denoise.device.detect import detect_device

__all__ = ["DeviceContext", "detect_device"]
