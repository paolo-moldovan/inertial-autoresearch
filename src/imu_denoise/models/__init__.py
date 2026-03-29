"""IMU denoising models.

Importing this package auto-registers all built-in models with the registry.
"""

from imu_denoise.models import conv1d as conv1d  # noqa: F401
from imu_denoise.models import lstm as lstm  # noqa: F401
from imu_denoise.models import transformer as transformer  # noqa: F401
from imu_denoise.models.base import BaseDenoiser
from imu_denoise.models.registry import get_model, list_models, register_model

__all__ = [
    "BaseDenoiser",
    "get_model",
    "list_models",
    "register_model",
]
