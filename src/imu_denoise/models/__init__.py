"""IMU denoising models.

Importing this package auto-discovers all model modules in the package directory.
"""

from imu_denoise.models.base import BaseDenoiser
from imu_denoise.models.registry import autodiscover_models, get_model, list_models, register_model

autodiscover_models()

__all__ = [
    "BaseDenoiser",
    "autodiscover_models",
    "get_model",
    "list_models",
    "register_model",
]
