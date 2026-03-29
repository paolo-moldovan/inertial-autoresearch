"""Simple complementary-style smoothing baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ComplementaryFilterBaseline:
    """Exponential smoothing baseline over accelerometer and gyroscope channels."""

    def __init__(self, alpha_accel: float = 0.9, alpha_gyro: float = 0.95) -> None:
        self.alpha_accel = alpha_accel
        self.alpha_gyro = alpha_gyro

    def denoise(self, windows: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply exponential smoothing to a batch of IMU windows."""
        denoised = windows.copy()
        for batch_index in range(windows.shape[0]):
            for time_index in range(1, windows.shape[1]):
                denoised[batch_index, time_index, :3] = (
                    self.alpha_accel * denoised[batch_index, time_index - 1, :3]
                    + (1.0 - self.alpha_accel) * windows[batch_index, time_index, :3]
                )
                denoised[batch_index, time_index, 3:] = (
                    self.alpha_gyro * denoised[batch_index, time_index - 1, 3:]
                    + (1.0 - self.alpha_gyro) * windows[batch_index, time_index, 3:]
                )
        return denoised
