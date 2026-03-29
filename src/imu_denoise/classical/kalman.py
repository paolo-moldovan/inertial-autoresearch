"""Per-axis Kalman smoothing baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class KalmanFilterBaseline:
    """Independent scalar Kalman filters for each IMU channel."""

    def __init__(self, process_variance: float = 1e-4, measurement_variance: float = 1e-2) -> None:
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def denoise(self, windows: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter each window and channel independently."""
        denoised = np.empty_like(windows)

        for batch_index in range(windows.shape[0]):
            for channel_index in range(windows.shape[2]):
                estimate = float(windows[batch_index, 0, channel_index])
                covariance = 1.0
                denoised[batch_index, 0, channel_index] = estimate

                for time_index in range(1, windows.shape[1]):
                    measurement = float(windows[batch_index, time_index, channel_index])
                    covariance += self.process_variance
                    gain = covariance / (covariance + self.measurement_variance)
                    estimate = estimate + gain * (measurement - estimate)
                    covariance = (1.0 - gain) * covariance
                    denoised[batch_index, time_index, channel_index] = estimate

        return denoised
