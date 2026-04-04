"""IMU-domain Mission Control control-plane wrapper."""

from __future__ import annotations

from autoresearch_core.observability.control import (
    LOOP_PAUSED,
    LOOP_RESUMED,
    LOOP_STOP_REQUESTED,
    LOOP_STOPPED,
    LOOP_TERMINATE_REQUESTED,
    LOOP_TERMINATED,
    QUEUE_APPLIED,
    QUEUE_CLAIMED,
    QUEUE_ENQUEUED,
    LoopAlreadyRunningError,
)
from autoresearch_core.observability.control import (
    LoopController as CoreLoopController,
)
from imu_denoise.config import ExperimentConfig
from imu_denoise.observability.writer import ObservabilityWriter


class LoopController(CoreLoopController):
    """Config-aware IMU wrapper over the reusable Mission Control controller."""

    @classmethod
    def from_experiment_config(
        cls,
        config: ExperimentConfig,
        *,
        writer: ObservabilityWriter | None = None,
    ) -> LoopController:
        resolved_writer = writer or ObservabilityWriter.from_experiment_config(config)
        if resolved_writer.store is None:
            raise RuntimeError("Observability must be enabled for loop control.")
        return cls(store=resolved_writer.store, writer=resolved_writer)


__all__ = [
    "LOOP_PAUSED",
    "LOOP_RESUMED",
    "LOOP_STOPPED",
    "LOOP_STOP_REQUESTED",
    "LOOP_TERMINATED",
    "LOOP_TERMINATE_REQUESTED",
    "QUEUE_APPLIED",
    "QUEUE_CLAIMED",
    "QUEUE_ENQUEUED",
    "LoopAlreadyRunningError",
    "LoopController",
]
