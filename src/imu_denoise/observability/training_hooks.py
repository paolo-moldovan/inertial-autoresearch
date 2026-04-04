"""Mission Control-backed training hook adapters."""

from __future__ import annotations

from typing import Any

from autoresearch_core.training import (
    LoopAwareTrainingControl,
    NoOpTrainingControl,
    NoOpTrainingHooks,
    WriterBackedTrainingHooks,
)
from imu_denoise.config.schema import ExperimentConfig
from imu_denoise.observability.writer import ObservabilityWriter


def build_training_hooks(
    *,
    config: ExperimentConfig,
    observability: ObservabilityWriter | None = None,
) -> Any:
    if observability is None:
        observability = ObservabilityWriter.from_experiment_config(config, logger=None)
    if observability.store is None:
        return NoOpTrainingHooks()
    return MissionControlTrainingHooks(observability)


def build_training_control(
    *,
    parent_run_id: str | None,
    run_id: str,
    observability: ObservabilityWriter | None,
) -> Any:
    if parent_run_id is None or observability is None or observability.store is None:
        return NoOpTrainingControl()
    return MissionControlTrainingControl(
        parent_run_id=parent_run_id,
        active_child_run_id=run_id,
        observability=observability,
    )


class MissionControlTrainingHooks(WriterBackedTrainingHooks):
    """Hook adapter that forwards trainer lifecycle signals to observability."""

    def __init__(self, observability: ObservabilityWriter) -> None:
        super().__init__(observability)
        self.observability = observability


class MissionControlTrainingControl(LoopAwareTrainingControl):
    """Control adapter used by the trainer without exposing store/controller internals."""

    def __init__(
        self,
        *,
        parent_run_id: str,
        active_child_run_id: str,
        observability: ObservabilityWriter,
    ) -> None:
        self.observability = observability
        from imu_denoise.observability.control import LoopController
        from imu_denoise.training.trainer import TrainingInterrupted

        store = self.observability.store
        if store is None:
            raise RuntimeError(
                "MissionControlTrainingControl requires an enabled observability store."
            )
        controller = LoopController(store=store, writer=self.observability)
        super().__init__(
            parent_run_id=parent_run_id,
            active_child_run_id=active_child_run_id,
            get_loop_state=controller.get_loop_state,
            heartbeat_updater=controller.heartbeat,
            interrupt_exception_factory=lambda status, message: TrainingInterrupted(
                status,
                message,
            ),
        )
