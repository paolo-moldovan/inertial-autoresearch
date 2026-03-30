"""Mission Control-backed training hook adapters."""

from __future__ import annotations

from typing import Any

from autoresearch_core.training import NoOpTrainingControl, NoOpTrainingHooks
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


class MissionControlTrainingHooks:
    """Hook adapter that forwards trainer lifecycle signals to observability."""

    def __init__(self, observability: ObservabilityWriter) -> None:
        self.observability = observability

    def make_run_id(self, *, name: str, phase: str) -> str:
        return self.observability.make_run_id(name=name, phase=phase)

    def start_run(self, **kwargs: Any) -> str:
        return self.observability.start_run(**kwargs)

    def create_log_handler(self, run_id: str) -> Any:
        return self.observability.create_log_handler(run_id)

    def update_status(self, **kwargs: Any) -> None:
        self.observability.update_status(**kwargs)

    def append_event(self, **kwargs: Any) -> None:
        self.observability.append_event(**kwargs)

    def register_artifact(self, **kwargs: Any) -> None:
        self.observability.register_artifact(**kwargs)

    def record_epoch(self, **kwargs: Any) -> None:
        self.observability.record_epoch(**kwargs)

    def finish_run(self, **kwargs: Any) -> None:
        self.observability.finish_run(**kwargs)


class MissionControlTrainingControl:
    """Control adapter used by the trainer without exposing store/controller internals."""

    def __init__(
        self,
        *,
        parent_run_id: str,
        active_child_run_id: str,
        observability: ObservabilityWriter,
    ) -> None:
        self.parent_run_id = parent_run_id
        self.active_child_run_id = active_child_run_id
        self.observability = observability

    def check_abort(self) -> None:
        from imu_denoise.observability.control import LoopController
        from imu_denoise.training.trainer import TrainingInterrupted

        store = self.observability.store
        if store is None:
            return
        controller = LoopController(store=store, writer=self.observability)
        loop_state = controller.get_loop_state(self.parent_run_id)
        if loop_state is None:
            return
        if bool(loop_state.get("terminate_requested")):
            raise TrainingInterrupted("terminated", "Training terminated by control-plane request.")

    def heartbeat(self, *, best_metric: float, active_child_run_id: str | None = None) -> None:
        from imu_denoise.observability.control import LoopController

        store = self.observability.store
        if store is None:
            return
        controller = LoopController(store=store, writer=self.observability)
        loop_state = controller.get_loop_state(self.parent_run_id)
        if loop_state is None:
            return
        controller.heartbeat(
            loop_run_id=self.parent_run_id,
            current_iteration=int(loop_state["current_iteration"]),
            max_iterations=int(loop_state["max_iterations"]),
            batch_size=(
                int(loop_state["batch_size"])
                if isinstance(loop_state.get("batch_size"), int)
                else None
            ),
            pause_after_iteration=(
                int(loop_state["pause_after_iteration"])
                if isinstance(loop_state.get("pause_after_iteration"), int)
                else None
            ),
            pause_requested=bool(loop_state.get("pause_requested")),
            stop_requested=bool(loop_state.get("stop_requested")),
            terminate_requested=bool(loop_state.get("terminate_requested")),
            best_metric=best_metric if best_metric != float("inf") else None,
            best_run_id=(
                str(loop_state["best_run_id"])
                if loop_state.get("best_run_id") is not None
                else None
            ),
            active_child_run_id=active_child_run_id or self.active_child_run_id,
            status=str(loop_state.get("status") or "running"),
        )
