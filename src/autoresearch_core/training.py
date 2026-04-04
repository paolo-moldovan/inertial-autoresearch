"""Training hook and control abstractions for domain runtimes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable


@runtime_checkable
class TrainingHooks(Protocol):
    """Hook surface used by training runtimes without coupling to observability internals."""

    def make_run_id(self, *, name: str, phase: str) -> str: ...

    def start_run(self, **kwargs: Any) -> str: ...

    def create_log_handler(self, run_id: str) -> logging.Handler: ...

    def update_status(self, **kwargs: Any) -> None: ...

    def append_event(self, **kwargs: Any) -> None: ...

    def register_artifact(self, **kwargs: Any) -> None: ...

    def record_epoch(self, **kwargs: Any) -> None: ...

    def finish_run(self, **kwargs: Any) -> None: ...


@runtime_checkable
class TrainingControl(Protocol):
    """Control-plane interaction surface used by the trainer."""

    def check_abort(self) -> None: ...

    def heartbeat(self, *, best_metric: float, active_child_run_id: str | None = None) -> None: ...


class NoOpTrainingHooks:
    """Minimal hook implementation for tests or non-observed runs."""

    def make_run_id(self, *, name: str, phase: str) -> str:
        return f"{phase}:{name}"

    def start_run(self, **kwargs: Any) -> str:
        run_id = kwargs.get("run_id")
        if isinstance(run_id, str):
            return run_id
        name = str(kwargs.get("name") or "run")
        phase = str(kwargs.get("phase") or "training")
        return self.make_run_id(name=name, phase=phase)

    def create_log_handler(self, run_id: str) -> logging.Handler:
        return logging.NullHandler()

    def update_status(self, **kwargs: Any) -> None:
        return None

    def append_event(self, **kwargs: Any) -> None:
        return None

    def register_artifact(self, **kwargs: Any) -> None:
        return None

    def record_epoch(self, **kwargs: Any) -> None:
        return None

    def finish_run(self, **kwargs: Any) -> None:
        return None


class NoOpTrainingControl:
    """Control stub for standalone training without a loop controller."""

    def check_abort(self) -> None:
        return None

    def heartbeat(self, *, best_metric: float, active_child_run_id: str | None = None) -> None:
        return None


class WriterBackedTrainingHooks:
    """Generic hook adapter that delegates trainer lifecycle calls to a writer-like object."""

    def __init__(self, writer: Any) -> None:
        self.writer = writer

    def make_run_id(self, *, name: str, phase: str) -> str:
        return cast(str, self.writer.make_run_id(name=name, phase=phase))

    def start_run(self, **kwargs: Any) -> str:
        return cast(str, self.writer.start_run(**kwargs))

    def create_log_handler(self, run_id: str) -> logging.Handler:
        return cast(logging.Handler, self.writer.create_log_handler(run_id))

    def update_status(self, **kwargs: Any) -> None:
        self.writer.update_status(**kwargs)

    def append_event(self, **kwargs: Any) -> None:
        self.writer.append_event(**kwargs)

    def register_artifact(self, **kwargs: Any) -> None:
        self.writer.register_artifact(**kwargs)

    def record_epoch(self, **kwargs: Any) -> None:
        self.writer.record_epoch(**kwargs)

    def finish_run(self, **kwargs: Any) -> None:
        self.writer.finish_run(**kwargs)


class LoopAwareTrainingControl:
    """Generic loop-aware training control that polls loop state and forwards heartbeats."""

    def __init__(
        self,
        *,
        parent_run_id: str,
        active_child_run_id: str,
        get_loop_state: Callable[[str], dict[str, Any] | None],
        heartbeat_updater: Callable[..., None],
        interrupt_exception_factory: Callable[[str, str], Exception],
    ) -> None:
        self.parent_run_id = parent_run_id
        self.active_child_run_id = active_child_run_id
        self._get_loop_state = get_loop_state
        self._heartbeat_updater = heartbeat_updater
        self._interrupt_exception_factory = interrupt_exception_factory

    def check_abort(self) -> None:
        loop_state = self._get_loop_state(self.parent_run_id)
        if loop_state is None:
            return
        if bool(loop_state.get("terminate_requested")):
            raise self._interrupt_exception_factory(
                "terminated",
                "Training terminated by control-plane request.",
            )

    def heartbeat(self, *, best_metric: float, active_child_run_id: str | None = None) -> None:
        loop_state = self._get_loop_state(self.parent_run_id)
        if loop_state is None:
            return
        batch_size = loop_state.get("batch_size")
        pause_after_iteration = loop_state.get("pause_after_iteration")
        best_run_id = loop_state.get("best_run_id")
        self._heartbeat_updater(
            loop_run_id=self.parent_run_id,
            current_iteration=int(loop_state["current_iteration"]),
            max_iterations=int(loop_state["max_iterations"]),
            batch_size=int(batch_size) if isinstance(batch_size, int) else None,
            pause_after_iteration=(
                int(pause_after_iteration) if isinstance(pause_after_iteration, int) else None
            ),
            pause_requested=bool(loop_state.get("pause_requested")),
            stop_requested=bool(loop_state.get("stop_requested")),
            terminate_requested=bool(loop_state.get("terminate_requested")),
            best_metric=best_metric if best_metric != float("inf") else None,
            best_run_id=str(best_run_id) if best_run_id is not None else None,
            active_child_run_id=active_child_run_id or self.active_child_run_id,
            status=str(loop_state.get("status") or "running"),
        )
