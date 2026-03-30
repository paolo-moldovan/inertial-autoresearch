"""Training hook and control abstractions for domain runtimes."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable


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
