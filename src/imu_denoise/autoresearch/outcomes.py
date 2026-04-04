"""Compatibility facade for IMU autoresearch outcome helpers."""

from __future__ import annotations

from .result_persistence import persist_iteration_result
from .result_recording import (
    IterationOutcome,
    handle_crash_outcome,
    handle_interrupted_outcome,
    record_success_outcome,
)

__all__ = [
    "IterationOutcome",
    "handle_crash_outcome",
    "handle_interrupted_outcome",
    "persist_iteration_result",
    "record_success_outcome",
]
