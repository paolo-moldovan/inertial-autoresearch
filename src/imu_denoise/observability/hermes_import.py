"""IMU-facing compatibility wrapper for Hermes session import."""

from __future__ import annotations

from pathlib import Path

from autoresearch_core.observability.hermes_import import import_hermes_state as _core_import
from imu_denoise.observability.events import HERMES_SESSION_IMPORTED, HERMES_TRANSCRIPT_IMPORTED
from imu_denoise.observability.writer import ObservabilityWriter


def import_hermes_state(*, writer: ObservabilityWriter, hermes_home: Path) -> dict[str, int]:
    """Import Hermes SQLite and JSON session state into the repo-local store."""
    return _core_import(
        writer=writer,
        hermes_home=hermes_home,
        origin="hermes_import",
        session_event=HERMES_SESSION_IMPORTED,
        transcript_event=HERMES_TRANSCRIPT_IMPORTED,
    )
