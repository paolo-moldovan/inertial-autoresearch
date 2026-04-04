"""Reusable log mirroring helpers for Mission Control."""

from __future__ import annotations

import logging
from typing import Any


class MissionControlLogHandler(logging.Handler):
    """Mirror log records into a Mission Control event stream."""

    def __init__(
        self,
        writer: Any,
        run_id: str,
        *,
        log_event_type: str,
    ) -> None:
        super().__init__()
        self.writer = writer
        self.run_id = run_id
        self.log_event_type = log_event_type

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info and record.exc_info[1] is not None:
                payload["exception"] = logging.Formatter().formatException(record.exc_info)
            self.writer.append_event(
                run_id=self.run_id,
                event_type=self.log_event_type,
                level=record.levelname,
                title=record.getMessage()[:120],
                payload=payload,
                source="runtime",
                created_at=record.created,
            )
        except Exception:
            pass
