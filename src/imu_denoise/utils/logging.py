"""Structured logging setup with console and optional JSON file output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path


class _RichConsoleFormatter(logging.Formatter):
    """Console formatter with level-specific coloring and structured layout."""

    _COLORS = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, self._RESET)
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        level = f"{color}{record.levelname:<8}{self._RESET}"
        return f"{ts} | {level} | {record.name} | {record.getMessage()}"


class _JsonLinesFormatter(logging.Formatter):
    """Formats each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logger(
    name: str,
    log_dir: str | None = None,
    log_filename: str | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """Create and configure a logger with console output and optional file logging.

    Args:
        name: Logger name, typically the module or experiment name.
        log_dir: If provided, write JSON-lines log to ``<log_dir>/<file>.jsonl``.
        log_filename: Optional filename stem for the JSON-lines log.
        level: Logging level as a string (``"DEBUG"``, ``"INFO"``, etc.).

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_RichConsoleFormatter())
    logger.addHandler(console)

    # Optional file handler (JSON lines)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_stem = log_filename or name
        fh = logging.FileHandler(log_path / f"{file_stem}.jsonl", encoding="utf-8")
        fh.setFormatter(_JsonLinesFormatter())
        logger.addHandler(fh)

    return logger
