"""Reusable redaction helpers for Mission Control payloads."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
KEY_PATTERN = re.compile(r"(?i)(api[_-]?key|authorization|auth[_-]?header|bearer)")
BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]+")
INLINE_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,]+)")


def redact_text(text: str) -> str:
    """Redact narrow secret patterns while leaving normal prompt content intact."""
    redacted = BEARER_PATTERN.sub("Bearer " + REDACTED, text)
    redacted = INLINE_SECRET_PATTERN.sub(r"\1" + REDACTED, redacted)
    return redacted


def redact_payload(payload: Any) -> Any:
    """Recursively redact auth-like fields from structured payloads."""
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            if KEY_PATTERN.search(key):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_payload(value)
        return redacted
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, str):
        return redact_text(payload)
    return payload
