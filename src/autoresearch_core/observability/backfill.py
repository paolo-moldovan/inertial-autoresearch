"""Generic helpers for replaying historical artifacts into Mission Control."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_backfill_run_id(name: str, phase: str) -> str:
    """Create a deterministic fallback run id for imported historical records."""
    return f"backfill:{phase}:{name}"


def parse_iso_timestamp(value: str) -> float:
    """Parse an ISO-8601 timestamp into epoch seconds."""
    return datetime.fromisoformat(value).timestamp()


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dict payloads."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(dict(json.loads(stripped)))
    return rows


def load_nearest_run_manifest(
    path: Path,
    *,
    stop_dir_name: str = "runs",
) -> dict[str, Any] | None:
    """Find the nearest `run.json` walking upward from an artifact or log path."""
    for parent in (path.parent, *path.parents):
        manifest_path = parent / "run.json"
        if manifest_path.exists():
            return dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        if parent.name == stop_dir_name:
            break
    return None


def resolve_manifest_run_reference(
    path: Path,
    *,
    default_name: str,
    default_phase: str,
) -> tuple[str, str, str]:
    """Resolve `(run_id, run_name, phase)` from nearby manifest data or fallbacks."""
    manifest = load_nearest_run_manifest(path)
    if manifest is None:
        return build_backfill_run_id(default_name, default_phase), default_name, default_phase
    run_name = str(manifest.get("name") or default_name)
    phase = str(manifest.get("phase") or default_phase)
    run_id = str(manifest.get("run_id") or build_backfill_run_id(run_name, phase))
    return run_id, run_name, phase


def ensure_backfill_run(
    writer: Any,
    *,
    run_id: str,
    run_name: str,
    phase: str,
    source: str = "backfill",
) -> None:
    """Ensure a logical run exists before replaying historical records into it."""
    writer.start_run(
        name=run_name,
        phase=phase,
        dataset=None,
        model=None,
        device=None,
        source=source,
        run_id=run_id,
    )
