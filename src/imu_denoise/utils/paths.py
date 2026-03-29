"""Helpers for consistent per-run artifact layout."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "run"


def _run_token(run_id: str) -> str:
    parts = run_id.split(":")
    if len(parts) >= 2:
        return f"{parts[-2]}-{parts[-1]}"
    return run_id[-24:] if len(run_id) > 24 else run_id


@dataclass(frozen=True)
class RunPaths:
    """Canonical filesystem paths for a single run."""

    run_id: str
    run_name: str
    root: Path
    logs_dir: Path
    checkpoints_dir: Path
    figures_dir: Path
    metrics_path: Path
    history_path: Path
    runtime_log_path: Path
    loop_results_path: Path
    manifest_path: Path


def build_run_paths(output_dir: str | Path, *, run_name: str, run_id: str) -> RunPaths:
    """Build the canonical per-run directory layout."""
    root = Path(output_dir) / "runs" / f"{_slugify(run_name)}--{_run_token(run_id)}"
    logs_dir = root / "logs"
    checkpoints_dir = root / "checkpoints"
    figures_dir = root / "figures"
    return RunPaths(
        run_id=run_id,
        run_name=run_name,
        root=root,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        figures_dir=figures_dir,
        metrics_path=root / "metrics.json",
        history_path=logs_dir / "history.jsonl",
        runtime_log_path=logs_dir / "runtime.jsonl",
        loop_results_path=root / "loop" / "results.tsv",
        manifest_path=root / "run.json",
    )


def write_run_manifest(run_paths: RunPaths, payload: dict[str, Any]) -> None:
    """Write a small run manifest for human inspection and disk-only recovery."""
    run_paths.root.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if run_paths.manifest_path.exists():
        existing = json.loads(run_paths.manifest_path.read_text(encoding="utf-8"))
    merged = {**existing, **payload}
    run_paths.manifest_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def update_run_manifest(run_paths: RunPaths, payload: dict[str, Any]) -> None:
    """Merge additional metadata into an existing run manifest."""
    write_run_manifest(run_paths, payload)
