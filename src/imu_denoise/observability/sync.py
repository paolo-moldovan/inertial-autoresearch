"""High-level sync orchestration for Mission Control Phase 2 adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imu_denoise.config import ExperimentConfig
from imu_denoise.observability.adapters import MlflowExporter, PhoenixExporter
from imu_denoise.observability.queries import MissionControlQueries


def sync_observability(
    *,
    config: ExperimentConfig,
    target: str = "all",
    run_id: str | None = None,
    limit: int = 100,
) -> dict[str, dict[str, Any]]:
    """Sync the local observability store into optional external adapters."""
    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )
    results: dict[str, dict[str, Any]] = {}

    want_mlflow = target in ("all", "mlflow")
    want_phoenix = target in ("all", "phoenix")

    if want_mlflow:
        if config.observability.mlflow_enabled:
            results["mlflow"] = MlflowExporter(
                config=config.observability,
                queries=queries,
            ).sync(run_id=run_id, limit=limit)
        else:
            results["mlflow"] = {"status": "disabled"}

    if want_phoenix:
        if config.observability.phoenix_enabled:
            results["phoenix"] = PhoenixExporter(
                config=config.observability,
                queries=queries,
            ).sync(run_id=run_id, limit=limit)
        else:
            results["phoenix"] = {"status": "disabled"}

    return results
