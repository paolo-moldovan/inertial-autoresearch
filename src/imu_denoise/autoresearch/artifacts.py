"""Artifact and manifest helpers for IMU autoresearch loop runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from imu_denoise.config import ExperimentConfig


RESULTS_HEADER = [
    "timestamp",
    "iteration",
    "run_name",
    "status",
    "proposal_source",
    "metric_key",
    "metric_value",
    "model_name",
    "description",
    "overrides",
    "metrics_path",
]


class SupportsLoopArtifactResult(Protocol):
    """Structural protocol for loop results written to autoresearch TSV artifacts."""

    @property
    def iteration(self) -> int: ...

    @property
    def run_name(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def proposal_source(self) -> str: ...

    @property
    def metric_key(self) -> str: ...

    @property
    def metric_value(self) -> float | None: ...

    @property
    def model_name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def overrides(self) -> list[str]: ...

    @property
    def metrics_path(self) -> Path | None: ...


def sanitize_tsv_field(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").strip()


def ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(RESULTS_HEADER) + "\n")


def append_result(path: Path, result: SupportsLoopArtifactResult) -> None:
    row = [
        datetime.now(tz=UTC).isoformat(),
        str(result.iteration),
        result.run_name,
        result.status,
        result.proposal_source,
        result.metric_key,
        "" if result.metric_value is None else f"{result.metric_value:.6f}",
        result.model_name,
        sanitize_tsv_field(result.description),
        sanitize_tsv_field(json.dumps(result.overrides, separators=(",", ":"))),
        "" if result.metrics_path is None else str(result.metrics_path),
    ]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def resolve_results_file(
    *,
    base_config: ExperimentConfig,
    loop_run_id: str,
    loop_name: str,
) -> Path:
    from imu_denoise.config import AutoResearchConfig
    from imu_denoise.utils.paths import build_run_paths

    configured = Path(base_config.autoresearch.results_file)
    default_path = Path(AutoResearchConfig().results_file)
    if configured == default_path:
        return build_run_paths(
            base_config.output_dir,
            run_name=loop_name,
            run_id=loop_run_id,
        ).loop_results_path
    return configured


def safe_update_run_manifest(run_paths: Any, payload: dict[str, Any]) -> None:
    try:
        from imu_denoise.utils.paths import update_run_manifest

        update_run_manifest(run_paths, payload)
    except Exception:
        return
