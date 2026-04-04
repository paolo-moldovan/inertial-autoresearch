"""Read models for mission-control TUI and dashboard surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from autoresearch_core.observability.queries import (
    CoreMissionControlQueries,
)
from imu_denoise.observability import mutation_queries as mutation_helpers
from imu_denoise.observability import regime_queries as regime_helpers
from imu_denoise.observability import summary_queries as summary_helpers
from imu_denoise.observability.store import ObservabilityStore


class MissionControlQueries(CoreMissionControlQueries):
    """High-level queries over the observability store."""

    def __init__(self, db_path: Path, blob_dir: Path) -> None:
        super().__init__(store=ObservabilityStore(db_path=db_path, blob_dir=blob_dir))

    def list_leaderboard(
        self,
        *,
        limit: int = 10,
        metric_key: str = "val_rmse",
        regime_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        return regime_helpers.list_leaderboard(
            self,
            limit=limit,
            metric_key=metric_key,
            regime_fingerprint=regime_fingerprint,
        )

    def find_best_global_incumbent(
        self,
        *,
        metric_key: str,
        dataset: str,
        direction: str = "minimize",
        reference_config: Mapping[str, Any] | Any | None = None,
    ) -> dict[str, Any] | None:
        return regime_helpers.find_best_global_incumbent(
            self,
            metric_key=metric_key,
            dataset=dataset,
            direction=direction,
            reference_config=reference_config,
        )

    def get_run_identity(self, run_id: str) -> dict[str, Any] | None:
        return regime_helpers.get_run_identity(self, run_id)

    def get_run_regime_fingerprint(self, run_id: str) -> str | None:
        return regime_helpers.get_run_regime_fingerprint(self, run_id)

    def get_run_reference(self, run_id: str | None) -> dict[str, Any] | None:
        return regime_helpers.get_run_reference(self, run_id)

    def get_related_mutation_lessons(self, run_id: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return mutation_helpers.get_related_mutation_lessons(self, run_id, limit=limit)

    def _get_run_detail_extensions(self, run_id: str) -> dict[str, Any]:
        return {
            "mutation_attempts": self.list_mutation_attempts(run_id=run_id, limit=100),
            "related_lessons": self.get_related_mutation_lessons(run_id, limit=8),
        }

    def get_mission_control_summary(self, *, limit: int = 10) -> dict[str, Any]:
        return summary_helpers.get_mission_control_summary(self, limit=limit)

    def list_mutation_leaderboard(
        self,
        *,
        limit: int = 10,
        regime_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        return mutation_helpers.list_mutation_leaderboard(
            self,
            limit=limit,
            regime_fingerprint=regime_fingerprint,
        )

    def list_recent_mutation_lessons(
        self,
        *,
        limit: int = 20,
        regime_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        return mutation_helpers.list_recent_mutation_lessons(
            self,
            limit=limit,
            regime_fingerprint=regime_fingerprint,
        )

    def list_mutation_attempts(
        self,
        *,
        run_id: str | None = None,
        regime_fingerprint: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        return mutation_helpers.list_mutation_attempts(
            self,
            run_id=run_id,
            regime_fingerprint=regime_fingerprint,
            limit=limit,
        )

    def get_mutation_stats_for_signatures(
        self,
        *,
        signatures: list[str],
        regime_fingerprint: str,
    ) -> dict[str, dict[str, Any]]:
        return mutation_helpers.get_mutation_stats_for_signatures(
            self,
            signatures=signatures,
            regime_fingerprint=regime_fingerprint,
        )

    def list_recent_loop_events(
        self,
        *,
        limit: int = 50,
        loop_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return summary_helpers.list_recent_loop_events(self, limit=limit, loop_run_id=loop_run_id)
