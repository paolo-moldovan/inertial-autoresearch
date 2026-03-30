"""IMU-specific adapter that bridges the domain runtime to the reusable core."""

from __future__ import annotations

from typing import Any

from autoresearch_core.contracts import CandidateProposal, ProjectAdapter
from imu_denoise.autoresearch.execution import (
    resolve_iteration_config,
    run_single_experiment,
)
from imu_denoise.autoresearch.mutations import default_mutation_pool
from imu_denoise.cli.common import resolve_config


class IMUProjectAdapter(ProjectAdapter):
    """Project adapter for the IMU denoising domain."""

    def resolve_base_config(
        self,
        *,
        config_paths: list[str],
        base_overrides: list[str],
    ) -> Any:
        return resolve_config(config_paths, base_overrides)

    def resolve_iteration_config(
        self,
        *,
        base_config: Any,
        base_overrides: list[str],
        proposal_overrides: list[str],
        incumbent_config: dict[str, Any] | None = None,
        extra_overrides: list[str] | None = None,
    ) -> Any:
        return resolve_iteration_config(
            base_config=base_config,
            base_overrides=base_overrides,
            proposal_overrides=proposal_overrides,
            incumbent_config=incumbent_config,
            extra_overrides=extra_overrides,
        )

    def execute_training_run(
        self,
        *,
        config: Any,
        overrides: list[str],
        metric_key: str,
        parent_run_id: str | None = None,
        iteration: int | None = None,
        run_id: str | None = None,
    ) -> tuple[Any, Any, str]:
        return run_single_experiment(
            config=config,
            overrides=overrides,
            metric_key=metric_key,
            parent_run_id=parent_run_id,
            iteration=iteration,
            run_id=run_id,
        )

    def execute_baseline_run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Classical baselines are still executed via the IMU CLI layer.")

    def get_mutation_catalog(self) -> list[CandidateProposal]:
        return default_mutation_pool()
