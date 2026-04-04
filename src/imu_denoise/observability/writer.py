"""High-level best-effort writer for mission-control observability."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from autoresearch_core.observability.redaction import (
    redact_payload as _core_redact_payload,
)
from autoresearch_core.observability.redaction import (
    redact_text as _core_redact_text,
)
from autoresearch_core.observability.writer import CoreObservabilityWriter
from imu_denoise.config import ExperimentConfig
from imu_denoise.config.schema import ObservabilityConfig
from imu_denoise.observability import experiment_tracking as experiment_helpers
from imu_denoise.observability import mutation_memory as mutation_helpers
from imu_denoise.observability.events import (
    ARTIFACT_REGISTERED,
    DECISION_EVENT,
    LLM_CALL_EVENT,
    LOG_EVENT,
    RUN_FINISHED,
    RUN_STARTED,
    RUN_STATUS,
    TRAINING_EPOCH,
)
from imu_denoise.observability.store import ObservabilityStore

redact_payload = _core_redact_payload
redact_text = _core_redact_text


class ObservabilityLogHandler(logging.Handler):
    """Mirror log records into the mission-control event stream."""

    def __init__(self, writer: ObservabilityWriter, run_id: str) -> None:
        super().__init__()
        self._delegate = writer.create_log_handler(run_id)

    def emit(self, record: logging.LogRecord) -> None:
        self._delegate.emit(record)


class ObservabilityWriter(CoreObservabilityWriter):
    """Best-effort facade over the SQLite observability store."""

    def __init__(
        self,
        *,
        config: ObservabilityConfig,
        store: ObservabilityStore | None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            config=config,
            store=store,
            logger=logger or logging.getLogger("imu_denoise.observability"),
            run_started_event=RUN_STARTED,
            run_finished_event=RUN_FINISHED,
            run_status_event=RUN_STATUS,
            training_epoch_event=TRAINING_EPOCH,
            log_event=LOG_EVENT,
            artifact_registered_event=ARTIFACT_REGISTERED,
            llm_call_event=LLM_CALL_EVENT,
            decision_event=DECISION_EVENT,
        )

    @classmethod
    def from_experiment_config(
        cls,
        config: ExperimentConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> ObservabilityWriter:
        if not config.observability.enabled:
            return cls(config=config.observability, store=None, logger=logger)
        store = ObservabilityStore(
            db_path=Path(config.observability.db_path),
            blob_dir=Path(config.observability.blob_dir),
        )
        return cls(config=config.observability, store=store, logger=logger)

    def ensure_experiment(
        self,
        *,
        config: ExperimentConfig,
        overrides: list[str] | None = None,
        objective_metric: str | None = None,
        objective_direction: str | None = None,
        source: str = "runtime",
        summary: dict[str, Any] | None = None,
    ) -> str | None:
        return experiment_helpers.ensure_experiment(
            self,
            config=config,
            overrides=overrides,
            objective_metric=objective_metric,
            objective_direction=objective_direction,
            source=source,
            summary=summary,
        )

    def start_run(
        self,
        *,
        name: str,
        phase: str,
        dataset: str | None = None,
        model: str | None = None,
        device: str | None = None,
        parent_run_id: str | None = None,
        iteration: int | None = None,
        experiment_id: str | None = None,
        config: Any | None = None,
        overrides: list[str] | None = None,
        objective_metric: str | None = None,
        objective_direction: str | None = None,
        source: str = "runtime",
        run_id: str | None = None,
    ) -> str:
        experiment_id = None
        if config is not None:
            experiment_id = self.ensure_experiment(
                config=config,
                overrides=overrides,
                objective_metric=objective_metric,
                objective_direction=objective_direction,
                source=source,
            )
        return super().start_run(
            name=name,
            phase=phase,
            dataset=dataset,
            model=model,
            device=device,
            parent_run_id=parent_run_id,
            iteration=iteration,
            experiment_id=experiment_id,
            source=source,
            run_id=run_id,
        )

    def record_change_set(
        self,
        *,
        run_id: str,
        loop_run_id: str | None,
        parent_run_id: str | None,
        incumbent_run_id: str | None,
        reference_kind: str,
        proposal_source: str,
        description: str,
        overrides: list[str],
        current_config: Mapping[str, Any] | Any,
        reference_config: Mapping[str, Any] | Any | None = None,
        source: str = "runtime",
        change_set_id: str | None = None,
    ) -> dict[str, Any]:
        return experiment_helpers.record_change_set(
            self,
            run_id=run_id,
            loop_run_id=loop_run_id,
            parent_run_id=parent_run_id,
            incumbent_run_id=incumbent_run_id,
            reference_kind=reference_kind,
            proposal_source=proposal_source,
            description=description,
            overrides=overrides,
            current_config=current_config,
            reference_config=reference_config,
            source=source,
            change_set_id=change_set_id,
        )

    def record_selection_event(
        self,
        *,
        run_id: str,
        loop_run_id: str | None,
        iteration: int | None,
        proposal_source: str,
        description: str,
        incumbent_run_id: str | None,
        candidate_count: int | None,
        rationale: str | None,
        policy_state: dict[str, Any] | None = None,
        source: str = "runtime",
        fingerprint: str | None = None,
    ) -> dict[str, Any]:
        return experiment_helpers.record_selection_event(
            self,
            run_id=run_id,
            loop_run_id=loop_run_id,
            iteration=iteration,
            proposal_source=proposal_source,
            description=description,
            incumbent_run_id=incumbent_run_id,
            candidate_count=candidate_count,
            rationale=rationale,
            policy_state=policy_state,
            source=source,
            fingerprint=fingerprint,
        )

    def record_mutation_outcome(
        self,
        *,
        run_id: str,
        loop_run_id: str | None,
        regime_fingerprint: str,
        proposal_source: str,
        description: str,
        change_items: list[dict[str, Any]],
        status: str,
        metric_key: str,
        metric_value: float | None,
        incumbent_metric: float | None,
        direction: str,
        source: str = "runtime",
    ) -> list[dict[str, Any]]:
        return mutation_helpers.record_mutation_outcome(
            self,
            run_id=run_id,
            loop_run_id=loop_run_id,
            regime_fingerprint=regime_fingerprint,
            proposal_source=proposal_source,
            description=description,
            change_items=change_items,
            status=status,
            metric_key=metric_key,
            metric_value=metric_value,
            incumbent_metric=incumbent_metric,
            direction=direction,
            source=source,
        )

    def config_payload(self, config: Mapping[str, Any] | Any) -> dict[str, Any]:
        return experiment_helpers.config_payload(self, config)
