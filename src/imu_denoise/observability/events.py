"""Typed event and record shapes for mission-control observability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RUN_STARTED = "run_started"
RUN_FINISHED = "run_finished"
RUN_STATUS = "run_status"
TRAINING_EPOCH = "training_epoch"
LOG_EVENT = "log"
ARTIFACT_REGISTERED = "artifact_registered"
LLM_CALL_EVENT = "llm_call"
DECISION_EVENT = "decision"
HERMES_SESSION_IMPORTED = "hermes_session_imported"
HERMES_TRANSCRIPT_IMPORTED = "hermes_transcript_imported"


@dataclass(frozen=True)
class RunRecord:
    """Current state of a logical pipeline run."""

    id: str
    name: str
    phase: str
    dataset: str | None
    model: str | None
    device: str | None
    status: str
    started_at: float
    ended_at: float | None
    parent_run_id: str | None
    iteration: int | None
    experiment_id: str | None
    source: str


@dataclass(frozen=True)
class DecisionRecord:
    """A single autoresearch decision outcome."""

    run_id: str
    iteration: int | None
    proposal_source: str
    description: str
    status: str
    metric_key: str
    metric_value: float | None
    overrides: list[str]
    reason: str | None
    llm_call_id: str | None
    candidates: list[dict[str, Any]] | None
    source: str


@dataclass(frozen=True)
class LLMCallRecord:
    """Captured LLM invocation metadata."""

    id: str
    run_id: str | None
    provider: str | None
    model: str | None
    base_url: str | None
    status: str
    latency_ms: float | None
    session_id: str | None
    parsed_payload: dict[str, Any] | None
    reason: str | None
    source: str


@dataclass(frozen=True)
class ArtifactRecord:
    """Artifact produced by a run."""

    run_id: str
    artifact_type: str
    path: str
    label: str | None
    metadata: dict[str, Any] | None
    source: str


@dataclass(frozen=True)
class StatusSnapshotRecord:
    """Latest live status for a run."""

    run_id: str
    phase: str | None
    epoch: int | None
    best_metric: float | None
    last_metric: float | None
    heartbeat_at: float
    message: str | None
