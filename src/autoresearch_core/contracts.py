"""Domain-neutral contracts for reusable autoresearch orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProjectConfigHandle:
    """Opaque project-level config payload plus optional metadata."""

    raw: Any
    name: str
    metric_key: str
    metric_direction: str


@dataclass(frozen=True)
class CandidateProposal:
    """Generic candidate proposal evaluated by the policy and providers."""

    description: str
    overrides: list[str]
    groups: tuple[str, ...] = ()
    architecture_change: bool = False


@dataclass(frozen=True)
class RunSpec:
    """Generic execution request emitted by the loop engine."""

    iteration: int
    name: str
    phase: str
    proposal: CandidateProposal
    overrides: list[str]
    parent_run_id: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class RunResult:
    """Generic result shape returned from a project adapter."""

    iteration: int
    run_name: str
    status: str
    proposal_source: str
    metric_key: str
    metric_value: float | None
    model_name: str
    description: str
    overrides: list[str]
    metrics_path: Path | None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionContext:
    """Context handed to providers and policy logic for one iteration."""

    iteration: int
    metric_key: str
    metric_direction: str
    history: list[dict[str, Any]]
    incumbent: dict[str, Any] | None
    search_space: dict[str, Any] | None
    mutation_lessons: list[dict[str, Any]] | None
    blocked_candidates: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionDecision:
    """Chosen proposal plus traceability for why it was selected."""

    selected: CandidateProposal
    selected_index: int
    proposal_source: str
    preferred_candidate_index: int | None = None
    blocked_candidates: dict[str, list[str]] = field(default_factory=dict)
    provider_trace: Any | None = None
    policy_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopSnapshot:
    """Current loop status for observability and UI surfaces."""

    loop_run_id: str
    loop_name: str
    current_iteration: int
    max_iterations: int
    status: str
    best_metric: float | None = None
    best_run_id: str | None = None


@dataclass(frozen=True)
class AnalyticsSnapshot:
    """High-level loop or multi-loop analytics summary."""

    loop_count: int
    total_runs: int
    keep_count: int
    discard_count: int
    crash_count: int
    source_counts: dict[str, int] = field(default_factory=dict)
    mutation_group_counts: dict[str, int] = field(default_factory=dict)
    model_family_wins: dict[str, int] = field(default_factory=dict)
    time_to_best_iteration: int | None = None
    total_improvement: float | None = None


@runtime_checkable
class ProjectAdapter(Protocol):
    """Project-specific bridge between the reusable core and domain runtime."""

    def resolve_base_config(
        self,
        *,
        config_paths: list[str],
        base_overrides: list[str],
    ) -> Any: ...

    def resolve_iteration_config(
        self,
        *,
        base_config: Any,
        base_overrides: list[str],
        proposal_overrides: list[str],
        incumbent_config: dict[str, Any] | None = None,
        extra_overrides: list[str] | None = None,
    ) -> Any: ...

    def execute_training_run(
        self,
        *,
        config: Any,
        overrides: list[str],
        metric_key: str,
        parent_run_id: str | None = None,
        iteration: int | None = None,
        run_id: str | None = None,
    ) -> tuple[Any, Any, str]: ...

    def execute_baseline_run(self, *args: Any, **kwargs: Any) -> Any: ...

    def get_mutation_catalog(self) -> list[CandidateProposal]: ...
