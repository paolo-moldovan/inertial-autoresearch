"""Reusable core abstractions for config-first ML autoresearch."""

from autoresearch_core.contracts import (
    AnalyticsSnapshot,
    CandidateProposal,
    LoopSnapshot,
    ProjectAdapter,
    ProjectConfigHandle,
    RunResult,
    RunSpec,
    SelectionContext,
    SelectionDecision,
)
from autoresearch_core.engine import ProviderSelectionResult, resolve_provider_selection
from autoresearch_core.training import (
    NoOpTrainingControl,
    NoOpTrainingHooks,
    TrainingControl,
    TrainingHooks,
)

__all__ = [
    "AnalyticsSnapshot",
    "CandidateProposal",
    "LoopSnapshot",
    "NoOpTrainingControl",
    "NoOpTrainingHooks",
    "ProjectAdapter",
    "ProjectConfigHandle",
    "ProviderSelectionResult",
    "RunResult",
    "RunSpec",
    "SelectionContext",
    "SelectionDecision",
    "TrainingControl",
    "TrainingHooks",
    "resolve_provider_selection",
]
