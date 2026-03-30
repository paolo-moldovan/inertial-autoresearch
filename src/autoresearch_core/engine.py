"""Reusable loop-engine helpers for provider selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from random import Random
from typing import Any

from autoresearch_core.contracts import CandidateProposal


@dataclass(frozen=True)
class ProviderSelectionResult:
    """Resolved provider-facing candidate pool and chosen index."""

    candidates: list[CandidateProposal]
    blocked_candidates: dict[str, list[str]] = field(default_factory=dict)
    preferred_candidate_index: int = 0
    proposal_source: str = "static"
    provider_trace: Any | None = None


def resolve_provider_selection(
    *,
    iteration: int,
    orchestrator: str,
    fallback_proposal: CandidateProposal,
    candidate_pool: list[CandidateProposal],
    blocked_candidates: dict[str, list[str]] | None = None,
    used_descriptions: set[str] | None = None,
    rng: Random | None = None,
    provider_ready: Callable[[], bool] | None = None,
    provider_select: (
        Callable[[list[CandidateProposal]], tuple[CandidateProposal, Any]] | None
    ) = None,
    provider_error: type[Exception] = Exception,
) -> ProviderSelectionResult:
    """Resolve the provider-visible pool and selected candidate for one iteration."""
    if iteration == 0:
        return ProviderSelectionResult(
            candidates=[fallback_proposal],
            blocked_candidates=dict(blocked_candidates or {}),
            preferred_candidate_index=0,
            proposal_source="static",
            provider_trace=None,
        )

    available_candidates = _available_candidates(
        candidate_pool=candidate_pool,
        used_descriptions=used_descriptions or set(),
        rng=rng,
    )
    fallback_index = _candidate_index(
        candidates=available_candidates,
        selected=fallback_proposal,
    )

    if orchestrator != "hermes":
        return ProviderSelectionResult(
            candidates=available_candidates,
            blocked_candidates=dict(blocked_candidates or {}),
            preferred_candidate_index=fallback_index,
            proposal_source="static",
            provider_trace=None,
        )

    if provider_ready is None or provider_select is None or not provider_ready():
        return ProviderSelectionResult(
            candidates=available_candidates,
            blocked_candidates=dict(blocked_candidates or {}),
            preferred_candidate_index=fallback_index,
            proposal_source="static-fallback",
            provider_trace=None,
        )

    try:
        proposal, trace = provider_select(available_candidates)
    except provider_error as exc:
        trace = getattr(exc, "trace", None)
        return ProviderSelectionResult(
            candidates=available_candidates,
            blocked_candidates=dict(blocked_candidates or {}),
            preferred_candidate_index=fallback_index,
            proposal_source="static-fallback",
            provider_trace=trace,
        )

    return ProviderSelectionResult(
        candidates=available_candidates,
        blocked_candidates=dict(blocked_candidates or {}),
        preferred_candidate_index=_candidate_index(
            candidates=available_candidates,
            selected=proposal,
        ),
        proposal_source="hermes",
        provider_trace=trace,
    )


def _available_candidates(
    *,
    candidate_pool: list[CandidateProposal],
    used_descriptions: set[str],
    rng: Random | None,
) -> list[CandidateProposal]:
    available = [
        proposal
        for proposal in candidate_pool
        if proposal.description not in used_descriptions
    ]
    if available:
        return available
    available = list(candidate_pool)
    if rng is not None:
        rng.shuffle(available)
    return available


def _candidate_index(
    *,
    candidates: list[CandidateProposal],
    selected: CandidateProposal,
) -> int:
    for index, candidate in enumerate(candidates):
        if (
            candidate.description == selected.description
            and candidate.overrides == selected.overrides
        ):
            return index
    return 0
