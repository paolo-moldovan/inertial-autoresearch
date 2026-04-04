"""Domain-specific proposal selection helpers for the IMU autoresearch loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autoresearch_core import SupportsRunResult
    from autoresearch_core.providers.hermes import HermesQueryTrace
    from imu_denoise.autoresearch.mutations import MutationProposal
    from imu_denoise.config import ExperimentConfig


def select_mutation_proposal(
    *,
    root: Path,
    iteration: int,
    base_config: ExperimentConfig,
    base_overrides: list[str],
    mutation_catalog: list[MutationProposal] | None = None,
    resolve_iteration_config_fn: Callable[..., ExperimentConfig],
    rng: Random,
    results: Sequence[SupportsRunResult],
    fallback_proposal: MutationProposal,
    hermes_used_descriptions: set[str],
    incumbent_summary: dict[str, object] | None,
    incumbent_config: dict[str, Any] | None,
    mutation_lessons: list[dict[str, object]] | None,
) -> tuple[
    list[MutationProposal],
    dict[str, list[str]],
    int | None,
    str,
    HermesQueryTrace | None,
]:
    """Resolve the effective candidate pool and provider preference for one iteration."""
    from autoresearch_core.engine import resolve_provider_selection, result_snapshot
    from autoresearch_loop.hermes import (
        HermesProposalError,
        choose_mutation_proposal_with_trace,
        hermes_backend_ready,
    )
    from imu_denoise.autoresearch.mutations import default_mutation_pool, filter_mutation_proposals
    from imu_denoise.observability.lineage import model_is_causal

    candidate_pool = list(mutation_catalog or default_mutation_pool())[1:]
    candidate_pool, blocked_candidates = filter_mutation_proposals(
        candidate_pool,
        base_config.autoresearch.search_space,
        incumbent_model_name=(
            str(incumbent_summary["model"])
            if incumbent_summary is not None and incumbent_summary.get("model") is not None
            else None
        ),
    )
    realtime_blocked: dict[str, list[str]] = {}
    if base_config.evaluation.realtime_mode:
        realtime_allowed: list[MutationProposal] = []
        for proposal in candidate_pool:
            candidate_config = resolve_iteration_config_fn(
                base_config=base_config,
                base_overrides=base_overrides,
                proposal_overrides=list(proposal.overrides),
                incumbent_config=incumbent_config,
            )
            if model_is_causal(candidate_config) is False:
                realtime_blocked[proposal.description] = ["realtime_requires_causal_model"]
                continue
            realtime_allowed.append(proposal)
        candidate_pool = realtime_allowed
        blocked_candidates.update(realtime_blocked)
    if not candidate_pool:
        raise RuntimeError(
            "No mutation candidates remain after applying the autoresearch search-space "
            f"constraints. Blocked candidates: {blocked_candidates}"
        )

    selection = resolve_provider_selection(
        iteration=iteration,
        orchestrator=base_config.autoresearch.orchestrator,
        fallback_proposal=fallback_proposal,
        candidate_pool=candidate_pool,
        blocked_candidates=blocked_candidates,
        used_descriptions=hermes_used_descriptions,
        rng=rng,
        provider_ready=lambda: hermes_backend_ready(base_config.autoresearch.hermes, root=root),
        provider_select=lambda available_candidates: choose_mutation_proposal_with_trace(
            config=base_config.autoresearch.hermes,
            iteration=iteration,
            metric_key=base_config.autoresearch.metric_key,
            metric_direction=base_config.autoresearch.metric_direction,
            history=[result_snapshot(result) for result in results],
            candidates=available_candidates,
            incumbent=incumbent_summary,
            search_space={
                **asdict(base_config.autoresearch.search_space),
                "blocked_candidates": blocked_candidates,
            },
            mutation_lessons=mutation_lessons,
            root=root,
        ),
        provider_error=HermesProposalError,
    )
    return (
        selection.candidates,
        selection.blocked_candidates,
        selection.preferred_candidate_index,
        selection.proposal_source,
        selection.provider_trace,
    )
