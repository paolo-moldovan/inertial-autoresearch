"""Selection-state helpers for one autoresearch iteration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

from autoresearch_core import recent_policy_results

from . import selection as selection_helpers

if TYPE_CHECKING:
    from autoresearch_core.providers.hermes import HermesQueryTrace
    from imu_denoise.autoresearch.mutations import MutationProposal
    from imu_denoise.config import ExperimentConfig


@dataclass(frozen=True)
class IterationSelection:
    """Resolved proposal-selection state for one loop iteration."""

    proposal: MutationProposal
    proposal_source: str
    hermes_trace: HermesQueryTrace | None
    preferred_candidate_index: int | None
    candidate_pool: list[MutationProposal] | None
    blocked_candidates: dict[str, list[str]]
    policy_decision: Any | None
    policy_candidate_payloads: list[dict[str, Any]] | None
    queue_row: dict[str, Any] | None
    incumbent_summary: dict[str, object] | None
    incumbent_config_for_policy: dict[str, Any] | None


def resolve_iteration_selection(
    *,
    loop_controller: Any,
    loop_run_id: str,
    iteration: int,
    base_config: ExperimentConfig,
    base_overrides: list[str],
    mutation_catalog: list[MutationProposal],
    adapter: Any,
    queries: Any,
    rng: Random,
    results: list[Any],
    fallback_proposal: MutationProposal,
    hermes_used_descriptions: set[str],
    best_run_id: str | None,
) -> IterationSelection:
    from imu_denoise.autoresearch.mutations import (
        MutationPolicyCandidate,
        MutationProposal,
        choose_policy_candidate,
    )
    from imu_denoise.observability.lineage import (
        build_change_items,
        build_mutation_signatures,
        data_regime_fingerprint,
    )

    queue_row = None
    candidate_pool: list[MutationProposal] | None = None
    blocked_candidates: dict[str, list[str]] = {}
    policy_decision: Any | None = None
    policy_candidate_payloads: list[dict[str, Any]] | None = None
    incumbent_summary: dict[str, object] | None = None
    incumbent_config_for_policy = (
        queries.get_run_config_payload(best_run_id) if best_run_id is not None else None
    )

    if iteration > 0:
        queue_row = loop_controller.claim_next_queued_proposal(loop_run_id=loop_run_id)
    if queue_row is not None:
        proposal = MutationProposal(
            description=str(queue_row["description"]),
            overrides=list(queue_row["overrides"]),
        )
        return IterationSelection(
            proposal=proposal,
            proposal_source="human-queued",
            hermes_trace=None,
            preferred_candidate_index=None,
            candidate_pool=None,
            blocked_candidates={},
            policy_decision=None,
            policy_candidate_payloads=None,
            queue_row=queue_row,
            incumbent_summary=None,
            incumbent_config_for_policy=incumbent_config_for_policy,
        )

    if best_run_id is not None:
        incumbent_reference = queries.get_run_reference(best_run_id)
        if incumbent_reference is not None:
            incumbent_summary = {
                "run_id": str(incumbent_reference["run_id"]),
                "run_name": str(incumbent_reference["run_name"]),
                "model": str(incumbent_reference["model"]),
                "phase": str(incumbent_reference["phase"]),
                "metric_value": best_run_id and queries.get_run_metric(
                    best_run_id,
                    metric_key=base_config.autoresearch.metric_key,
                ),
            }
    mutation_lessons = queries.list_recent_mutation_lessons(
        limit=6,
        regime_fingerprint=data_regime_fingerprint(base_config),
    )
    (
        candidate_pool,
        blocked_candidates,
        preferred_candidate_index,
        proposal_source,
        hermes_trace,
    ) = selection_helpers.select_mutation_proposal(
        root=Path(__file__).resolve().parents[3],
        iteration=iteration,
        base_config=base_config,
        base_overrides=base_overrides,
        mutation_catalog=mutation_catalog,
        resolve_iteration_config_fn=adapter.resolve_iteration_config,
        rng=rng,
        results=results,
        fallback_proposal=fallback_proposal,
        hermes_used_descriptions=hermes_used_descriptions,
        incumbent_summary=incumbent_summary,
        incumbent_config=incumbent_config_for_policy,
        mutation_lessons=mutation_lessons,
    )
    policy_candidates: list[MutationPolicyCandidate] = []
    policy_candidate_payloads = []
    current_regime_fingerprint = data_regime_fingerprint(base_config)
    for candidate_index, candidate in enumerate(candidate_pool):
        candidate_config = adapter.resolve_iteration_config(
            base_config=base_config,
            base_overrides=base_overrides,
            proposal_overrides=list(candidate.overrides),
            incumbent_config=incumbent_config_for_policy,
        )
        candidate_change_items = build_change_items(
            current_config=candidate_config,
            reference_config=(
                incumbent_config_for_policy
                if incumbent_config_for_policy is not None
                else base_config
            ),
            overrides=list(candidate.overrides),
        )
        candidate_signatures = build_mutation_signatures(candidate_change_items)
        candidate_regime_fingerprint = data_regime_fingerprint(candidate_config)
        signature_stats = queries.get_mutation_stats_for_signatures(
            signatures=[str(item["signature"]) for item in candidate_signatures],
            regime_fingerprint=candidate_regime_fingerprint,
        )
        policy_candidates.append(
            MutationPolicyCandidate(
                proposal=candidate,
                signatures=[str(item["signature"]) for item in candidate_signatures],
                stats=[
                    signature_stats[str(item["signature"])]
                    for item in candidate_signatures
                    if str(item["signature"]) in signature_stats
                ],
                hermes_preferred=(
                    preferred_candidate_index is not None
                    and candidate_index == preferred_candidate_index
                ),
                regime_compatible=(candidate_regime_fingerprint == current_regime_fingerprint),
            )
        )
        policy_candidate_payloads.append(
            {
                "description": candidate.description,
                "overrides": list(candidate.overrides),
                "signatures": [str(item["signature"]) for item in candidate_signatures],
                "regime_fingerprint": candidate_regime_fingerprint,
                "regime_compatible": candidate_regime_fingerprint == current_regime_fingerprint,
                "hermes_preferred": (
                    preferred_candidate_index is not None
                    and candidate_index == preferred_candidate_index
                ),
            }
        )
    policy_decision = choose_policy_candidate(
        candidates=policy_candidates,
        strategy=base_config.autoresearch.strategy,
        recent_results=recent_policy_results(results),
        rng=rng,
    )
    assert policy_decision is not None
    proposal = policy_decision.selected
    if proposal_source == "hermes":
        hermes_used_descriptions.add(proposal.description)
    return IterationSelection(
        proposal=proposal,
        proposal_source=proposal_source,
        hermes_trace=hermes_trace,
        preferred_candidate_index=preferred_candidate_index,
        candidate_pool=candidate_pool,
        blocked_candidates=blocked_candidates,
        policy_decision=policy_decision,
        policy_candidate_payloads=policy_candidate_payloads,
        queue_row=None,
        incumbent_summary=incumbent_summary,
        incumbent_config_for_policy=incumbent_config_for_policy,
    )
