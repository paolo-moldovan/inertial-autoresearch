"""Reusable candidate filtering and explore/exploit scoring."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from autoresearch_core.contracts import CandidateProposal


@dataclass(frozen=True)
class PolicyCandidate:
    """Candidate plus historical evidence used by the local policy."""

    proposal: CandidateProposal
    signatures: list[str]
    stats: list[dict[str, Any]]
    hermes_preferred: bool = False
    regime_compatible: bool = True


@dataclass(frozen=True)
class PolicyScore:
    """Scored view of a candidate used for policy decisions and UI traceability."""

    index: int
    proposal: CandidateProposal
    total_score: float
    exploration_score: float
    novelty_score: float
    hermes_preferred: bool
    signature_count: int
    total_tries: int
    avg_metric_delta: float
    confidence: float
    keep_count: int
    discard_count: int
    crash_count: int
    reasons: list[str]


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of the local explore/exploit selector."""

    selected_index: int
    selected: CandidateProposal
    mode: str
    stagnating: bool
    explore_probability: float
    scored_candidates: list[PolicyScore]


def proposal_paths(proposal: CandidateProposal) -> list[str]:
    """Return dotted override paths touched by a proposal."""
    paths: list[str] = []
    for override in proposal.overrides:
        if "=" not in override:
            continue
        path, _value = override.split("=", 1)
        paths.append(path.strip())
    return paths


def filter_candidate_proposals(
    proposals: list[CandidateProposal],
    search_space: Any,
    *,
    incumbent_model_name: str | None = None,
) -> tuple[list[CandidateProposal], dict[str, list[str]]]:
    """Filter proposals against the configured search-space constraints."""
    allowed: list[CandidateProposal] = []
    blocked: dict[str, list[str]] = {}
    for proposal in proposals:
        is_allowed, reasons = _proposal_allowed(
            proposal,
            search_space,
            incumbent_model_name=incumbent_model_name,
        )
        if is_allowed:
            allowed.append(proposal)
        else:
            blocked[proposal.description] = reasons
    return allowed, blocked


def choose_policy_candidate(
    *,
    candidates: list[PolicyCandidate],
    strategy: Any,
    recent_results: list[dict[str, Any]],
    rng: Random,
) -> PolicyDecision:
    """Choose a candidate using mutation memory plus an explore/exploit policy."""
    if not candidates:
        raise ValueError("No mutation candidates were provided to the policy selector.")

    scored = [
        _score_candidate(index=index, candidate=candidate, strategy=strategy)
        for index, candidate in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item.total_score, item.index))

    stagnating = _is_stagnating(recent_results, patience=int(strategy.stagnation_patience))
    explore_probability = _explore_probability(strategy=strategy, stagnating=stagnating)
    mode = _select_mode(strategy=strategy, explore_probability=explore_probability, rng=rng)
    if mode == "explore":
        ranked = sorted(scored, key=lambda item: (-item.exploration_score, item.index))
        top_k = max(1, min(int(strategy.exploit_top_k), len(ranked)))
        selected = ranked[rng.randrange(top_k)]
    else:
        selected = scored[0]

    return PolicyDecision(
        selected_index=selected.index,
        selected=selected.proposal,
        mode=mode,
        stagnating=stagnating,
        explore_probability=explore_probability,
        scored_candidates=scored,
    )


def _proposal_allowed(
    proposal: CandidateProposal,
    search_space: Any,
    *,
    incumbent_model_name: str | None = None,
) -> tuple[bool, list[str]]:
    groups = set(proposal.groups)
    paths = proposal_paths(proposal)
    reasons: list[str] = []

    architecture_mode = str(getattr(search_space, "architecture_mode", "branch"))
    if architecture_mode == "fixed" and proposal.architecture_change:
        reasons.append("architecture_fixed")
    if architecture_mode == "tune" and proposal.architecture_change:
        reasons.append("architecture_tune_only")
    proposed_model_name = _proposed_model_name(proposal)
    baseline_mode = str(getattr(search_space, "baseline_mode", "branch_from_baseline"))
    if (
        baseline_mode == "exploit"
        and incumbent_model_name
        and proposed_model_name is not None
        and proposed_model_name != incumbent_model_name
    ):
        reasons.append(f"exploit_incumbent_model={incumbent_model_name}")

    deny_groups = {item for item in getattr(search_space, "deny_groups", []) if item}
    blocked_groups = sorted(groups & deny_groups)
    if blocked_groups:
        reasons.append(f"deny_groups={','.join(blocked_groups)}")

    for path in paths:
        if any(
            _path_matches_prefix(path, prefix)
            for prefix in getattr(search_space, "freeze", [])
        ):
            reasons.append(f"frozen:{path}")
        if any(
            _path_matches_prefix(path, prefix)
            for prefix in getattr(search_space, "deny", [])
        ):
            reasons.append(f"deny:{path}")

    allow_paths = [item for item in getattr(search_space, "allow", []) if item]
    allow_groups = {item for item in getattr(search_space, "allow_groups", []) if item}
    if allow_paths or allow_groups:
        allowed_by_group = bool(groups & allow_groups)
        allowed_by_path = (
            all(any(_path_matches_prefix(path, prefix) for prefix in allow_paths) for path in paths)
            if paths
            else False
        )
        if not allowed_by_group and not allowed_by_path:
            reasons.append("outside_allowed_search_space")

    return not reasons, reasons


def _path_matches_prefix(path: str, prefix: str) -> bool:
    normalized_prefix = prefix.strip()
    return path == normalized_prefix or path.startswith(normalized_prefix + ".")


def _proposed_model_name(proposal: CandidateProposal) -> str | None:
    for override in proposal.overrides:
        if not override.startswith("model.name="):
            continue
        _key, value = override.split("=", 1)
        return value.strip()
    return None


def _score_candidate(
    *,
    index: int,
    candidate: PolicyCandidate,
    strategy: Any,
) -> PolicyScore:
    known_stats = candidate.stats
    signature_count = len(candidate.signatures)
    total_tries = sum(int(item.get("tries", 0)) for item in known_stats)
    keep_count = sum(int(item.get("keep_count", 0)) for item in known_stats)
    discard_count = sum(int(item.get("discard_count", 0)) for item in known_stats)
    crash_count = sum(int(item.get("crash_count", 0)) for item in known_stats)
    avg_metric_delta = (
        sum(
            float(item["avg_metric_delta"])
            for item in known_stats
            if isinstance(item.get("avg_metric_delta"), (int, float))
        )
        / max(
            1,
            sum(
                1
                for item in known_stats
                if isinstance(item.get("avg_metric_delta"), (int, float))
            ),
        )
    )
    confidence = (
        sum(float(item.get("confidence", 0.0)) for item in known_stats) / max(1, len(known_stats))
    )
    prior_strength = (
        sum(float(item.get("prior_strength", 0.0)) for item in known_stats)
        / max(
            1,
            sum(1 for item in known_stats if float(item.get("prior_strength", 0.0)) > 0.0),
        )
    ) if known_stats else 0.0
    prior_avg_delta = (
        sum(
            float(item.get("prior_avg_metric_delta", 0.0))
            * float(item.get("prior_strength", 0.0))
            for item in known_stats
        )
        / max(1e-8, sum(float(item.get("prior_strength", 0.0)) for item in known_stats))
    ) if known_stats else 0.0
    prior_confidence = (
        sum(float(item.get("prior_confidence", 0.0)) for item in known_stats)
        / max(
            1,
            sum(1 for item in known_stats if float(item.get("prior_confidence", 0.0)) > 0.0),
        )
    ) if known_stats else 0.0
    prior_discard_rate = (
        sum(float(item.get("prior_discard_rate", 0.0)) for item in known_stats)
        / max(
            1,
            sum(1 for item in known_stats if float(item.get("prior_strength", 0.0)) > 0.0),
        )
    ) if known_stats else 0.0
    prior_crash_rate = (
        sum(float(item.get("prior_crash_rate", 0.0)) for item in known_stats)
        / max(
            1,
            sum(1 for item in known_stats if float(item.get("prior_strength", 0.0)) > 0.0),
        )
    ) if known_stats else 0.0
    known_signatures = {str(item.get("signature")) for item in known_stats}
    novelty_count = sum(
        1 for signature in candidate.signatures if signature not in known_signatures
    )
    novelty_score = float(strategy.novelty_bonus) if novelty_count > 0 else 0.0
    if not candidate.regime_compatible:
        return PolicyScore(
            index=index,
            proposal=candidate.proposal,
            total_score=-1_000_000.0,
            exploration_score=-1_000_000.0,
            novelty_score=0.0,
            hermes_preferred=candidate.hermes_preferred,
            signature_count=signature_count,
            total_tries=total_tries,
            avg_metric_delta=avg_metric_delta,
            confidence=confidence,
            keep_count=keep_count,
            discard_count=discard_count,
            crash_count=crash_count,
            reasons=["regime_incompatible"],
        )
    retry_penalty = 0.0
    if int(strategy.max_retries_per_signature) > 0:
        over_limit = sum(
            max(0, int(item.get("tries", 0)) - int(strategy.max_retries_per_signature))
            for item in known_stats
        )
        retry_penalty = 0.05 * over_limit
    total_score = avg_metric_delta
    if prior_strength > 0.0:
        total_score += prior_strength * prior_avg_delta
    total_score += float(strategy.confidence_weight) * (confidence - 0.5)
    if prior_strength > 0.0:
        total_score += 0.5 * float(strategy.confidence_weight) * prior_strength * (
            prior_confidence - 0.5
        )
    total_score += novelty_score
    total_score -= float(strategy.discard_penalty) * discard_count
    total_score -= float(strategy.crash_penalty) * crash_count
    if prior_strength > 0.0:
        total_score -= float(strategy.discard_penalty) * prior_strength * prior_discard_rate
        total_score -= float(strategy.crash_penalty) * prior_strength * prior_crash_rate
    total_score -= retry_penalty
    reasons: list[str] = []
    if avg_metric_delta != 0.0:
        reasons.append(f"avg_delta={avg_metric_delta:.4f}")
    if confidence > 0:
        reasons.append(f"confidence={confidence:.2f}")
    if prior_strength > 0.0:
        reasons.append(f"cross_regime_prior={prior_avg_delta:.4f}@{prior_strength:.2f}")
    if novelty_count > 0:
        reasons.append(f"novelty+{novelty_score:.2f}")
    if discard_count > 0:
        reasons.append(f"discard_penalty={discard_count}")
    if crash_count > 0:
        reasons.append(f"crash_penalty={crash_count}")
    if retry_penalty > 0:
        reasons.append(f"retry_penalty={retry_penalty:.2f}")
    if candidate.hermes_preferred:
        total_score += float(strategy.hermes_bonus)
        reasons.append(f"hermes_bonus={float(strategy.hermes_bonus):.2f}")

    exploration_score = novelty_score + max(0.0, float(strategy.novelty_bonus) - 0.03 * total_tries)
    if candidate.hermes_preferred:
        exploration_score += 0.5 * float(strategy.hermes_bonus)
    return PolicyScore(
        index=index,
        proposal=candidate.proposal,
        total_score=total_score,
        exploration_score=exploration_score,
        novelty_score=novelty_score,
        hermes_preferred=candidate.hermes_preferred,
        signature_count=signature_count,
        total_tries=total_tries,
        avg_metric_delta=avg_metric_delta,
        confidence=confidence,
        keep_count=keep_count,
        discard_count=discard_count,
        crash_count=crash_count,
        reasons=reasons,
    )


def _is_stagnating(recent_results: list[dict[str, Any]], *, patience: int) -> bool:
    if patience <= 0:
        return False
    completed = [
        result
        for result in recent_results
        if result.get("status") in {"keep", "discard", "baseline"}
    ]
    window = completed[-patience:]
    if len(window) < patience:
        return False
    return not any(result.get("status") == "keep" for result in window)


def _explore_probability(*, strategy: Any, stagnating: bool) -> float:
    probability = float(strategy.explore_probability)
    if stagnating:
        probability += float(strategy.stagnation_explore_boost)
    return max(0.0, min(1.0, probability))


def _select_mode(*, strategy: Any, explore_probability: float, rng: Random) -> str:
    mode = str(strategy.mode)
    if mode == "explore":
        return "explore"
    if mode == "exploit":
        return "exploit"
    return "explore" if rng.random() < explore_probability else "exploit"
