"""Mutation proposals for local auto-research runs."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from imu_denoise.config.schema import AutoResearchSearchSpaceConfig, AutoResearchStrategyConfig


@dataclass(frozen=True)
class MutationProposal:
    """A config mutation to evaluate in the autoresearch loop."""

    description: str
    overrides: list[str]
    groups: tuple[str, ...] = ()
    architecture_change: bool = False


@dataclass(frozen=True)
class MutationPolicyCandidate:
    """Candidate plus mutation-memory evidence used by the local policy."""

    proposal: MutationProposal
    signatures: list[str]
    stats: list[dict[str, Any]]
    hermes_preferred: bool = False
    regime_compatible: bool = True


@dataclass(frozen=True)
class MutationPolicyScore:
    """Scored view of a mutation candidate."""

    index: int
    proposal: MutationProposal
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
class MutationPolicyDecision:
    """Outcome of the local explore/exploit selector."""

    selected_index: int
    selected: MutationProposal
    mode: str
    stagnating: bool
    explore_probability: float
    scored_candidates: list[MutationPolicyScore]


def choose_policy_candidate(
    *,
    candidates: list[MutationPolicyCandidate],
    strategy: AutoResearchStrategyConfig,
    recent_results: list[dict[str, Any]],
    rng: Random,
) -> MutationPolicyDecision:
    """Choose a candidate using mutation memory plus an explore/exploit policy."""
    if not candidates:
        raise ValueError("No mutation candidates were provided to the policy selector.")

    scored = [
        _score_candidate(index=index, candidate=candidate, strategy=strategy)
        for index, candidate in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item.total_score, item.index))

    stagnating = _is_stagnating(recent_results, patience=strategy.stagnation_patience)
    explore_probability = _explore_probability(strategy=strategy, stagnating=stagnating)
    mode = _select_mode(strategy=strategy, explore_probability=explore_probability, rng=rng)
    if mode == "explore":
        ranked = sorted(scored, key=lambda item: (-item.exploration_score, item.index))
        top_k = max(1, min(strategy.exploit_top_k, len(ranked)))
        selected = ranked[rng.randrange(top_k)]
    else:
        selected = scored[0]

    return MutationPolicyDecision(
        selected_index=selected.index,
        selected=selected.proposal,
        mode=mode,
        stagnating=stagnating,
        explore_probability=explore_probability,
        scored_candidates=scored,
    )


def default_mutation_pool() -> list[MutationProposal]:
    """Return a broader pool of safe config-only experiment mutations."""
    return [
        MutationProposal(
            description="baseline",
            overrides=[],
        ),
        MutationProposal(
            description="lower learning rate",
            overrides=["training.lr=0.0003"],
            groups=("training_core", "optimizer"),
        ),
        MutationProposal(
            description="higher learning rate",
            overrides=["training.lr=0.003"],
            groups=("training_core", "optimizer"),
        ),
        MutationProposal(
            description="much lower learning rate",
            overrides=["training.lr=0.0001"],
            groups=("training_core", "optimizer"),
        ),
        MutationProposal(
            description="moderately higher learning rate",
            overrides=["training.lr=0.002"],
            groups=("training_core", "optimizer"),
        ),
        MutationProposal(
            description="switch to huber loss",
            overrides=["training.loss=huber"],
            groups=("loss",),
        ),
        MutationProposal(
            description="stronger weight decay",
            overrides=["training.weight_decay=0.001"],
            groups=("optimizer", "regularization"),
        ),
        MutationProposal(
            description="lighter weight decay",
            overrides=["training.weight_decay=0.00001"],
            groups=("optimizer", "regularization"),
        ),
        MutationProposal(
            description="no scheduler",
            overrides=["training.scheduler=none"],
            groups=("scheduler",),
        ),
        MutationProposal(
            description="plateau scheduler",
            overrides=["training.scheduler=plateau"],
            groups=("scheduler",),
        ),
        MutationProposal(
            description="step scheduler",
            overrides=["training.scheduler=step"],
            groups=("scheduler",),
        ),
        MutationProposal(
            description="larger batch size",
            overrides=["training.batch_size=32"],
            groups=("training_core",),
        ),
        MutationProposal(
            description="smaller batch size",
            overrides=["training.batch_size=8"],
            groups=("training_core",),
        ),
        MutationProposal(
            description="reduced dropout",
            overrides=["model.dropout=0.0"],
            groups=("regularization", "architecture_tuning"),
        ),
        MutationProposal(
            description="higher dropout",
            overrides=["model.dropout=0.3"],
            groups=("regularization", "architecture_tuning"),
        ),
        MutationProposal(
            description="wider current model",
            overrides=["model.hidden_dim=256"],
            groups=("architecture_tuning",),
        ),
        MutationProposal(
            description="narrower current model",
            overrides=["model.hidden_dim=64"],
            groups=("architecture_tuning",),
        ),
        MutationProposal(
            description="deeper current model",
            overrides=["model.num_layers=4"],
            groups=("architecture_tuning",),
        ),
        MutationProposal(
            description="shallower current model",
            overrides=["model.num_layers=1"],
            groups=("architecture_tuning",),
        ),
        MutationProposal(
            description="conv1d baseline",
            overrides=[
                "model.name=conv1d",
                "model.hidden_dim=64",
                "model.num_layers=4",
            ],
            groups=("architecture",),
            architecture_change=True,
        ),
        MutationProposal(
            description="wider conv1d",
            overrides=[
                "model.name=conv1d",
                "model.hidden_dim=128",
                "model.num_layers=5",
                "model.kernel_size=9",
            ],
            groups=("architecture",),
            architecture_change=True,
        ),
        MutationProposal(
            description="small transformer",
            overrides=[
                "model.name=transformer",
                "model.hidden_dim=64",
                "model.num_layers=2",
                "model.num_heads=4",
            ],
            groups=("architecture",),
            architecture_change=True,
        ),
        MutationProposal(
            description="medium transformer",
            overrides=[
                "model.name=transformer",
                "model.hidden_dim=128",
                "model.num_layers=3",
                "model.num_heads=4",
            ],
            groups=("architecture",),
            architecture_change=True,
        ),
        MutationProposal(
            description="deeper lstm",
            overrides=[
                "model.name=lstm",
                "model.hidden_dim=128",
                "model.num_layers=3",
            ],
            groups=("architecture",),
            architecture_change=True,
        ),
        MutationProposal(
            description="causal lstm",
            overrides=[
                "model.name=lstm",
                "model.hidden_dim=128",
                "model.num_layers=2",
                "model.bidirectional=false",
            ],
            groups=("architecture",),
            architecture_change=True,
        ),
        MutationProposal(
            description="enable augmentation",
            overrides=["data.augment=true"],
            groups=("augmentation", "data"),
        ),
        MutationProposal(
            description="disable normalization",
            overrides=["data.normalize=false"],
            groups=("data",),
        ),
    ]


def build_mutation_schedule(
    max_iterations: int,
    rng: Random,
    *,
    include_baseline: bool = True,
) -> list[MutationProposal]:
    """Build a mutation schedule with an optional baseline-first seed proposal."""
    pool = default_mutation_pool()
    baseline = pool[0]
    candidates = pool[1:]
    rng.shuffle(candidates)

    selected = [baseline] if include_baseline else []
    target_length = max_iterations + 1 if include_baseline else max_iterations
    while len(selected) < target_length:
        selected.extend(candidates)
    return selected[:target_length]


def proposal_paths(proposal: MutationProposal) -> list[str]:
    """Return the dotted config paths touched by a proposal."""
    paths: list[str] = []
    for override in proposal.overrides:
        if "=" not in override:
            continue
        path, _value = override.split("=", 1)
        paths.append(path.strip())
    return paths


def filter_mutation_proposals(
    proposals: list[MutationProposal],
    search_space: AutoResearchSearchSpaceConfig,
    *,
    incumbent_model_name: str | None = None,
) -> tuple[list[MutationProposal], dict[str, list[str]]]:
    """Filter proposals against the configured search-space constraints."""
    allowed: list[MutationProposal] = []
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


def _proposal_allowed(
    proposal: MutationProposal,
    search_space: AutoResearchSearchSpaceConfig,
    *,
    incumbent_model_name: str | None = None,
) -> tuple[bool, list[str]]:
    groups = set(proposal.groups)
    paths = proposal_paths(proposal)
    reasons: list[str] = []

    if search_space.architecture_mode == "fixed" and proposal.architecture_change:
        reasons.append("architecture_fixed")
    if search_space.architecture_mode == "tune" and proposal.architecture_change:
        reasons.append("architecture_tune_only")
    proposed_model_name = _proposed_model_name(proposal)
    if (
        search_space.baseline_mode == "exploit"
        and incumbent_model_name
        and proposed_model_name is not None
        and proposed_model_name != incumbent_model_name
    ):
        reasons.append(f"exploit_incumbent_model={incumbent_model_name}")

    deny_groups = {item for item in search_space.deny_groups if item}
    blocked_groups = sorted(groups & deny_groups)
    if blocked_groups:
        reasons.append(f"deny_groups={','.join(blocked_groups)}")

    for path in paths:
        if any(_path_matches_prefix(path, prefix) for prefix in search_space.freeze):
            reasons.append(f"frozen:{path}")
        if any(_path_matches_prefix(path, prefix) for prefix in search_space.deny):
            reasons.append(f"deny:{path}")

    allow_paths = [item for item in search_space.allow if item]
    allow_groups = {item for item in search_space.allow_groups if item}
    if allow_paths or allow_groups:
        allowed_by_group = bool(groups & allow_groups)
        allowed_by_path = all(
            any(_path_matches_prefix(path, prefix) for prefix in allow_paths)
            for path in paths
        ) if paths else False
        if not allowed_by_group and not allowed_by_path:
            reasons.append("outside_allowed_search_space")

    return not reasons, reasons


def _path_matches_prefix(path: str, prefix: str) -> bool:
    normalized_prefix = prefix.strip()
    return path == normalized_prefix or path.startswith(normalized_prefix + ".")


def _proposed_model_name(proposal: MutationProposal) -> str | None:
    for override in proposal.overrides:
        if not override.startswith("model.name="):
            continue
        _key, value = override.split("=", 1)
        return value.strip()
    return None


def _score_candidate(
    *,
    index: int,
    candidate: MutationPolicyCandidate,
    strategy: AutoResearchStrategyConfig,
) -> MutationPolicyScore:
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
            float(item.get("prior_avg_metric_delta", 0.0)) * float(item.get("prior_strength", 0.0))
            for item in known_stats
        ) / max(
            1e-8,
            sum(float(item.get("prior_strength", 0.0)) for item in known_stats),
        )
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
    novelty_score = strategy.novelty_bonus if novelty_count > 0 else 0.0
    if not candidate.regime_compatible:
        return MutationPolicyScore(
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
    if strategy.max_retries_per_signature > 0:
        over_limit = sum(
            max(0, int(item.get("tries", 0)) - strategy.max_retries_per_signature)
            for item in known_stats
        )
        retry_penalty = 0.05 * over_limit
    total_score = avg_metric_delta
    if prior_strength > 0.0:
        total_score += prior_strength * prior_avg_delta
    total_score += strategy.confidence_weight * (confidence - 0.5)
    if prior_strength > 0.0:
        total_score += 0.5 * strategy.confidence_weight * prior_strength * (prior_confidence - 0.5)
    total_score += novelty_score
    total_score -= strategy.discard_penalty * discard_count
    total_score -= strategy.crash_penalty * crash_count
    if prior_strength > 0.0:
        total_score -= strategy.discard_penalty * prior_strength * prior_discard_rate
        total_score -= strategy.crash_penalty * prior_strength * prior_crash_rate
    total_score -= retry_penalty
    reasons: list[str] = []
    if avg_metric_delta != 0.0:
        reasons.append(f"avg_delta={avg_metric_delta:.4f}")
    if confidence > 0:
        reasons.append(f"confidence={confidence:.2f}")
    if prior_strength > 0.0:
        reasons.append(
            "cross_regime_prior="
            f"{prior_avg_delta:.4f}@{prior_strength:.2f}"
        )
    if novelty_count > 0:
        reasons.append(f"novelty+{novelty_score:.2f}")
    if discard_count > 0:
        reasons.append(f"discard_penalty={discard_count}")
    if crash_count > 0:
        reasons.append(f"crash_penalty={crash_count}")
    if retry_penalty > 0:
        reasons.append(f"retry_penalty={retry_penalty:.2f}")
    if candidate.hermes_preferred:
        total_score += strategy.hermes_bonus
        reasons.append(f"hermes_bonus={strategy.hermes_bonus:.2f}")

    exploration_score = novelty_score + max(0.0, strategy.novelty_bonus - 0.03 * total_tries)
    if candidate.hermes_preferred:
        exploration_score += 0.5 * strategy.hermes_bonus
    return MutationPolicyScore(
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


def _explore_probability(
    *,
    strategy: AutoResearchStrategyConfig,
    stagnating: bool,
) -> float:
    probability = strategy.explore_probability
    if stagnating:
        probability += strategy.stagnation_explore_boost
    return max(0.0, min(1.0, probability))


def _select_mode(
    *,
    strategy: AutoResearchStrategyConfig,
    explore_probability: float,
    rng: Random,
) -> str:
    if strategy.mode == "explore":
        return "explore"
    if strategy.mode == "exploit":
        return "exploit"
    return "explore" if rng.random() < explore_probability else "exploit"
