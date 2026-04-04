"""IMU-domain mutation catalog layered on top of generic policy helpers."""

from __future__ import annotations

from random import Random

from autoresearch_core.contracts import CandidateProposal
from autoresearch_core.policy import (
    PolicyCandidate,
    PolicyDecision,
    PolicyScore,
    filter_candidate_proposals,
)
from autoresearch_core.policy import (
    choose_policy_candidate as _choose_policy_candidate,
)
from autoresearch_core.policy import (
    proposal_paths as _proposal_paths,
)

MutationProposal = CandidateProposal
MutationPolicyCandidate = PolicyCandidate
MutationPolicyDecision = PolicyDecision
MutationPolicyScore = PolicyScore
choose_policy_candidate = _choose_policy_candidate
filter_mutation_proposals = filter_candidate_proposals
proposal_paths = _proposal_paths


def default_mutation_pool() -> list[MutationProposal]:
    """Return a broad IMU-safe pool of config-only experiment mutations."""
    return [
        MutationProposal(description="baseline", overrides=[]),
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
            overrides=["model.name=conv1d", "model.hidden_dim=64", "model.num_layers=4"],
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
            overrides=["model.name=lstm", "model.hidden_dim=128", "model.num_layers=3"],
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
    catalog: list[MutationProposal] | None = None,
) -> list[MutationProposal]:
    """Build a mutation schedule with an optional baseline-first seed proposal."""
    pool = list(catalog or default_mutation_pool())
    baseline = pool[0]
    candidates = pool[1:]
    rng.shuffle(candidates)

    selected = [baseline] if include_baseline else []
    target_length = max_iterations + 1 if include_baseline else max_iterations
    while len(selected) < target_length:
        selected.extend(candidates)
    return selected[:target_length]
