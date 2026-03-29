"""Mutation proposals for local auto-research runs."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class MutationProposal:
    """A config mutation to evaluate in the autoresearch loop."""

    description: str
    overrides: list[str]


def default_mutation_pool() -> list[MutationProposal]:
    """Return a compact pool of safe config-only experiment mutations."""
    return [
        MutationProposal(
            description="baseline",
            overrides=[],
        ),
        MutationProposal(
            description="lower learning rate",
            overrides=["training.lr=0.0003"],
        ),
        MutationProposal(
            description="higher learning rate",
            overrides=["training.lr=0.003"],
        ),
        MutationProposal(
            description="switch to huber loss",
            overrides=["training.loss=huber"],
        ),
        MutationProposal(
            description="larger batch size",
            overrides=["training.batch_size=32"],
        ),
        MutationProposal(
            description="smaller batch size",
            overrides=["training.batch_size=8"],
        ),
        MutationProposal(
            description="conv1d baseline",
            overrides=[
                "model.name=conv1d",
                "model.hidden_dim=64",
                "model.num_layers=4",
            ],
        ),
        MutationProposal(
            description="small transformer",
            overrides=[
                "model.name=transformer",
                "model.hidden_dim=64",
                "model.num_layers=2",
                "model.num_heads=4",
            ],
        ),
        MutationProposal(
            description="deeper lstm",
            overrides=[
                "model.name=lstm",
                "model.hidden_dim=128",
                "model.num_layers=3",
            ],
        ),
        MutationProposal(
            description="enable augmentation",
            overrides=["data.augment=true"],
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
