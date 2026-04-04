"""Reusable proposal providers for autoresearch loops."""

from autoresearch_core.providers.hermes import (
    HermesProposalError,
    HermesQueryTrace,
    choose_mutation_proposal,
    choose_mutation_proposal_with_trace,
    hermes_backend_ready,
)

__all__ = [
    "HermesProposalError",
    "HermesQueryTrace",
    "choose_mutation_proposal",
    "choose_mutation_proposal_with_trace",
    "hermes_backend_ready",
]
