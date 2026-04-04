"""Compatibility wrapper for Hermes proposal selection."""

from __future__ import annotations

import subprocess as _subprocess
from pathlib import Path
from typing import Any

from autoresearch_core.providers import hermes as core_hermes
from autoresearch_core.providers.hermes import (
    HermesProposalError,
    HermesQueryTrace,
    _build_prompt,
    hermes_backend_ready,
)

_PROJECT_SKILLS_DIR = Path(__file__).resolve().parent / "hermes_skills"
subprocess = _subprocess


def _run_hermes_query(
    *,
    prompt: str,
    config: Any,
    root: Path,
) -> HermesQueryTrace:
    return core_hermes._run_hermes_query(
        prompt=prompt,
        config=config,
        root=root,
        project_skills_dir=_PROJECT_SKILLS_DIR,
    )


def choose_mutation_proposal_with_trace(
    *,
    config: Any,
    iteration: int,
    metric_key: str,
    metric_direction: str,
    history: list[dict[str, object]],
    candidates: list[Any],
    incumbent: dict[str, object] | None,
    search_space: dict[str, object] | None,
    mutation_lessons: list[dict[str, object]] | None,
    root: Path,
) -> tuple[Any, HermesQueryTrace]:
    return core_hermes.choose_mutation_proposal_with_trace(
        config=config,
        iteration=iteration,
        metric_key=metric_key,
        metric_direction=metric_direction,
        history=history,
        candidates=candidates,
        incumbent=incumbent,
        search_space=search_space,
        mutation_lessons=mutation_lessons,
        root=root,
        project_skills_dir=_PROJECT_SKILLS_DIR,
    )


def choose_mutation_proposal(
    *,
    config: Any,
    iteration: int,
    metric_key: str,
    metric_direction: str,
    history: list[dict[str, object]],
    candidates: list[Any],
    incumbent: dict[str, object] | None,
    search_space: dict[str, object] | None,
    mutation_lessons: list[dict[str, object]] | None,
    root: Path,
) -> Any:
    return core_hermes.choose_mutation_proposal(
        config=config,
        iteration=iteration,
        metric_key=metric_key,
        metric_direction=metric_direction,
        history=history,
        candidates=candidates,
        incumbent=incumbent,
        search_space=search_space,
        mutation_lessons=mutation_lessons,
        root=root,
        project_skills_dir=_PROJECT_SKILLS_DIR,
    )


__all__ = [
    "HermesProposalError",
    "HermesQueryTrace",
    "_PROJECT_SKILLS_DIR",
    "_build_prompt",
    "_run_hermes_query",
    "choose_mutation_proposal",
    "choose_mutation_proposal_with_trace",
    "hermes_backend_ready",
]
