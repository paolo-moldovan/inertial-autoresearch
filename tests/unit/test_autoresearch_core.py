"""Focused unit tests for the reusable autoresearch core helpers."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import pytest

from autoresearch_core import ProjectAdapter
from autoresearch_core.contracts import CandidateProposal
from autoresearch_core.engine import resolve_provider_selection
from autoresearch_core.observability.analytics import (
    compute_loop_analytics,
    compute_multi_loop_analytics,
)
from autoresearch_core.observability.read_models import (
    build_current_candidate_pool,
    build_mission_control_summary_payload,
)


@dataclass
class FakeAdapter:
    """Toy adapter used to exercise the core protocol without IMU imports."""

    def resolve_base_config(
        self,
        *,
        config_paths: list[str],
        base_overrides: list[str],
    ) -> dict[str, Any]:
        return {"config_paths": config_paths, "base_overrides": base_overrides}

    def resolve_iteration_config(
        self,
        *,
        base_config: Any,
        base_overrides: list[str],
        proposal_overrides: list[str],
        incumbent_config: dict[str, Any] | None = None,
        extra_overrides: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "base_config": base_config,
            "base_overrides": base_overrides,
            "proposal_overrides": proposal_overrides,
            "incumbent_config": incumbent_config,
            "extra_overrides": extra_overrides,
        }

    def execute_training_run(
        self,
        *,
        config: Any,
        overrides: list[str],
        metric_key: str,
        parent_run_id: str | None = None,
        iteration: int | None = None,
        run_id: str | None = None,
    ) -> tuple[Any, Any, str]:
        summary = {"metric_key": metric_key, "overrides": overrides, "config": config}
        return config, summary, run_id or "fake-run"

    def execute_baseline_run(self, *args: Any, **kwargs: Any) -> Any:
        return {"args": args, "kwargs": kwargs}

    def get_mutation_catalog(self) -> list[CandidateProposal]:
        return [CandidateProposal(description="baseline", overrides=[])]


def test_project_adapter_protocol_accepts_fake_adapter() -> None:
    adapter = FakeAdapter()
    assert isinstance(adapter, ProjectAdapter)


def test_resolve_provider_selection_prefers_provider_but_falls_back_cleanly() -> None:
    fallback = CandidateProposal(description="fallback", overrides=["training.lr=0.001"])
    candidates = [
        CandidateProposal(description="already-used", overrides=["training.lr=0.01"]),
        fallback,
    ]
    selection = resolve_provider_selection(
        iteration=2,
        orchestrator="hermes",
        fallback_proposal=fallback,
        candidate_pool=candidates,
        blocked_candidates={"forbidden": ["freeze:model.name"]},
        used_descriptions={"already-used"},
        rng=Random(7),
        provider_ready=lambda: False,
        provider_select=None,
    )
    assert selection.proposal_source == "static-fallback"
    assert selection.candidates == [fallback]
    assert selection.preferred_candidate_index == 0
    assert selection.blocked_candidates == {"forbidden": ["freeze:model.name"]}


def test_loop_and_multi_loop_analytics_are_computed_from_read_models() -> None:
    analytics = compute_loop_analytics(
        progress=[
            {"iteration": 0, "metric_value": 0.5},
            {"iteration": 1, "metric_value": 0.4},
            {"iteration": 2, "metric_value": 0.45},
        ],
        decisions=[
            {"proposal_source": "hermes", "status": "keep", "groups": ["architecture"]},
            {"proposal_source": "static-fallback", "status": "discard", "groups": ["optimizer"]},
        ],
        leaderboard=[
            {"model": "conv1d", "decision_status": "keep"},
            {"model": "lstm", "decision_status": "discard"},
        ],
    )
    assert analytics.total_runs == 3
    assert analytics.keep_count == 1
    assert analytics.discard_count == 1
    assert analytics.model_family_wins == {"conv1d": 1}
    assert analytics.time_to_best_iteration == 1
    assert analytics.total_improvement == pytest.approx(0.1)

    multi = compute_multi_loop_analytics(
        [
            {"analytics": analytics.__dict__},
            {
                "analytics": {
                    "total_runs": 2,
                    "keep_count": 0,
                    "discard_count": 1,
                    "crash_count": 1,
                    "source_counts": {"static": 2},
                    "model_family_wins": {"transformer": 1},
                }
            },
        ]
    )
    assert multi.loop_count == 2
    assert multi.total_runs == 5
    assert multi.crash_count == 1
    assert multi.source_counts["hermes"] == 1
    assert multi.source_counts["static-fallback"] == 1
    assert multi.source_counts["static"] == 2
    assert multi.model_family_wins["conv1d"] == 1
    assert multi.model_family_wins["transformer"] == 1


def test_summary_read_model_helpers_keep_payload_assembly_pure() -> None:
    current_run = {
        "run_id": "run-1",
        "run_name": "autoresearch_004",
        "proposal_source": "hermes",
        "policy_mode": "exploit",
        "selection_rationale": "recent conv1d wins",
        "selected_candidate_index": 1,
        "preferred_candidate_index": 0,
        "preferred_candidate_description": "lower learning rate",
        "hermes_status": "ok",
        "hermes_reason": "best tradeoff",
        "candidate_pool": [{"description": "lower learning rate"}],
        "blocked_candidates": {"small transformer": ["architecture_fixed"]},
    }

    pool = build_current_candidate_pool(current_run)
    assert pool is not None
    assert pool["candidate_count"] == 1
    assert pool["candidates"][0]["description"] == "lower learning rate"
    assert pool["blocked_candidates"]["small transformer"] == ["architecture_fixed"]

    summary = build_mission_control_summary_payload(
        loop_state={"status": "running"},
        current_run=current_run,
        best_result={"run_id": "run-1"},
        leaderboard=[],
        progress=[],
        queued_proposals=[],
        recent_loop_events=[],
        recent_decisions=[],
        recent_llm_calls=[],
        regime_fingerprint="regime-1",
        comparison_metric_key="sequence_rmse",
        mutation_leaderboard=[],
        recent_mutation_lessons=[],
        hermes_runtime={"provider": "custom"},
        analytics={"total_runs": 1},
        multi_loop_analytics={"loop_count": 1},
    )
    assert summary["current_candidate_pool"] == pool
    assert summary["comparison_metric_key"] == "sequence_rmse"
