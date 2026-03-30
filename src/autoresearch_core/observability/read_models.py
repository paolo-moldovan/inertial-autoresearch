"""Pure read-model assembly helpers for Mission Control views."""

from __future__ import annotations

from typing import Any


def build_current_candidate_pool(current_run: dict[str, Any] | None) -> dict[str, Any] | None:
    """Derive the current candidate-pool panel payload from the active run summary."""
    if current_run is None:
        return None
    return {
        "run_id": current_run.get("run_id"),
        "run_name": current_run.get("run_name"),
        "proposal_source": current_run.get("proposal_source"),
        "policy_mode": current_run.get("policy_mode"),
        "selection_rationale": current_run.get("selection_rationale"),
        "selected_candidate_index": current_run.get("selected_candidate_index"),
        "preferred_candidate_index": current_run.get("preferred_candidate_index"),
        "preferred_candidate_description": current_run.get("preferred_candidate_description"),
        "hermes_status": current_run.get("hermes_status"),
        "hermes_reason": current_run.get("hermes_reason"),
        "candidate_count": len(current_run.get("candidate_pool") or []),
        "candidates": list(current_run.get("candidate_pool") or []),
        "blocked_candidates": dict(current_run.get("blocked_candidates") or {}),
    }


def build_mission_control_summary_payload(
    *,
    loop_state: dict[str, Any] | None,
    current_run: dict[str, Any] | None,
    best_result: dict[str, Any] | None,
    leaderboard: list[dict[str, Any]],
    progress: list[dict[str, Any]],
    queued_proposals: list[dict[str, Any]],
    recent_loop_events: list[dict[str, Any]],
    recent_decisions: list[dict[str, Any]],
    recent_llm_calls: list[dict[str, Any]],
    regime_fingerprint: str | None,
    comparison_metric_key: str,
    mutation_leaderboard: list[dict[str, Any]],
    recent_mutation_lessons: list[dict[str, Any]],
    hermes_runtime: dict[str, Any] | None,
    analytics: dict[str, Any],
    multi_loop_analytics: dict[str, Any],
) -> dict[str, Any]:
    """Build the top-level Mission Control summary payload used by UI surfaces."""
    return {
        "loop_state": loop_state,
        "current_run": current_run,
        "current_candidate_pool": build_current_candidate_pool(current_run),
        "best_result": best_result,
        "leaderboard": leaderboard,
        "progress": progress,
        "queued_proposals": queued_proposals,
        "recent_loop_events": recent_loop_events,
        "recent_decisions": recent_decisions,
        "recent_llm_calls": recent_llm_calls,
        "regime_fingerprint": regime_fingerprint,
        "comparison_metric_key": comparison_metric_key,
        "mutation_leaderboard": mutation_leaderboard,
        "recent_mutation_lessons": recent_mutation_lessons,
        "hermes_runtime": hermes_runtime,
        "analytics": analytics,
        "multi_loop_analytics": multi_loop_analytics,
    }
