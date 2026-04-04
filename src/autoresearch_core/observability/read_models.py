"""Pure read-model assembly helpers for Mission Control views."""

from __future__ import annotations

from collections.abc import Mapping
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


def build_run_policy_context(selection_event: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a normalized policy-context payload from a selection event."""
    if selection_event is None:
        return None
    policy_state = selection_event.get("policy_state")
    if not isinstance(policy_state, dict):
        return None
    candidates = policy_state.get("policy_candidates")
    blocked_candidates = policy_state.get("blocked_candidates")
    return {
        "proposal_source": selection_event.get("proposal_source"),
        "description": selection_event.get("description"),
        "rationale": selection_event.get("rationale"),
        "strategy": policy_state.get("strategy"),
        "policy_mode": policy_state.get("policy_mode"),
        "stagnating": policy_state.get("policy_stagnating"),
        "explore_probability": policy_state.get("policy_explore_probability"),
        "selected_candidate_index": policy_state.get("selected_candidate_index"),
        "preferred_candidate_index": policy_state.get("preferred_candidate_index"),
        "preferred_candidate_description": policy_state.get("preferred_candidate_description"),
        "blocked_candidates": blocked_candidates if isinstance(blocked_candidates, dict) else {},
        "hermes_status": policy_state.get("hermes_status"),
        "hermes_reason": policy_state.get("hermes_reason"),
        "policy_candidates": candidates if isinstance(candidates, list) else [],
    }


def build_current_run_summary(
    *,
    current_run_id: str,
    run: Mapping[str, Any],
    identity: Mapping[str, Any] | None,
    latest_decision: Mapping[str, Any] | None,
    evaluation_config: Mapping[str, Any] | None,
    policy_context: Mapping[str, Any] | None,
    llm_call_count: int,
    artifact_count: int,
    is_active: bool,
) -> dict[str, Any]:
    """Build the current-run summary payload used by the live UI surfaces."""
    return {
        "run_id": run["id"],
        "run_name": run["name"],
        "phase": run["phase"],
        "status": run["status"],
        "model": run["model"],
        "causal": None if identity is None else identity.get("causal"),
        "dataset": run["dataset"],
        "epoch": run.get("epoch"),
        "last_metric": run.get("last_metric"),
        "best_metric": run.get("best_metric"),
        "heartbeat_at": run.get("heartbeat_at"),
        "status_message": run.get("status_message"),
        "decision_status": None if latest_decision is None else latest_decision.get("status"),
        "decision_description": (
            None if latest_decision is None else latest_decision.get("description")
        ),
        "metric_key": None if latest_decision is None else latest_decision.get("metric_key"),
        "metric_value": None if latest_decision is None else latest_decision.get("metric_value"),
        "realtime_mode": (
            bool(evaluation_config.get("realtime_mode"))
            if isinstance(evaluation_config, Mapping)
            else None
        ),
        "reconstruction": (
            evaluation_config.get("reconstruction")
            if isinstance(evaluation_config, Mapping)
            else None
        ),
        "evaluation_metrics": (
            list(evaluation_config.get("metrics") or [])
            if isinstance(evaluation_config, Mapping)
            else []
        ),
        "proposal_source": (
            None if policy_context is None else policy_context.get("proposal_source")
        ),
        "policy_mode": None if policy_context is None else policy_context.get("policy_mode"),
        "selection_rationale": None if policy_context is None else policy_context.get("rationale"),
        "selected_candidate_index": (
            None if policy_context is None else policy_context.get("selected_candidate_index")
        ),
        "preferred_candidate_index": (
            None if policy_context is None else policy_context.get("preferred_candidate_index")
        ),
        "preferred_candidate_description": (
            None
            if policy_context is None
            else policy_context.get("preferred_candidate_description")
        ),
        "hermes_status": None if policy_context is None else policy_context.get("hermes_status"),
        "hermes_reason": None if policy_context is None else policy_context.get("hermes_reason"),
        "candidate_pool": (
            list(policy_context.get("policy_candidates") or [])
            if isinstance(policy_context, Mapping)
            else []
        ),
        "blocked_candidates": (
            dict(policy_context.get("blocked_candidates") or {})
            if isinstance(policy_context, Mapping)
            else {}
        ),
        "llm_call_count": llm_call_count,
        "artifact_count": artifact_count,
        "is_active": is_active,
        "run_id_short": current_run_id[:8],
    }


def build_hermes_runtime_summary(
    *,
    hermes_config: Mapping[str, Any] | None,
    latest_llm: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a Hermes runtime panel payload from config plus latest LLM trace."""
    if not isinstance(hermes_config, Mapping):
        return None
    return {
        "provider": hermes_config.get("provider"),
        "model": hermes_config.get("model"),
        "toolsets": list(hermes_config.get("toolsets") or []),
        "skills": list(hermes_config.get("skills") or []),
        "pass_session_id": bool(hermes_config.get("pass_session_id")),
        "home_dir": hermes_config.get("home_dir"),
        "max_turns": hermes_config.get("max_turns"),
        "latest_session_id": None if latest_llm is None else latest_llm.get("session_id"),
        "latest_status": None if latest_llm is None else latest_llm.get("status"),
        "latest_latency_ms": None if latest_llm is None else latest_llm.get("latency_ms"),
        "latest_reason": None if latest_llm is None else latest_llm.get("reason"),
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
