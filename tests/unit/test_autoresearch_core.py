"""Focused unit tests for the reusable autoresearch core helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from random import Random
from typing import Any

import pytest

from autoresearch_core import ProjectAdapter, RunResult, recent_policy_results, result_snapshot
from autoresearch_core.contracts import CandidateProposal
from autoresearch_core.engine import (
    LoopProgressState,
    initialize_progress_state,
    resolve_loop_control,
    resolve_provider_selection,
    run_loop_schedule,
)
from autoresearch_core.observability.analytics import (
    compute_loop_analytics,
    compute_multi_loop_analytics,
)
from autoresearch_core.observability.queries import CoreMissionControlQueries
from autoresearch_core.observability.read_models import (
    build_current_candidate_pool,
    build_current_run_summary,
    build_hermes_runtime_summary,
    build_mission_control_summary_payload,
    build_run_policy_context,
)
from autoresearch_core.observability.services import compose_mission_control_services
from autoresearch_core.training import LoopAwareTrainingControl, WriterBackedTrainingHooks


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


class _FakeBlobStore:
    def read_text(self, ref: str) -> str:
        return f"blob:{ref}"

    def read_json(self, ref: str) -> dict[str, str]:
        return {"ref": ref}


class _FakeQueryStore:
    def __init__(self) -> None:
        self.blobs = _FakeBlobStore()

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        if "FROM decisions WHERE run_id = ?" in query:
            return [
                {
                    "id": "decision-1",
                    "run_id": "run-1",
                    "status": "keep",
                    "metric_key": "sequence_rmse",
                    "metric_value": 0.12,
                    "description": "lower learning rate",
                    "proposal_source": "hermes",
                    "overrides_json": '["training.lr=0.0003"]',
                    "candidates_json": '[{"description":"lower learning rate"}]',
                    "created_at": "2026-04-04T12:00:00+00:00",
                }
            ]
        if "FROM llm_calls WHERE run_id = ?" in query or "FROM llm_calls l" in query:
            return [
                {
                    "id": "llm-1",
                    "run_id": "run-1",
                    "status": "ok",
                    "latency_ms": 321.0,
                    "session_id": "session-1",
                    "reason": "best tradeoff",
                    "parsed_payload_json": '{"candidate_index": 0}',
                }
            ]
        if "FROM tool_calls" in query:
            return []
        if "FROM events" in query and "event_type = ?" in query:
            return [{"payload_json": '{"epoch": 1, "val_rmse": 0.12}'}]
        if "FROM events" in query:
            return []
        if "FROM artifacts WHERE run_id = ?" in query:
            return [
                {
                    "id": "artifact-1",
                    "run_id": "run-1",
                    "artifact_type": "history",
                    "path": "/tmp/fake-history.jsonl",
                    "metadata_json": '{"kind":"history"}',
                }
            ]
        return []

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        if "SELECT e.objective_metric" in query:
            return {"objective_metric": "sequence_rmse"}
        if "COALESCE(rd.metric_value, s.best_metric) AS metric_value" in query:
            return {"metric_value": 0.12}
        if "SELECT\n                r.*," in query and "WHERE r.id = ?" in query:
            return {
                "id": "run-1",
                "name": "autoresearch_001",
                "phase": "training",
                "status": "completed",
                "dataset": "euroc",
                "model": "conv1d",
                "experiment_id": "exp-1",
                "epoch": 1,
                "best_metric": 0.12,
                "last_metric": 0.12,
                "heartbeat_at": "2026-04-04T12:00:00+00:00",
                "status_message": "epoch 1",
            }
        if "SELECT * FROM experiments WHERE id = ?" in query:
            return {
                "id": "exp-1",
                "name": "temporal-euroc",
                "config_json": (
                    '{"evaluation":{"realtime_mode":true,"reconstruction":"hann",'
                    '"metrics":["sequence_rmse","smoothness"]},'
                    '"autoresearch":{"hermes":{"provider":"custom","model":"qwen"}}}'
                ),
                "overrides_json": '["training.lr=0.0003"]',
                "summary_json": '{"objective_metric":"sequence_rmse"}',
            }
        if "SELECT e.config_json" in query:
            return {
                "config_json": (
                    '{"evaluation":{"realtime_mode":true,"reconstruction":"hann",'
                    '"metrics":["sequence_rmse"]},'
                    '"autoresearch":{"hermes":{"provider":"custom","model":"qwen"}}}'
                )
            }
        if "SELECT * FROM change_sets WHERE run_id = ?" in query:
            return {
                "id": "changeset-1",
                "run_id": "run-1",
                "parent_run_id": "run-parent",
                "incumbent_run_id": "run-incumbent",
                "overrides_json": '["training.lr=0.0003"]',
                "change_items_json": (
                    '[{"path":"training.lr","category":"training","before":0.001,"after":0.0003}]'
                ),
                "summary_json": '{"change_count":1}',
            }
        if "SELECT * FROM selection_events WHERE run_id = ?" in query:
            return {
                "id": "selection-1",
                "run_id": "run-1",
                "proposal_source": "hermes",
                "description": "lower learning rate",
                "rationale": "recent wins",
                "policy_state_json": (
                    '{"policy_mode":"exploit","selected_candidate_index":0,'
                    '"preferred_candidate_index":0,'
                    '"preferred_candidate_description":"lower learning rate",'
                    '"policy_candidates":[{"description":"lower learning rate"}],'
                    '"blocked_candidates":{"small transformer":["architecture_fixed"]},'
                    '"hermes_status":"ok","hermes_reason":"best tradeoff"}'
                ),
            }
        if "r.id AS run_id" in query and "e.name AS experiment_name" in query:
            return {
                "run_id": "run-1",
                "run_name": "autoresearch_001",
                "phase": "training",
                "run_status": "completed",
                "iteration": 1,
                "experiment_id": "exp-1",
                "regime_fingerprint": "regime-1",
                "experiment_name": "temporal-euroc",
            }
        if (
            "COALESCE(d.metric_value, s.best_metric) AS metric_value" in query
            and "r.model" in query
        ):
            run_id = params[0]
            return {
                "run_id": run_id,
                "run_name": f"name-{run_id}",
                "phase": "training",
                "run_status": "completed",
                "iteration": 1,
                "model": "conv1d",
                "dataset": "euroc",
                "regime_fingerprint": "regime-1",
                "metric_value": 0.12,
                "metric_key": "sequence_rmse",
            }
        if "SELECT * FROM loop_state WHERE loop_run_id = ?" in query:
            return {"loop_run_id": "loop-1", "active_child_run_id": "run-1"}
        if "FROM runs" in query and "WHERE parent_run_id = ?" in query:
            return {"id": "run-1"}
        return None

    def fetch_active_loop_state(self) -> dict[str, Any] | None:
        return None

    def fetch_latest_loop_state(self) -> dict[str, Any] | None:
        return None

    def fetch_queued_proposals(self, *, loop_run_id: str) -> list[dict[str, Any]]:
        return []


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


def test_generic_loop_progress_and_control_helpers_cover_pause_stop_and_terminate() -> None:
    state = initialize_progress_state(baseline_metric=0.25, baseline_run_id="baseline-1")
    assert state.best_metric == pytest.approx(0.25)
    assert state.best_run_id == "baseline-1"
    assert state.results == []
    assert state.provider_used_descriptions == set()

    paused = resolve_loop_control(
        loop_state={"pause_requested": True, "pause_after_iteration": None},
        completed_iterations=1,
    )
    assert paused.should_pause is True
    assert paused.pause_reason == "manual"

    batched = resolve_loop_control(
        loop_state={"pause_requested": False, "pause_after_iteration": 2},
        completed_iterations=2,
    )
    assert batched.should_pause is True
    assert batched.pause_reason == "batch"

    stopped = resolve_loop_control(
        loop_state={"stop_requested": True},
        completed_iterations=3,
    )
    assert stopped.terminal_status == "stopped"
    assert stopped.terminal_message == "stopped after 3 iterations"

    terminated = resolve_loop_control(
        loop_state={"terminate_requested": True},
        completed_iterations=4,
    )
    assert terminated.terminal_status == "terminated"
    assert terminated.terminal_message == "terminated after 4 iterations"


def test_run_result_helpers_build_provider_history_views() -> None:
    result = RunResult(
        iteration=2,
        run_name="autoresearch_002",
        status="keep",
        proposal_source="hermes",
        metric_key="sequence_rmse",
        metric_value=0.123,
        model_name="conv1d",
        description="lower learning rate",
        overrides=["training.lr=0.0003"],
        metrics_path=None,
    )

    snapshot = result_snapshot(result)
    assert snapshot["run_name"] == "autoresearch_002"
    assert snapshot["proposal_source"] == "hermes"
    assert snapshot["overrides"] == ["training.lr=0.0003"]

    policy_history = recent_policy_results([result])
    assert policy_history == [
        {
            "iteration": 2,
            "status": "keep",
            "proposal_source": "hermes",
            "metric_value": 0.123,
            "description": "lower learning rate",
        }
    ]


def test_run_loop_schedule_delegates_control_flow_to_callbacks() -> None:
    progress = initialize_progress_state(baseline_metric=0.5, baseline_run_id="baseline-1")
    persisted: list[str] = []
    finished: list[str] = []

    def _persist_iteration(
        result: Any,
        loop_state: Mapping[str, Any],
        state: LoopProgressState,
    ) -> None:
        del loop_state
        state.results.append(result)
        persisted.append(result)

    def _finish_completed(state: LoopProgressState) -> None:
        del state
        finished.append("completed")

    result = run_loop_schedule(
        schedule=["a", "b"],
        progress_state=progress,
        fetch_loop_state=lambda: {"status": "running"},
        wait_while_paused=lambda state: {"status": "running", "seen": len(state.results)},
        apply_pause=lambda control, loop_state, state: loop_state,
        handle_terminal=lambda control, state: finished.append(control.terminal_status or "none"),
        select_iteration=lambda iteration, fallback, state: {
            "iteration": iteration,
            "fallback": fallback,
            "before": len(state.results),
        },
        prepare_iteration=lambda iteration, selection, loop_state, state: {
            "iteration": iteration,
            "selection": selection,
            "loop_state": loop_state,
        },
        execute_iteration=lambda iteration, selection, prepared, state: (
            f"result-{iteration}",
            0.4 - 0.1 * iteration,
            f"run-{iteration}",
        ),
        persist_iteration=_persist_iteration,
        handle_interrupted=lambda exc, iteration, selection, prepared, loop_state, state: None,
        handle_crash=lambda exc, iteration, selection, prepared, loop_state, state: "crash",
        finish_completed=_finish_completed,
        interrupted_exception_type=RuntimeError,
    )

    assert result == ["result-0", "result-1"]
    assert progress.best_metric == pytest.approx(0.3)
    assert progress.best_run_id == "run-1"
    assert persisted == ["result-0", "result-1"]
    assert finished == ["completed"]


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


def test_current_run_and_hermes_read_models_stay_domain_agnostic() -> None:
    selection_event = {
        "proposal_source": "hermes",
        "description": "lower learning rate",
        "rationale": "recent wins",
        "policy_state": {
            "strategy": {"mode": "adaptive"},
            "policy_mode": "exploit",
            "policy_stagnating": False,
            "policy_explore_probability": 0.15,
            "selected_candidate_index": 1,
            "preferred_candidate_index": 0,
            "preferred_candidate_description": "lower learning rate",
            "blocked_candidates": {"small transformer": ["architecture_fixed"]},
            "hermes_status": "ok",
            "hermes_reason": "best tradeoff",
            "policy_candidates": [{"description": "lower learning rate"}],
        },
    }
    policy_context = build_run_policy_context(selection_event)
    assert policy_context is not None
    assert policy_context["policy_mode"] == "exploit"
    assert policy_context["policy_candidates"][0]["description"] == "lower learning rate"

    current_run = build_current_run_summary(
        current_run_id="run-12345678",
        run={
            "id": "run-12345678",
            "name": "autoresearch_009",
            "phase": "training",
            "status": "running",
            "model": "conv1d",
            "dataset": "euroc",
            "epoch": 3,
            "last_metric": 0.18,
            "best_metric": 0.17,
            "heartbeat_at": "2026-03-31T12:00:00Z",
            "status_message": "epoch 3/10",
        },
        identity={"causal": True},
        latest_decision={
            "status": "keep",
            "description": "lower lr",
            "metric_key": "val_rmse",
            "metric_value": 0.17,
        },
        evaluation_config={
            "realtime_mode": True,
            "reconstruction": "hann",
            "metrics": ["rmse", "smoothness"],
        },
        policy_context=policy_context,
        llm_call_count=2,
        artifact_count=4,
        is_active=True,
    )
    assert current_run["causal"] is True
    assert current_run["realtime_mode"] is True
    assert current_run["blocked_candidates"]["small transformer"] == ["architecture_fixed"]
    assert current_run["run_id_short"] == "run-1234"

    hermes_runtime = build_hermes_runtime_summary(
        hermes_config={
            "provider": "custom",
            "model": "qwen3.5:latest",
            "toolsets": ["file", "memory"],
            "skills": ["imu-autoresearch-policy"],
            "pass_session_id": True,
            "home_dir": ".hermes",
            "max_turns": 3,
        },
        latest_llm={
            "session_id": "sess-1",
            "status": "ok",
            "latency_ms": 4321,
            "reason": "selected",
        },
    )
    assert hermes_runtime is not None
    assert hermes_runtime["toolsets"] == ["file", "memory"]
    assert hermes_runtime["latest_session_id"] == "sess-1"


def test_mission_control_service_bundle_composes_generic_facade() -> None:
    queries = object()
    controller = object()
    writer = object()

    services = compose_mission_control_services(
        queries=queries,
        controller=controller,
        writer=writer,
    )

    assert services.queries is queries
    assert services.controller is controller
    assert services.writer is writer
    assert services.facade.queries is queries
    assert services.facade.controller is controller


def test_generic_training_adapters_delegate_without_domain_imports() -> None:
    calls: list[tuple[str, Any]] = []

    class _Writer:
        def make_run_id(self, *, name: str, phase: str) -> str:
            calls.append(("make_run_id", (name, phase)))
            return f"{phase}:{name}"

        def start_run(self, **kwargs: Any) -> str:
            calls.append(("start_run", kwargs))
            return "run-1"

        def create_log_handler(self, run_id: str) -> Any:
            calls.append(("create_log_handler", run_id))
            return object()

        def update_status(self, **kwargs: Any) -> None:
            calls.append(("update_status", kwargs))

        def append_event(self, **kwargs: Any) -> None:
            calls.append(("append_event", kwargs))

        def register_artifact(self, **kwargs: Any) -> None:
            calls.append(("register_artifact", kwargs))

        def record_epoch(self, **kwargs: Any) -> None:
            calls.append(("record_epoch", kwargs))

        def finish_run(self, **kwargs: Any) -> None:
            calls.append(("finish_run", kwargs))

    hooks = WriterBackedTrainingHooks(_Writer())
    assert hooks.make_run_id(name="trial", phase="training") == "training:trial"
    assert hooks.start_run(name="trial", phase="training") == "run-1"
    hooks.update_status(run_id="run-1", phase="training")
    hooks.record_epoch(
        run_id="run-1",
        epoch=1,
        train_loss=1.0,
        val_loss=0.5,
        val_rmse=0.4,
        lr=1e-3,
        best_metric=0.4,
    )
    hooks.finish_run(run_id="run-1", status="completed")
    assert [name for name, _ in calls[:3]] == ["make_run_id", "start_run", "update_status"]

    loop_states = [
        {
            "terminate_requested": False,
            "current_iteration": 1,
            "max_iterations": 5,
            "batch_size": None,
            "pause_after_iteration": None,
            "pause_requested": False,
            "stop_requested": False,
            "best_run_id": "run-1",
            "status": "running",
        },
        {
            "terminate_requested": True,
            "current_iteration": 1,
            "max_iterations": 5,
            "batch_size": None,
            "pause_after_iteration": None,
            "pause_requested": False,
            "stop_requested": False,
            "best_run_id": "run-1",
            "status": "running",
        },
    ]
    heartbeat_calls: list[dict[str, Any]] = []

    class _Interrupted(RuntimeError):
        def __init__(self, status: str, message: str) -> None:
            super().__init__(message)
            self.status = status

    control = LoopAwareTrainingControl(
        parent_run_id="loop-1",
        active_child_run_id="run-1",
        get_loop_state=lambda _: loop_states.pop(0),
        heartbeat_updater=lambda **kwargs: heartbeat_calls.append(kwargs),
        interrupt_exception_factory=_Interrupted,
    )
    control.heartbeat(best_metric=0.4)
    assert heartbeat_calls[0]["loop_run_id"] == "loop-1"
    with pytest.raises(_Interrupted, match="terminated by control-plane request"):
        control.check_abort()


def test_core_query_helpers_build_generic_run_detail_and_runtime_views() -> None:
    queries = CoreMissionControlQueries(store=_FakeQueryStore())

    assert queries.get_run_objective_metric("run-1") == "sequence_rmse"
    assert queries.get_run_metric("run-1", metric_key="sequence_rmse") == pytest.approx(0.12)

    detail = queries.get_run_detail("run-1")
    assert detail is not None
    assert detail["identity"]["run_id_short"] == "run-1"
    assert detail["change_diff"][0]["before_text"] == "0.001"
    assert detail["lineage"]["parent"]["run_id"] == "run-parent"
    assert detail["policy_context"]["policy_mode"] == "exploit"
    assert detail["curves"][0]["epoch"] == 1

    current_run = queries.get_current_run_summary("loop-1")
    assert current_run is not None
    assert current_run["realtime_mode"] is True
    assert current_run["candidate_pool"][0]["description"] == "lower learning rate"

    hermes = queries.get_hermes_runtime_summary(loop_run_id="loop-1")
    assert hermes is not None
    assert hermes["provider"] == "custom"
    assert hermes["latest_status"] == "ok"
