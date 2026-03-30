"""Tests for mission-control observability storage and import flows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from imu_denoise.config import AutoResearchConfig, ExperimentConfig, ObservabilityConfig
from imu_denoise.observability import (
    BlobStore,
    LoopController,
    MissionControlQueries,
    ObservabilityWriter,
    backfill_observability,
    import_hermes_state,
    redact_payload,
)
from imu_denoise.observability.lineage import data_regime_fingerprint
from imu_denoise.observability.monitor_app import _kill_mission_control_tmux_session
from imu_denoise.observability.store import ObservabilityStore


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="obs-test",
        output_dir=str(tmp_path / "artifacts"),
        log_dir=str(tmp_path / "artifacts" / "logs"),
        autoresearch=AutoResearchConfig(
            results_file=str(tmp_path / "artifacts" / "autoresearch" / "results.tsv")
        ),
        observability=ObservabilityConfig(
            enabled=True,
            db_path=str(tmp_path / "artifacts" / "observability" / "mission_control.db"),
            blob_dir=str(tmp_path / "artifacts" / "observability" / "blobs"),
        ),
    )


def test_blob_store_deduplicates_content(tmp_path: Path) -> None:
    """Content-addressed blobs should deduplicate identical payloads."""
    store = BlobStore(tmp_path / "blobs")
    first = store.write_text("same payload")
    second = store.write_text("same payload")

    assert first == second
    assert store.read_text(first) == "same payload"


def test_redact_payload_redacts_auth_fields() -> None:
    """Secret-like keys and bearer tokens should be redacted."""
    redacted = redact_payload(
        {
            "api_key": "secret",
            "headers": {"Authorization": "Bearer abc123"},
            "prompt": "normal content",
        }
    )

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["headers"]["Authorization"] == "[REDACTED]"
    assert redacted["prompt"] == "normal content"


def test_writer_records_run_llm_decision_and_artifacts(tmp_path: Path) -> None:
    """The writer should persist the core run, trace, decision, and artifact records."""
    config = _config(tmp_path)
    writer = ObservabilityWriter.from_experiment_config(config)
    run_id = writer.start_run(
        name="obs-test",
        phase="autoresearch_loop",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=config,
        source="runtime",
    )
    llm_call_id = writer.record_llm_call(
        run_id=run_id,
        provider="custom",
        model="qwen3.5:latest",
        base_url="http://127.0.0.1:11434/v1",
        status="ok",
        latency_ms=123.0,
        prompt="choose a mutation",
        response='{"candidate_index": 1, "reason": "good"}',
        stdout_text='{"candidate_index": 1, "reason": "good"}',
        stderr_text="",
        parsed_payload={"candidate_index": 1, "reason": "good"},
        command={"argv": ["hermes"]},
        session_id="session-1",
        reason="good",
        source="runtime",
    )
    writer.record_decision(
        run_id=run_id,
        iteration=1,
        proposal_source="hermes",
        description="switch to huber loss",
        status="keep",
        metric_key="val_rmse",
        metric_value=0.1,
        overrides=["training.loss=huber"],
        reason="good",
        llm_call_id=llm_call_id,
        source="runtime",
    )
    writer.record_selection_event(
        run_id=run_id,
        loop_run_id=run_id,
        iteration=1,
        proposal_source="hermes",
        description="switch to huber loss",
        incumbent_run_id=None,
        candidate_count=3,
        rationale="good",
        policy_state={"best_metric": 0.2},
        source="runtime",
    )
    writer.record_change_set(
        run_id=run_id,
        loop_run_id=run_id,
        parent_run_id=None,
        incumbent_run_id=None,
        reference_kind="manual",
        proposal_source="hermes",
        description="switch to huber loss",
        overrides=["training.loss=huber"],
        current_config=config,
        reference_config=None,
        source="runtime",
    )
    writer.record_mutation_outcome(
        run_id=run_id,
        loop_run_id=run_id,
        regime_fingerprint=data_regime_fingerprint(config),
        proposal_source="hermes",
        description="switch to huber loss",
        change_items=[
            {
                "path": "training.loss",
                "category": "training",
                "before": "mse",
                "after": "huber",
            }
        ],
        status="keep",
        metric_key="val_rmse",
        metric_value=0.1,
        incumbent_metric=0.2,
        direction="minimize",
        source="runtime",
    )
    metrics_path = tmp_path / "artifacts" / "obs-test" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text('{"best_val_rmse": 0.1}', encoding="utf-8")
    writer.register_artifact(
        run_id=run_id,
        path=metrics_path,
        artifact_type="training_metrics",
        label="metrics",
        source="runtime",
    )
    writer.finish_run(run_id=run_id, status="completed", summary={"best_val_rmse": 0.1})

    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )
    run_detail = queries.get_run_detail(run_id)

    assert run_detail is not None
    assert run_detail["run"]["status"] == "completed"
    assert len(run_detail["decisions"]) == 1
    assert len(run_detail["llm_calls"]) == 1
    assert run_detail["selection_event"] is not None
    assert run_detail["change_set"] is not None
    assert run_detail["change_set"]["summary"]["change_count"] >= 1
    assert run_detail["mutation_attempts"]
    assert queries.list_mutation_leaderboard(limit=10)
    assert queries.list_recent_mutation_lessons(limit=10)
    assert any(
        artifact["artifact_type"] == "training_metrics"
        for artifact in run_detail["artifacts"]
    )


def test_critical_mutation_writes_raise_but_noncritical_events_do_not(tmp_path: Path) -> None:
    """Critical writes should surface failures while UI-only events stay best-effort."""
    config = _config(tmp_path)
    writer = ObservabilityWriter.from_experiment_config(config)
    run_id = writer.start_run(
        name="obs-critical",
        phase="autoresearch_loop",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=config,
        source="runtime",
    )
    assert writer.store is not None
    store_any: Any = writer.store

    original_insert_attempt = store_any.insert_mutation_attempt
    original_insert_event = store_any.insert_event

    def fail_attempt(**_: object) -> None:
        raise RuntimeError("mutation attempt write failed")

    def fail_event(**_: object) -> None:
        raise RuntimeError("event write failed")

    store_any.insert_mutation_attempt = fail_attempt
    with pytest.raises(RuntimeError, match="Critical observability write failed"):
        writer.record_mutation_outcome(
            run_id=run_id,
            loop_run_id=run_id,
            regime_fingerprint=data_regime_fingerprint(config),
            proposal_source="hermes",
            description="switch to huber loss",
            change_items=[
                {
                    "path": "training.loss",
                    "category": "training",
                    "before": "mse",
                    "after": "huber",
                }
            ],
            status="keep",
            metric_key="val_rmse",
            metric_value=0.1,
            incumbent_metric=0.2,
            direction="minimize",
            source="runtime",
        )

    store_any.insert_mutation_attempt = original_insert_attempt
    store_any.insert_event = fail_event
    writer.append_event(
        run_id=run_id,
        event_type="ui_ping",
        level="INFO",
        title="ping",
        payload={"ok": True},
        source="runtime",
    )
    store_any.insert_event = original_insert_event


def test_cross_regime_mutation_prior_is_advisory_only_when_local_evidence_is_sparse(
    tmp_path: Path,
) -> None:
    """Cross-regime priors should warm-start new regimes without overriding strong local stats."""
    config = _config(tmp_path)
    writer = ObservabilityWriter.from_experiment_config(config)
    run_id = writer.start_run(
        name="obs-prior",
        phase="autoresearch_loop",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=config,
        source="runtime",
    )
    other_regime = "other-regime"
    target_regime = data_regime_fingerprint(config)
    for metric_value, incumbent_metric, status in [(0.18, 0.20, "keep"), (0.25, 0.20, "discard")]:
        writer.record_mutation_outcome(
            run_id=run_id,
            loop_run_id=run_id,
            regime_fingerprint=other_regime,
            proposal_source="hermes",
            description="lower learning rate",
            change_items=[
                {
                    "path": "training.lr",
                    "category": "training",
                    "before": 0.001,
                    "after": 0.0003,
                }
            ],
            status=status,
            metric_key="val_rmse",
            metric_value=metric_value,
            incumbent_metric=incumbent_metric,
            direction="minimize",
            source="runtime",
        )

    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )
    prior_only = queries.get_mutation_stats_for_signatures(
        signatures=["training.lr:0.001->0.0003"],
        regime_fingerprint=target_regime,
    )["training.lr:0.001->0.0003"]
    assert prior_only["prior_strength"] > 0.0
    assert prior_only["evidence_scope"] == "cross_regime_prior"

    for metric_value in (0.19, 0.18, 0.17):
        writer.record_mutation_outcome(
            run_id=run_id,
            loop_run_id=run_id,
            regime_fingerprint=target_regime,
            proposal_source="hermes",
            description="lower learning rate",
            change_items=[
                {
                    "path": "training.lr",
                    "category": "training",
                    "before": 0.001,
                    "after": 0.0003,
                }
            ],
            status="keep",
            metric_key="val_rmse",
            metric_value=metric_value,
            incumbent_metric=0.20,
            direction="minimize",
            source="runtime",
        )
    strong_local = queries.get_mutation_stats_for_signatures(
        signatures=["training.lr:0.001->0.0003"],
        regime_fingerprint=target_regime,
    )["training.lr:0.001->0.0003"]
    assert strong_local.get("prior_strength", 0.0) == 0.0


def test_dead_loop_pid_does_not_block_new_loop_slot(tmp_path: Path) -> None:
    """A stale loop lease with a dead PID should be reclaimed immediately."""
    config = _config(tmp_path)
    writer = ObservabilityWriter.from_experiment_config(config)
    store = ObservabilityStore(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )
    controller = LoopController(store=store, writer=writer)

    stale_run_id = writer.start_run(
        name="stale-loop",
        phase="autoresearch_loop",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=config,
        source="runtime",
    )
    next_run_id = writer.start_run(
        name="fresh-loop",
        phase="autoresearch_loop",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=config,
        source="runtime",
    )

    assert store.acquire_loop_slot(
        loop_run_id=stale_run_id,
        pid=999_999,
        status="running",
        current_iteration=0,
        max_iterations=3,
        batch_size=None,
        pause_after_iteration=None,
        pause_requested=False,
        stop_requested=False,
        terminate_requested=False,
        best_metric=None,
        best_run_id=None,
        active_child_run_id=None,
        heartbeat_at=time.time(),
        updated_at=time.time(),
    ) is None

    controller.initialize_loop(
        loop_run_id=next_run_id,
        max_iterations=3,
        batch_size=None,
        pause_enabled=False,
    )

    stale_state = controller.get_loop_state(stale_run_id)
    next_state = controller.get_loop_state(next_run_id)
    assert stale_state is not None
    assert stale_state["status"] == "terminated"
    assert next_state is not None
    assert next_state["status"] == "running"


def test_monitor_quit_kills_mission_control_tmux_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monitor quit should kill the full Mission Control tmux session."""
    monkeypatch.setenv("TMUX", "1")
    calls: list[list[str]] = []

    class _Result:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    def _run(command: list[str], **kwargs: Any) -> _Result:
        calls.append(command)
        if command[:3] == ["tmux", "display-message", "-p"]:
            return _Result(0, "imu-mission-control-euroc\n")
        return _Result(0, "")

    monkeypatch.setattr("subprocess.run", _run)

    assert _kill_mission_control_tmux_session() is True
    assert ["tmux", "kill-session", "-t", "imu-mission-control-euroc"] in calls


def test_import_hermes_json_session_populates_llm_and_skill_views(tmp_path: Path) -> None:
    """JSON session fallback import should create session, LLM trace, and skill records."""
    config = _config(tmp_path)
    writer = ObservabilityWriter.from_experiment_config(config)
    hermes_home = tmp_path / ".hermes"
    sessions_dir = hermes_home / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_payload = {
        "session_id": "session_123",
        "model": "qwen3.5:latest",
        "base_url": "http://127.0.0.1:11434/v1",
        "platform": "cli",
        "session_start": "2026-03-29T17:02:15.060297",
        "last_updated": "2026-03-29T17:02:32.495609",
        "system_prompt": '[SYSTEM: The user has invoked the "research" skill.]',
        "messages": [
            {"role": "user", "content": "choose a mutation"},
            {"role": "assistant", "content": '{"candidate_index": 2, "reason": "robust"}'},
            {"role": "tool", "tool_name": "memory_search", "content": "found prior run"},
        ],
    }
    (sessions_dir / "session_123.json").write_text(
        json.dumps(session_payload),
        encoding="utf-8",
    )

    counts = import_hermes_state(writer=writer, hermes_home=hermes_home)
    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )

    assert counts["json_sessions"] == 1
    assert any(
        call["session_id"] == "session_123"
        for call in queries.list_recent_llm_calls(limit=10)
    )
    assert any(
        event["summary"] == "skills detected in Hermes session context"
        for event in queries.list_skill_events(limit=10)
    )
    assert any(
        event["key_name"] == "memory_search"
        for event in queries.list_memory_events(limit=10)
    )


def test_backfill_is_idempotent_for_logs_and_results(tmp_path: Path) -> None:
    """Backfill should tolerate repeated runs without duplicating stable records."""
    config = _config(tmp_path)
    logs_dir = Path(config.log_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config.output_dir)
    (logs_dir / "demo.history.jsonl").write_text(
        '{"epoch": 1, "train_loss": 0.2, "val_loss": 0.1, "val_rmse": 0.3, "lr": 0.001}\n',
        encoding="utf-8",
    )
    (logs_dir / "demo.jsonl").write_text(
        (
            '{"timestamp": "2026-03-29T10:53:10.766847+00:00", '
            '"level": "INFO", "logger": "demo", "message": "epoch done"}\n'
        ),
        encoding="utf-8",
    )
    metrics_path = output_dir / "demo" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text('{"best_val_rmse": 0.3}', encoding="utf-8")
    results_path = Path(config.autoresearch.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        "\t".join(
            [
                "timestamp",
                "iteration",
                "run_name",
                "status",
                "proposal_source",
                "metric_key",
                "metric_value",
                "model_name",
                "description",
                "overrides",
                "metrics_path",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "2026-03-29T10:00:00+00:00",
                "0",
                "demo",
                "baseline",
                "static",
                "val_rmse",
                "0.3",
                "lstm",
                "baseline",
                "[]",
                str(metrics_path),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    writer = ObservabilityWriter.from_experiment_config(config)
    first = backfill_observability(config=config, writer=writer)
    second = backfill_observability(config=config, writer=writer)
    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )

    assert first["events"] >= 1
    assert second["events"] >= 1
    decisions = [
        row
        for row in queries.list_recent_decisions(limit=10)
        if row["run_name"] == "demo"
    ]
    assert len(decisions) == 1
