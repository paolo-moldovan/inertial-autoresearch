"""Tests for mission-control observability storage and import flows."""

from __future__ import annotations

import json
from pathlib import Path

from imu_denoise.config import AutoResearchConfig, ExperimentConfig, ObservabilityConfig
from imu_denoise.observability import (
    BlobStore,
    MissionControlQueries,
    ObservabilityWriter,
    backfill_observability,
    import_hermes_state,
    redact_payload,
)


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
    assert any(
        artifact["artifact_type"] == "training_metrics"
        for artifact in run_detail["artifacts"]
    )


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
