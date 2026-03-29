"""Tests for Mission Control Phase 2 sync adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from imu_denoise.config import AutoResearchConfig, ExperimentConfig, ObservabilityConfig
from imu_denoise.observability import MissionControlQueries, ObservabilityWriter, sync_observability
from imu_denoise.observability.adapters.mlflow import MlflowExporter
from imu_denoise.observability.adapters.phoenix import PhoenixExporter, _PhoenixBackend


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="obs-phase2",
        output_dir=str(tmp_path / "artifacts"),
        log_dir=str(tmp_path / "artifacts" / "logs"),
        autoresearch=AutoResearchConfig(
            results_file=str(tmp_path / "artifacts" / "autoresearch" / "results.tsv")
        ),
        observability=ObservabilityConfig(
            enabled=True,
            db_path=str(tmp_path / "artifacts" / "observability" / "mission_control.db"),
            blob_dir=str(tmp_path / "artifacts" / "observability" / "blobs"),
            mlflow_enabled=True,
            phoenix_enabled=True,
        ),
    )


def _seed_run(tmp_path: Path) -> tuple[ExperimentConfig, str]:
    config = _config(tmp_path)
    writer = ObservabilityWriter.from_experiment_config(config)
    run_id = writer.start_run(
        name="phase2-run",
        phase="training",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=config,
        source="runtime",
    )
    writer.record_epoch(
        run_id=run_id,
        epoch=1,
        train_loss=0.2,
        val_loss=0.15,
        val_rmse=0.3,
        lr=0.001,
        best_metric=0.3,
        source="runtime",
    )
    llm_call_id = writer.record_llm_call(
        run_id=run_id,
        provider="custom",
        model="qwen3.5:latest",
        base_url="http://127.0.0.1:11434/v1",
        status="ok",
        latency_ms=88.0,
        prompt="choose a mutation",
        response='{"candidate_index": 0, "reason": "stable"}',
        stdout_text='{"candidate_index": 0, "reason": "stable"}',
        stderr_text="",
        parsed_payload={"candidate_index": 0, "reason": "stable"},
        command={"argv": ["hermes"]},
        session_id="session-42",
        reason="stable",
        source="runtime",
    )
    writer.record_tool_call(
        run_id=run_id,
        llm_call_id=llm_call_id,
        session_id="session-42",
        tool_name="memory_search",
        args_summary="recent denoise runs",
        result_summary="1 prior run",
        duration_ms=12.0,
        status="ok",
        payload={"query": "recent denoise runs"},
        source="runtime",
    )
    writer.record_memory_event(
        run_id=run_id,
        session_id="session-42",
        event_type="search",
        key_name="memory_search",
        item_count=1,
        summary="queried prior experiments",
        payload={"hits": 1},
        source="runtime",
    )
    writer.record_skill_event(
        run_id=run_id,
        session_id="session-42",
        requested=["research"],
        resolved=["research"],
        missing=[],
        status="ok",
        summary="resolved research skill",
        source="runtime",
    )
    writer.record_decision(
        run_id=run_id,
        iteration=1,
        proposal_source="hermes",
        description="keep huber loss",
        status="keep",
        metric_key="val_rmse",
        metric_value=0.3,
        overrides=["training.loss=huber"],
        reason="stable",
        llm_call_id=llm_call_id,
        source="runtime",
    )
    metrics_path = tmp_path / "artifacts" / "phase2-run" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text('{"best_val_rmse": 0.3}', encoding="utf-8")
    writer.register_artifact(
        run_id=run_id,
        path=metrics_path,
        artifact_type="training_metrics",
        label="metrics",
        source="runtime",
    )
    writer.finish_run(run_id=run_id, status="completed", summary={"best_val_rmse": 0.3})
    return config, run_id


@dataclass
class _FakeExperiment:
    experiment_id: str


@dataclass
class _FakeRunInfo:
    run_id: str


class _FakeActiveRun:
    def __init__(self, mlflow: _FakeMlflow, run_id: str) -> None:
        self._mlflow = mlflow
        self.info = _FakeRunInfo(run_id=run_id)

    def __enter__(self) -> _FakeActiveRun:
        self._mlflow.current_run_id = self.info.run_id
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._mlflow.current_run_id = None


class _FakeMlflow:
    def __init__(self) -> None:
        self.current_run_id: str | None = None
        self.tracking_uri: str | None = None
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, dict[str, Any]] = {}

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def get_experiment_by_name(self, name: str) -> _FakeExperiment | None:
        experiment_id = self.experiments.get(name)
        if experiment_id is None:
            return None
        return _FakeExperiment(experiment_id=experiment_id)

    def create_experiment(self, name: str) -> str:
        experiment_id = str(len(self.experiments) + 1)
        self.experiments[name] = experiment_id
        return experiment_id

    def search_runs(
        self,
        *,
        experiment_ids: list[str],
        filter_string: str,
        output_format: str,
        max_results: int,
    ) -> list[SimpleNamespace]:
        _ = output_format
        _ = max_results
        target = filter_string.split("'")[1]
        matches: list[SimpleNamespace] = []
        for run_id, state in self.runs.items():
            if state["experiment_id"] not in experiment_ids:
                continue
            if state["tags"].get("mission_control_run_id") == target:
                matches.append(SimpleNamespace(info=_FakeRunInfo(run_id=run_id)))
        return matches

    def start_run(
        self,
        *,
        run_id: str | None = None,
        experiment_id: str | None = None,
        run_name: str | None = None,
    ) -> _FakeActiveRun:
        active_run_id = run_id or f"mlflow-run-{len(self.runs) + 1}"
        if active_run_id not in self.runs:
            self.runs[active_run_id] = {
                "experiment_id": str(experiment_id),
                "run_name": run_name,
                "tags": {},
                "params": {},
                "metrics": [],
                "dict_artifacts": [],
                "artifacts": [],
            }
        return _FakeActiveRun(self, active_run_id)

    def set_tag(self, key: str, value: str) -> None:
        assert self.current_run_id is not None
        self.runs[self.current_run_id]["tags"][key] = value

    def log_param(self, key: str, value: Any) -> None:
        assert self.current_run_id is not None
        self.runs[self.current_run_id]["params"][key] = value

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        assert self.current_run_id is not None
        self.runs[self.current_run_id]["metrics"].append((key, value, step))

    def log_dict(self, payload: Any, artifact_file: str) -> None:
        assert self.current_run_id is not None
        self.runs[self.current_run_id]["dict_artifacts"].append((artifact_file, payload))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        assert self.current_run_id is not None
        self.runs[self.current_run_id]["artifacts"].append((path, artifact_path))


class _FakeSpan:
    def __init__(self, name: str, *, start_time: int | None, context: Any = None) -> None:
        self.name = name
        self.start_time = start_time
        self.context = context
        self.attributes: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.end_time: int | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        timestamp: int | None = None,
    ) -> None:
        self.events.append(
            {
                "name": name,
                "attributes": attributes or {},
                "timestamp": timestamp,
            }
        )

    def end(self, *, end_time: int | None = None) -> None:
        self.end_time = end_time


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    def start_span(
        self,
        name: str,
        *,
        start_time: int | None = None,
        context: Any = None,
    ) -> _FakeSpan:
        span = _FakeSpan(name=name, start_time=start_time, context=context)
        self.spans.append(span)
        return span


class _FakeProvider:
    def __init__(self) -> None:
        self.flushed = False
        self.shut_down = False

    def force_flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.shut_down = True


def test_queries_include_tool_calls_in_run_detail(tmp_path: Path) -> None:
    config, run_id = _seed_run(tmp_path)
    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )

    detail = queries.get_run_detail(run_id)

    assert detail is not None
    assert len(detail["tool_calls"]) == 1
    assert detail["tool_calls"][0]["tool_name"] == "memory_search"
    assert len(queries.list_tool_calls(run_id=run_id, include_payload=True)) == 1


def test_mlflow_exporter_replays_run_into_mlflow(tmp_path: Path) -> None:
    config, _run_id = _seed_run(tmp_path)
    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )
    fake_mlflow = _FakeMlflow()

    result = MlflowExporter(
        config=config.observability,
        queries=queries,
        mlflow_module=fake_mlflow,
    ).sync(limit=10)

    assert result["runs_synced"] == 1
    assert len(fake_mlflow.runs) == 1
    synced_run = next(iter(fake_mlflow.runs.values()))
    assert synced_run["tags"]["mission_control.phase"] == "training"
    assert any(metric[0] == "val_rmse" for metric in synced_run["metrics"])
    assert any(
        artifact[0] == "mission_control/tool_calls.json"
        for artifact in synced_run["dict_artifacts"]
    )
    assert any(
        "training_metrics" in (artifact_path or "")
        for _path, artifact_path in synced_run["artifacts"]
    )


def test_phoenix_exporter_replays_run_into_trace_spans(tmp_path: Path) -> None:
    config, _run_id = _seed_run(tmp_path)
    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )
    tracer = _FakeTracer()
    provider = _FakeProvider()

    result = PhoenixExporter(
        config=config.observability,
        queries=queries,
        backend_factory=lambda _config: _PhoenixBackend(
            tracer=tracer,
            provider=provider,
            set_span_in_context=lambda span: span,
        ),
    ).sync(limit=10)

    assert result["runs_synced"] == 1
    assert result["llm_spans"] == 1
    assert result["tool_spans"] == 1
    assert result["decision_spans"] == 1
    assert result["memory_events"] == 1
    assert result["skill_events"] == 1
    assert provider.flushed is True
    assert provider.shut_down is True
    assert any(span.name == "mission_control.run" for span in tracer.spans)
    assert any(span.name.startswith("llm:") for span in tracer.spans)


def test_sync_observability_returns_disabled_for_unconfigured_targets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = ExperimentConfig(
        name=config.name,
        device=config.device,
        data=config.data,
        model=config.model,
        training=config.training,
        autoresearch=config.autoresearch,
        observability=ObservabilityConfig(
            **{
                **config.observability.__dict__,
                "mlflow_enabled": False,
                "phoenix_enabled": False,
            }
        ),
        output_dir=config.output_dir,
        log_dir=config.log_dir,
    )

    result = sync_observability(config=config, target="all", limit=5)

    assert result == {
        "mlflow": {"status": "disabled"},
        "phoenix": {"status": "disabled"},
    }
