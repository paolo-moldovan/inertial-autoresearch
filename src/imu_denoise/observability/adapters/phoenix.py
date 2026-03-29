"""Sync mission-control traces into a Phoenix-compatible OTLP endpoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from imu_denoise.config.schema import ObservabilityConfig
from imu_denoise.observability.queries import MissionControlQueries


def _to_ns(timestamp_s: float | None) -> int | None:
    if timestamp_s is None:
        return None
    return int(timestamp_s * 1_000_000_000)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_attribute(value: Any) -> bool | int | float | str:
    if isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


@dataclass(frozen=True)
class _PhoenixBackend:
    tracer: Any
    provider: Any
    set_span_in_context: Callable[[Any], Any]


def _build_backend(config: ObservabilityConfig) -> _PhoenixBackend:
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "OpenTelemetry exporters are not installed. Install "
            "`imu-denoise[monitor-adapters]` to sync traces to Phoenix."
        ) from exc

    resource = Resource.create(
        {
            "service.name": "imu-mission-control",
            "phoenix.project_name": config.phoenix_project_name,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=config.phoenix_endpoint)
    processor_cls = BatchSpanProcessor if config.phoenix_batch else SimpleSpanProcessor
    provider.add_span_processor(processor_cls(exporter))
    tracer = provider.get_tracer("imu_denoise.observability.phoenix")
    return _PhoenixBackend(
        tracer=tracer,
        provider=provider,
        set_span_in_context=trace.set_span_in_context,
    )


class PhoenixExporter:
    """Replay mission-control run state as traces for Phoenix."""

    def __init__(
        self,
        *,
        config: ObservabilityConfig,
        queries: MissionControlQueries,
        backend_factory: Callable[[ObservabilityConfig], _PhoenixBackend] | None = None,
    ) -> None:
        self.config = config
        self.queries = queries
        self._backend_factory = backend_factory or _build_backend

    def sync(self, *, run_id: str | None = None, limit: int = 100) -> dict[str, int]:
        backend = self._backend_factory(self.config)
        run_rows = self._select_runs(run_id=run_id, limit=limit)
        counts = {
            "runs_seen": len(run_rows),
            "runs_synced": 0,
            "llm_spans": 0,
            "tool_spans": 0,
            "decision_spans": 0,
            "memory_events": 0,
            "skill_events": 0,
        }
        for row in run_rows:
            detail = self.queries.get_run_detail(str(row["id"]))
            if detail is None:
                continue
            child_counts = self._sync_run(backend=backend, detail=detail)
            counts["runs_synced"] += 1
            for key, value in child_counts.items():
                counts[key] += value

        force_flush = getattr(backend.provider, "force_flush", None)
        if callable(force_flush):
            force_flush()
        shutdown = getattr(backend.provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
        return counts

    def _select_runs(self, *, run_id: str | None, limit: int) -> list[dict[str, Any]]:
        if run_id is not None:
            detail = self.queries.get_run_detail(run_id)
            return [] if detail is None else [detail["run"]]
        return self.queries.list_runs(limit=limit)

    def _sync_run(
        self,
        *,
        backend: _PhoenixBackend,
        detail: dict[str, Any],
    ) -> dict[str, int]:
        run = detail["run"]
        run_start_ns = _to_ns(_maybe_float(run.get("started_at")))
        run_end_ns = _to_ns(_maybe_float(run.get("ended_at"))) or _to_ns(
            _maybe_float(run.get("heartbeat_at"))
        )
        if run_end_ns is None and run_start_ns is not None:
            run_end_ns = run_start_ns + 1
        run_span = backend.tracer.start_span(
            "mission_control.run",
            start_time=run_start_ns,
        )
        self._set_attributes(
            run_span,
            {
                "mission_control.run_id": run["id"],
                "mission_control.name": run["name"],
                "mission_control.phase": run["phase"],
                "mission_control.dataset": run.get("dataset"),
                "mission_control.model": run.get("model"),
                "mission_control.device": run.get("device"),
                "mission_control.status": run["status"],
                "mission_control.iteration": run.get("iteration"),
            },
        )
        parent_context = backend.set_span_in_context(run_span)

        llm_count = 0
        for llm_call in detail.get("llm_calls", []):
            llm_detail = self.queries.get_llm_call(str(llm_call["id"])) or llm_call
            span = backend.tracer.start_span(
                f"llm:{llm_detail.get('model') or 'unknown'}",
                context=parent_context,
                start_time=_to_ns(_maybe_float(llm_detail.get("created_at"))),
            )
            self._set_attributes(
                span,
                {
                    "llm.provider": llm_detail.get("provider"),
                    "llm.model": llm_detail.get("model"),
                    "llm.base_url": llm_detail.get("base_url"),
                    "llm.status": llm_detail.get("status"),
                    "llm.latency_ms": llm_detail.get("latency_ms"),
                    "llm.session_id": llm_detail.get("session_id"),
                    "llm.reason": llm_detail.get("reason"),
                    "input.value": llm_detail.get("prompt"),
                    "output.value": llm_detail.get("response"),
                    "llm.parsed_payload": llm_detail.get("parsed_payload"),
                },
            )
            span.end(end_time=_to_ns(_maybe_float(llm_detail.get("created_at"))) or run_end_ns)
            llm_count += 1

        tool_count = 0
        for tool_call in detail.get("tool_calls", []):
            span = backend.tracer.start_span(
                f"tool:{tool_call['tool_name']}",
                context=parent_context,
                start_time=_to_ns(_maybe_float(tool_call.get("created_at"))),
            )
            self._set_attributes(
                span,
                {
                    "tool.name": tool_call["tool_name"],
                    "tool.status": tool_call["status"],
                    "tool.args_summary": tool_call.get("args_summary"),
                    "tool.result_summary": tool_call.get("result_summary"),
                    "tool.duration_ms": tool_call.get("duration_ms"),
                    "tool.llm_call_id": tool_call.get("llm_call_id"),
                    "tool.session_id": tool_call.get("session_id"),
                },
            )
            span.end(end_time=_to_ns(_maybe_float(tool_call.get("created_at"))) or run_end_ns)
            tool_count += 1

        decision_count = 0
        for decision in detail.get("decisions", []):
            span = backend.tracer.start_span(
                f"decision:{decision['status']}",
                context=parent_context,
                start_time=_to_ns(_maybe_float(decision.get("created_at"))),
            )
            self._set_attributes(
                span,
                {
                    "decision.iteration": decision["iteration"],
                    "decision.source": decision["proposal_source"],
                    "decision.description": decision["description"],
                    "decision.status": decision["status"],
                    "decision.metric_key": decision["metric_key"],
                    "decision.metric_value": decision.get("metric_value"),
                    "decision.reason": decision.get("reason"),
                    "decision.overrides": decision.get("overrides"),
                },
            )
            span.end(end_time=_to_ns(_maybe_float(decision.get("created_at"))) or run_end_ns)
            decision_count += 1

        memory_event_count = 0
        for memory_event in self.queries.list_memory_events(run_id=str(run["id"]), limit=500):
            run_span.add_event(
                f"memory:{memory_event['event_type']}",
                attributes={
                    "memory.key_name": _normalize_attribute(memory_event.get("key_name")),
                    "memory.summary": _normalize_attribute(memory_event.get("summary")),
                    "memory.item_count": _normalize_attribute(memory_event.get("item_count")),
                },
                timestamp=_to_ns(_maybe_float(memory_event.get("created_at"))),
            )
            memory_event_count += 1

        skill_event_count = 0
        for skill_event in self.queries.list_skill_events(run_id=str(run["id"]), limit=500):
            run_span.add_event(
                "skill",
                attributes={
                    "skill.status": _normalize_attribute(skill_event.get("status")),
                    "skill.summary": _normalize_attribute(skill_event.get("summary")),
                    "skill.requested": _normalize_attribute(skill_event.get("requested")),
                    "skill.resolved": _normalize_attribute(skill_event.get("resolved")),
                    "skill.missing": _normalize_attribute(skill_event.get("missing")),
                },
                timestamp=_to_ns(_maybe_float(skill_event.get("created_at"))),
            )
            skill_event_count += 1

        run_span.end(end_time=run_end_ns)
        return {
            "llm_spans": llm_count,
            "tool_spans": tool_count,
            "decision_spans": decision_count,
            "memory_events": memory_event_count,
            "skill_events": skill_event_count,
        }

    def _set_attributes(self, span: Any, attributes: dict[str, Any]) -> None:
        for key, value in attributes.items():
            if value is None:
                continue
            span.set_attribute(key, _normalize_attribute(value))
