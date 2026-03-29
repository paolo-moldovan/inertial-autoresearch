"""High-level best-effort writer for mission-control observability."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from imu_denoise.config import ExperimentConfig
from imu_denoise.config.schema import ObservabilityConfig
from imu_denoise.observability.events import (
    ARTIFACT_REGISTERED,
    DECISION_EVENT,
    LLM_CALL_EVENT,
    LOG_EVENT,
    RUN_FINISHED,
    RUN_STARTED,
    RUN_STATUS,
    TRAINING_EPOCH,
)
from imu_denoise.observability.store import ObservabilityStore

_REDACTED = "[REDACTED]"
_KEY_PATTERN = re.compile(r"(?i)(api[_-]?key|authorization|auth[_-]?header|bearer)")
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]+")
_INLINE_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,]+)")


def _now_ts() -> float:
    return time.time()


def redact_text(text: str) -> str:
    """Redact narrow secret patterns while leaving normal prompt content intact."""
    redacted = _BEARER_PATTERN.sub("Bearer " + _REDACTED, text)
    redacted = _INLINE_SECRET_PATTERN.sub(r"\1" + _REDACTED, redacted)
    return redacted


def redact_payload(payload: Any) -> Any:
    """Recursively redact auth-like fields from structured payloads."""
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            if _KEY_PATTERN.search(key):
                redacted[key] = _REDACTED
            else:
                redacted[key] = redact_payload(value)
        return redacted
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, str):
        return redact_text(payload)
    return payload


class ObservabilityLogHandler(logging.Handler):
    """Mirror log records into the mission-control event stream."""

    def __init__(self, writer: ObservabilityWriter, run_id: str) -> None:
        super().__init__()
        self.writer = writer
        self.run_id = run_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info and record.exc_info[1] is not None:
                payload["exception"] = logging.Formatter().formatException(record.exc_info)
            self.writer.append_event(
                run_id=self.run_id,
                event_type=LOG_EVENT,
                level=record.levelname,
                title=record.getMessage()[:120],
                payload=payload,
                source="runtime",
                created_at=record.created,
            )
        except Exception:
            pass


class ObservabilityWriter:
    """Best-effort facade over the SQLite observability store."""

    def __init__(
        self,
        *,
        config: ObservabilityConfig,
        store: ObservabilityStore | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.logger = logger or logging.getLogger("imu_denoise.observability")

    @classmethod
    def from_experiment_config(
        cls,
        config: ExperimentConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> ObservabilityWriter:
        if not config.observability.enabled:
            return cls(config=config.observability, store=None, logger=logger)
        store = ObservabilityStore(
            db_path=Path(config.observability.db_path),
            blob_dir=Path(config.observability.blob_dir),
        )
        return cls(config=config.observability, store=store, logger=logger)

    @property
    def enabled(self) -> bool:
        return self.store is not None

    def ensure_experiment(
        self,
        *,
        config: ExperimentConfig,
        overrides: list[str] | None = None,
        objective_metric: str | None = None,
        objective_direction: str | None = None,
        source: str = "runtime",
        summary: dict[str, Any] | None = None,
    ) -> str | None:
        if self.store is None:
            return None
        config_payload = self._config_payload(config)
        overrides_payload = list(overrides or [])
        experiment_id = sha256(
            json.dumps(
                {
                    "config": config_payload,
                    "overrides": overrides_payload,
                    "objective_metric": objective_metric,
                    "objective_direction": objective_direction,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._safe(
            self.store.upsert_experiment,
            experiment_id=experiment_id,
            name=config.name,
            config_json=config_payload,
            overrides=overrides_payload,
            objective_metric=objective_metric,
            objective_direction=objective_direction,
            summary=summary,
            source=source,
        )
        return experiment_id

    def start_run(
        self,
        *,
        name: str,
        phase: str,
        dataset: str | None = None,
        model: str | None = None,
        device: str | None = None,
        parent_run_id: str | None = None,
        iteration: int | None = None,
        config: ExperimentConfig | None = None,
        overrides: list[str] | None = None,
        objective_metric: str | None = None,
        objective_direction: str | None = None,
        source: str = "runtime",
        run_id: str | None = None,
    ) -> str:
        created_run_id = run_id or self._make_run_id(name=name, phase=phase)
        experiment_id = None
        if config is not None:
            experiment_id = self.ensure_experiment(
                config=config,
                overrides=overrides,
                objective_metric=objective_metric,
                objective_direction=objective_direction,
                source=source,
            )
        started_at = _now_ts()
        if self.store is not None:
            self._safe(
                self.store.upsert_run,
                run_id=created_run_id,
                name=name,
                phase=phase,
                dataset=dataset,
                model=model,
                device=device,
                status="running",
                started_at=started_at,
                ended_at=None,
                parent_run_id=parent_run_id,
                iteration=iteration,
                experiment_id=experiment_id,
                source=source,
            )
            self._safe(
                self.store.upsert_status_snapshot,
                run_id=created_run_id,
                phase=phase,
                epoch=None,
                best_metric=None,
                last_metric=None,
                heartbeat_at=started_at,
                message="started",
            )
            self.append_event(
                run_id=created_run_id,
                event_type=RUN_STARTED,
                level="INFO",
                title=f"{phase} started",
                payload={"name": name, "dataset": dataset, "model": model, "device": device},
                source=source,
                created_at=started_at,
                fingerprint=self._fingerprint(created_run_id, RUN_STARTED, source),
            )
        return created_run_id

    def make_run_id(self, *, name: str, phase: str) -> str:
        """Create a stable unique run identifier without writing any records."""
        return self._make_run_id(name=name, phase=phase)

    def finish_run(
        self,
        *,
        run_id: str,
        status: str,
        summary: dict[str, Any] | None = None,
        source: str = "runtime",
    ) -> None:
        finished_at = _now_ts()
        if self.store is None:
            return
        self._safe(self.store.update_run, run_id=run_id, status=status, ended_at=finished_at)
        message = None
        last_metric = None
        if summary is not None:
            message = summary.get("message") if isinstance(summary.get("message"), str) else None
            metric_value = summary.get("best_val_rmse") or summary.get("rmse")
            if isinstance(metric_value, (int, float)):
                last_metric = float(metric_value)
        self._safe(
            self.store.upsert_status_snapshot,
            run_id=run_id,
            phase=status,
            epoch=None,
            best_metric=last_metric,
            last_metric=last_metric,
            heartbeat_at=finished_at,
            message=message or status,
        )
        self.append_event(
            run_id=run_id,
            event_type=RUN_FINISHED,
            level="INFO",
            title=f"run finished: {status}",
            payload=summary,
            source=source,
            created_at=finished_at,
            fingerprint=self._fingerprint(run_id, RUN_FINISHED, status, source),
        )

    def update_status(
        self,
        *,
        run_id: str,
        phase: str | None,
        epoch: int | None = None,
        best_metric: float | None = None,
        last_metric: float | None = None,
        message: str | None = None,
        source: str = "runtime",
    ) -> None:
        heartbeat_at = _now_ts()
        if self.store is None:
            return
        self._safe(
            self.store.upsert_status_snapshot,
            run_id=run_id,
            phase=phase,
            epoch=epoch,
            best_metric=best_metric,
            last_metric=last_metric,
            heartbeat_at=heartbeat_at,
            message=message,
        )
        self.append_event(
            run_id=run_id,
            event_type=RUN_STATUS,
            level="INFO",
            title=message or (phase or "status"),
            payload={
                "phase": phase,
                "epoch": epoch,
                "best_metric": best_metric,
                "last_metric": last_metric,
            },
            source=source,
            created_at=heartbeat_at,
            fingerprint=(
                self._fingerprint(run_id, RUN_STATUS, phase, epoch, message, source)
                if epoch is not None or message is not None
                else None
            ),
        )

    def record_epoch(
        self,
        *,
        run_id: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_rmse: float,
        lr: float,
        best_metric: float | None,
        source: str = "runtime",
    ) -> None:
        self.update_status(
            run_id=run_id,
            phase="training",
            epoch=epoch,
            best_metric=best_metric,
            last_metric=val_rmse,
            message=f"epoch {epoch}",
            source=source,
        )
        self.append_event(
            run_id=run_id,
            event_type=TRAINING_EPOCH,
            level="INFO",
            title=f"epoch {epoch}",
            payload={
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "lr": lr,
            },
            source=source,
            created_at=_now_ts(),
            fingerprint=self._fingerprint(run_id, TRAINING_EPOCH, epoch, source),
        )

    def register_artifact(
        self,
        *,
        run_id: str | None,
        path: str | Path,
        artifact_type: str,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "runtime",
    ) -> None:
        abs_path = str(Path(path).resolve())
        if self.store is None:
            return
        self._safe(
            self.store.insert_artifact,
            run_id=run_id,
            artifact_type=artifact_type,
            path=abs_path,
            label=label,
            metadata=metadata,
            created_at=_now_ts(),
            source=source,
        )
        self.append_event(
            run_id=run_id,
            event_type=ARTIFACT_REGISTERED,
            level="INFO",
            title=label or artifact_type,
            payload={"path": abs_path, "artifact_type": artifact_type, "metadata": metadata},
            source=source,
            created_at=_now_ts(),
            fingerprint=self._fingerprint(run_id, ARTIFACT_REGISTERED, abs_path, source),
        )

    def record_decision(
        self,
        *,
        run_id: str | None,
        iteration: int | None,
        proposal_source: str,
        description: str,
        status: str,
        metric_key: str,
        metric_value: float | None,
        overrides: list[str],
        candidates: list[dict[str, Any]] | None = None,
        reason: str | None = None,
        llm_call_id: str | None = None,
        source: str = "runtime",
        fingerprint: str | None = None,
    ) -> None:
        if self.store is None:
            return
        created_at = _now_ts()
        self._safe(
            self.store.insert_decision,
            fingerprint=fingerprint,
            run_id=run_id,
            iteration=iteration,
            proposal_source=proposal_source,
            description=description,
            status=status,
            metric_key=metric_key,
            metric_value=metric_value,
            overrides=overrides,
            candidates=candidates,
            reason=reason,
            llm_call_id=llm_call_id,
            created_at=created_at,
            source=source,
        )
        self.append_event(
            run_id=run_id,
            event_type=DECISION_EVENT,
            level="INFO",
            title=description,
            payload={
                "iteration": iteration,
                "proposal_source": proposal_source,
                "status": status,
                "metric_key": metric_key,
                "metric_value": metric_value,
                "reason": reason,
            },
            source=source,
            created_at=created_at,
            fingerprint=(
                fingerprint
                or self._fingerprint(run_id, DECISION_EVENT, iteration, description)
            ),
        )

    def record_llm_call(
        self,
        *,
        run_id: str | None,
        provider: str | None,
        model: str | None,
        base_url: str | None,
        status: str,
        latency_ms: float | None,
        prompt: str | None = None,
        response: str | None = None,
        stdout_text: str | None = None,
        stderr_text: str | None = None,
        parsed_payload: dict[str, Any] | None = None,
        command: dict[str, Any] | None = None,
        session_id: str | None = None,
        reason: str | None = None,
        source: str = "runtime",
        call_id: str | None = None,
    ) -> str:
        created_call_id = call_id or f"llm_{uuid4().hex}"
        if self.store is None:
            return created_call_id
        prompt_ref = self._store_blob(prompt, as_json=False)
        response_ref = self._store_blob(response, as_json=False)
        stdout_ref = self._store_blob(stdout_text, as_json=False)
        stderr_ref = self._store_blob(stderr_text, as_json=False)
        created_at = _now_ts()
        redacted_command = self._sanitize(command)
        redacted_payload = self._sanitize(parsed_payload)
        self._safe(
            self.store.insert_llm_call,
            call_id=created_call_id,
            run_id=run_id,
            provider=provider,
            model=model,
            base_url=base_url,
            status=status,
            latency_ms=latency_ms,
            command=redacted_command,
            parsed_payload=redacted_payload,
            prompt_blob_ref=prompt_ref,
            response_blob_ref=response_ref,
            stdout_blob_ref=stdout_ref,
            stderr_blob_ref=stderr_ref,
            session_id=session_id,
            reason=reason,
            created_at=created_at,
            source=source,
        )
        self.append_event(
            run_id=run_id,
            session_id=session_id,
            event_type=LLM_CALL_EVENT,
            level="INFO" if status == "ok" else "WARNING",
            title=f"{provider or 'llm'}:{model or 'unknown'}",
            payload={
                "call_id": created_call_id,
                "status": status,
                "latency_ms": latency_ms,
                "reason": reason,
            },
            source=source,
            created_at=created_at,
            fingerprint=self._fingerprint(run_id, session_id, LLM_CALL_EVENT, created_call_id),
        )
        return created_call_id

    def record_tool_call(
        self,
        *,
        run_id: str | None,
        llm_call_id: str | None,
        session_id: str | None,
        tool_name: str,
        args_summary: str | None,
        result_summary: str | None,
        duration_ms: float | None,
        status: str,
        payload: Any | None = None,
        source: str = "runtime",
        fingerprint: str | None = None,
    ) -> None:
        if self.store is None:
            return
        payload_ref = self._store_blob(payload, as_json=True)
        self._safe(
            self.store.insert_tool_call,
            fingerprint=fingerprint,
            run_id=run_id,
            llm_call_id=llm_call_id,
            session_id=session_id,
            tool_name=tool_name,
            args_summary=args_summary,
            result_summary=result_summary,
            duration_ms=duration_ms,
            status=status,
            payload_blob_ref=payload_ref,
            created_at=_now_ts(),
            source=source,
        )

    def record_memory_event(
        self,
        *,
        run_id: str | None,
        session_id: str | None,
        event_type: str,
        key_name: str | None,
        item_count: int | None,
        summary: str | None,
        payload: Any | None = None,
        source: str = "runtime",
        fingerprint: str | None = None,
    ) -> None:
        if self.store is None:
            return
        payload_ref = self._store_blob(payload, as_json=True)
        self._safe(
            self.store.insert_memory_event,
            fingerprint=fingerprint,
            run_id=run_id,
            session_id=session_id,
            event_type=event_type,
            key_name=key_name,
            item_count=item_count,
            summary=summary,
            payload_blob_ref=payload_ref,
            created_at=_now_ts(),
            source=source,
        )

    def record_skill_event(
        self,
        *,
        run_id: str | None,
        session_id: str | None,
        requested: list[str] | None,
        resolved: list[str] | None,
        missing: list[str] | None,
        status: str,
        summary: str | None,
        source: str = "runtime",
        fingerprint: str | None = None,
    ) -> None:
        if self.store is None:
            return
        self._safe(
            self.store.insert_skill_event,
            fingerprint=fingerprint,
            run_id=run_id,
            session_id=session_id,
            requested=requested,
            resolved=resolved,
            missing=missing,
            status=status,
            summary=summary,
            created_at=_now_ts(),
            source=source,
        )

    def append_event(
        self,
        *,
        run_id: str | None,
        event_type: str,
        level: str | None,
        title: str | None,
        payload: dict[str, Any] | None,
        source: str,
        created_at: float | None = None,
        session_id: str | None = None,
        fingerprint: str | None = None,
    ) -> None:
        if self.store is None:
            return
        self._safe(
            self.store.insert_event,
            fingerprint=fingerprint,
            run_id=run_id,
            session_id=session_id,
            event_type=event_type,
            level=level,
            title=title,
            payload=self._sanitize(payload),
            created_at=created_at if created_at is not None else _now_ts(),
            source=source,
        )

    def create_log_handler(self, run_id: str) -> logging.Handler:
        return ObservabilityLogHandler(self, run_id)

    def read_blob_text(self, ref: str) -> str:
        if self.store is None:
            raise FileNotFoundError(ref)
        return self.store.blobs.read_text(ref)

    def read_blob_json(self, ref: str) -> Any:
        if self.store is None:
            raise FileNotFoundError(ref)
        payload = self.store.blobs.read_json(ref)
        if isinstance(payload, Mapping):
            return dict(payload)
        return payload

    def store_text_blob(self, content: str) -> str | None:
        return self._store_blob(content, as_json=False)

    def store_json_blob(self, payload: Any) -> str | None:
        return self._store_blob(payload, as_json=True)

    def _store_blob(self, payload: Any | None, *, as_json: bool) -> str | None:
        if self.store is None or payload is None:
            return None
        if not self.config.capture_raw_llm and not as_json:
            return None
        sanitized = self._sanitize(payload)
        if as_json:
            return self.store.blobs.write_json(sanitized)
        if isinstance(sanitized, str):
            return self.store.blobs.write_text(sanitized)
        return self.store.blobs.write_text(json.dumps(sanitized, ensure_ascii=False))

    def _sanitize(self, payload: Any) -> Any:
        if not self.config.redact_secrets:
            return payload
        return redact_payload(payload)

    def _config_payload(self, config: ExperimentConfig) -> dict[str, Any]:
        raw = asdict(config)
        sanitized = self._sanitize(raw)
        if isinstance(sanitized, dict):
            return sanitized
        return raw

    def _make_run_id(self, *, name: str, phase: str) -> str:
        normalized_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-") or "run"
        normalized_phase = re.sub(r"[^a-zA-Z0-9_.-]+", "-", phase).strip("-") or "phase"
        return f"{normalized_phase}:{normalized_name}:{int(_now_ts() * 1000)}:{uuid4().hex[:8]}"

    def _safe(self, func: Any, /, **kwargs: Any) -> None:
        try:
            func(**kwargs)
        except Exception as exc:
            self.logger.warning("Observability write failed: %s", exc)

    @staticmethod
    def _fingerprint(*parts: object) -> str:
        normalized: list[Any] = []
        for part in parts:
            if is_dataclass(part) and not isinstance(part, type):
                normalized.append(asdict(part))
            else:
                normalized.append(part)
        return sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
