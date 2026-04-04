"""Reusable best-effort writer for Mission Control observability."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from autoresearch_core.observability.logging import MissionControlLogHandler
from autoresearch_core.observability.redaction import redact_payload


def _now_ts() -> float:
    return time.time()


class CoreObservabilityWriter:
    """Reusable best-effort facade over a Mission Control store."""

    def __init__(
        self,
        *,
        config: Any,
        store: Any | None,
        logger: logging.Logger | None = None,
        run_started_event: str = "run_started",
        run_finished_event: str = "run_finished",
        run_status_event: str = "run_status",
        training_epoch_event: str = "training_epoch",
        log_event: str = "log",
        artifact_registered_event: str = "artifact_registered",
        llm_call_event: str = "llm_call",
        decision_event: str = "decision",
    ) -> None:
        self.config = config
        self.store = store
        self.logger = logger or logging.getLogger("autoresearch_core.observability")
        self.run_started_event = run_started_event
        self.run_finished_event = run_finished_event
        self.run_status_event = run_status_event
        self.training_epoch_event = training_epoch_event
        self.log_event = log_event
        self.artifact_registered_event = artifact_registered_event
        self.llm_call_event = llm_call_event
        self.decision_event = decision_event

    @property
    def enabled(self) -> bool:
        return self.store is not None

    def _critical(self, func: Any, /, *, retries: int = 2, **kwargs: Any) -> None:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                func(**kwargs)
                return
            except Exception as exc:  # pragma: no cover - exercised via call sites
                last_exc = exc
                if attempt < retries:
                    self.logger.warning(
                        "Critical observability write failed (attempt %d/%d): %s",
                        attempt + 1,
                        retries + 1,
                        exc,
                    )
                    time.sleep(0.05)
                    continue
                self.logger.error("Critical observability write failed: %s", exc)
        raise RuntimeError(f"Critical observability write failed: {last_exc}")

    def _safe(self, func: Any, /, **kwargs: Any) -> None:
        try:
            func(**kwargs)
        except Exception as exc:
            self.logger.warning("Observability write failed: %s", exc)

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
        experiment_id: str | None = None,
        config: Any | None = None,
        overrides: list[str] | None = None,
        objective_metric: str | None = None,
        objective_direction: str | None = None,
        source: str = "runtime",
        run_id: str | None = None,
    ) -> str:
        del config, overrides, objective_metric, objective_direction
        created_run_id = run_id or self._make_run_id(name=name, phase=phase)
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
                event_type=self.run_started_event,
                level="INFO",
                title=f"{phase} started",
                payload={"name": name, "dataset": dataset, "model": model, "device": device},
                source=source,
                created_at=started_at,
                fingerprint=self._fingerprint(created_run_id, self.run_started_event, source),
            )
        return created_run_id

    def make_run_id(self, *, name: str, phase: str) -> str:
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
            event_type=self.run_finished_event,
            level="INFO",
            title=f"run finished: {status}",
            payload=summary,
            source=source,
            created_at=finished_at,
            fingerprint=self._fingerprint(run_id, self.run_finished_event, status, source),
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
            event_type=self.run_status_event,
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
                self._fingerprint(run_id, self.run_status_event, phase, epoch, message, source)
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
        val_rmse: float | None,
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
            event_type=self.training_epoch_event,
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
            fingerprint=self._fingerprint(run_id, self.training_epoch_event, epoch, source),
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
            event_type=self.artifact_registered_event,
            level="INFO",
            title=label or artifact_type,
            payload={"path": abs_path, "artifact_type": artifact_type, "metadata": metadata},
            source=source,
            created_at=_now_ts(),
            fingerprint=self._fingerprint(run_id, self.artifact_registered_event, abs_path, source),
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
            event_type=self.decision_event,
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
                or self._fingerprint(run_id, self.decision_event, iteration, description)
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
            event_type=self.llm_call_event,
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
            fingerprint=self._fingerprint(run_id, session_id, self.llm_call_event, created_call_id),
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
        return MissionControlLogHandler(self, run_id, log_event_type=self.log_event)

    def read_blob_text(self, ref: str) -> str:
        if self.store is None:
            raise FileNotFoundError(ref)
        return cast(str, self.store.blobs.read_text(ref))

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
        if not bool(getattr(self.config, "capture_raw_llm", False)) and not as_json:
            return None
        sanitized = self._sanitize(payload)
        if as_json:
            return cast(str | None, self.store.blobs.write_json(sanitized))
        if isinstance(sanitized, str):
            return cast(str | None, self.store.blobs.write_text(sanitized))
        return cast(
            str | None,
            self.store.blobs.write_text(json.dumps(sanitized, ensure_ascii=False)),
        )

    def _sanitize(self, payload: Any) -> Any:
        if not bool(getattr(self.config, "redact_secrets", False)):
            return payload
        return redact_payload(payload)

    def _make_run_id(self, *, name: str, phase: str) -> str:
        normalized_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-") or "run"
        normalized_phase = re.sub(r"[^a-zA-Z0-9_.-]+", "-", phase).strip("-") or "phase"
        return f"{normalized_phase}:{normalized_name}:{int(_now_ts() * 1000)}:{uuid4().hex[:8]}"

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
