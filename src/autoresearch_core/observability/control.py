"""Reusable Mission Control control-plane helpers."""

from __future__ import annotations

import json
import os
import time
from typing import Any

LOOP_PAUSED = "loop_paused"
LOOP_RESUMED = "loop_resumed"
LOOP_STOP_REQUESTED = "loop_stop_requested"
LOOP_TERMINATE_REQUESTED = "loop_terminate_requested"
LOOP_STOPPED = "loop_stopped"
LOOP_TERMINATED = "loop_terminated"
QUEUE_CLAIMED = "queue_claimed"
QUEUE_APPLIED = "queue_applied"
QUEUE_ENQUEUED = "queue_enqueued"


class LoopAlreadyRunningError(RuntimeError):
    """Raised when a second loop tries to acquire the singleton control plane."""

    def __init__(self, blocking_loop_run_id: str) -> None:
        super().__init__(f"Another loop is already active: {blocking_loop_run_id}")
        self.blocking_loop_run_id = blocking_loop_run_id


def _now_ts() -> float:
    return time.time()


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


class LoopController:
    """Single-loop control plane backed by a Mission Control store."""

    def __init__(
        self,
        *,
        store: Any,
        writer: Any | None = None,
    ) -> None:
        self.store = store
        self.writer = writer

    def _critical_store_write(
        self,
        func: Any,
        /,
        *,
        retries: int = 2,
        **kwargs: Any,
    ) -> Any:
        logger = None if self.writer is None else self.writer.logger
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return func(**kwargs)
            except Exception as exc:  # pragma: no cover - exercised via call sites
                last_exc = exc
                if attempt < retries:
                    if logger is not None:
                        logger.warning(
                            "Critical control-plane write failed (attempt %d/%d): %s",
                            attempt + 1,
                            retries + 1,
                            exc,
                        )
                    time.sleep(0.05)
                    continue
                if logger is not None:
                    logger.error("Critical control-plane write failed: %s", exc)
        raise RuntimeError(f"Critical control-plane write failed: {last_exc}")

    def initialize_loop(
        self,
        *,
        loop_run_id: str,
        max_iterations: int,
        batch_size: int | None,
        pause_enabled: bool,
        current_iteration: int = 0,
        best_metric: float | None = None,
        best_run_id: str | None = None,
    ) -> None:
        pause_after_iteration = None
        if pause_enabled and batch_size is not None:
            pause_after_iteration = current_iteration + batch_size
        blocking_loop_run_id = self._critical_store_write(
            self.store.acquire_loop_slot,
            loop_run_id=loop_run_id,
            pid=os.getpid(),
            status="running",
            current_iteration=current_iteration,
            max_iterations=max_iterations,
            batch_size=batch_size,
            pause_after_iteration=pause_after_iteration,
            pause_requested=False,
            stop_requested=False,
            terminate_requested=False,
            best_metric=best_metric,
            best_run_id=best_run_id,
            active_child_run_id=None,
            heartbeat_at=_now_ts(),
            updated_at=_now_ts(),
        )
        if blocking_loop_run_id is not None:
            raise LoopAlreadyRunningError(blocking_loop_run_id)

    def heartbeat(
        self,
        *,
        loop_run_id: str,
        current_iteration: int,
        max_iterations: int,
        batch_size: int | None,
        pause_after_iteration: int | None,
        pause_requested: bool,
        stop_requested: bool | None = None,
        terminate_requested: bool | None = None,
        best_metric: float | None,
        best_run_id: str | None,
        active_child_run_id: str | None,
        status: str,
    ) -> None:
        now = _now_ts()
        existing = self.get_loop_state(loop_run_id)
        self._critical_store_write(
            self.store.upsert_loop_state,
            loop_run_id=loop_run_id,
            pid=(
                int(existing["pid"])
                if existing is not None and isinstance(existing.get("pid"), int)
                else os.getpid()
            ),
            status=status,
            current_iteration=current_iteration,
            max_iterations=max_iterations,
            batch_size=batch_size,
            pause_after_iteration=pause_after_iteration,
            pause_requested=pause_requested,
            stop_requested=(
                stop_requested
                if stop_requested is not None
                else bool(existing.get("stop_requested")) if existing else False
            ),
            terminate_requested=(
                terminate_requested
                if terminate_requested is not None
                else bool(existing.get("terminate_requested")) if existing else False
            ),
            best_metric=best_metric,
            best_run_id=best_run_id,
            active_child_run_id=active_child_run_id,
            heartbeat_at=now,
            updated_at=now,
        )

    def complete_loop(
        self,
        *,
        loop_run_id: str,
        current_iteration: int,
        max_iterations: int,
        batch_size: int | None,
        best_metric: float | None,
        best_run_id: str | None,
        status: str,
    ) -> None:
        self.heartbeat(
            loop_run_id=loop_run_id,
            current_iteration=current_iteration,
            max_iterations=max_iterations,
            batch_size=batch_size,
            pause_after_iteration=None,
            pause_requested=False,
            stop_requested=False,
            terminate_requested=False,
            best_metric=best_metric,
            best_run_id=best_run_id,
            active_child_run_id=None,
            status=status,
        )

    def get_active_loop_state(self) -> dict[str, Any] | None:
        row = self.store.fetch_active_loop_state()
        return self._normalize_loop_state(row)

    def get_loop_state(self, loop_run_id: str) -> dict[str, Any] | None:
        row = self.store.fetch_loop_state(loop_run_id)
        return self._normalize_loop_state(row)

    def request_pause(self, *, loop_run_id: str | None = None) -> dict[str, Any] | None:
        state = self._resolve_target_loop(loop_run_id)
        if state is None:
            return None
        self._critical_store_write(
            self.store.update_loop_state,
            loop_run_id=str(state["loop_run_id"]),
            values={"pause_requested": 1},
        )
        if self.writer is not None:
            self.writer.append_event(
                run_id=str(state["loop_run_id"]),
                event_type=LOOP_PAUSED,
                level="INFO",
                title="pause requested",
                payload={"source": "control"},
                source="runtime",
            )
        return self.get_loop_state(str(state["loop_run_id"]))

    def resume_loop(self, *, loop_run_id: str | None = None) -> dict[str, Any] | None:
        state = self._resolve_target_loop(loop_run_id, require_paused=True)
        if state is None:
            return None
        batch_size = state.get("batch_size")
        pause_after_iteration = None
        if isinstance(batch_size, int) and batch_size > 0:
            pause_after_iteration = int(state["current_iteration"]) + batch_size
        self._critical_store_write(
            self.store.update_loop_state,
            loop_run_id=str(state["loop_run_id"]),
            values={
                "status": "running",
                "pause_requested": 0,
                "stop_requested": 0,
                "pause_after_iteration": pause_after_iteration,
                "heartbeat_at": _now_ts(),
            },
        )
        if self.writer is not None:
            self.writer.append_event(
                run_id=str(state["loop_run_id"]),
                event_type=LOOP_RESUMED,
                level="INFO",
                title="loop resumed",
                payload={"source": "control"},
                source="runtime",
            )
        return self.get_loop_state(str(state["loop_run_id"]))

    def request_stop(self, *, loop_run_id: str | None = None) -> dict[str, Any] | None:
        state = self._resolve_target_loop(loop_run_id)
        if state is None:
            return None
        self._critical_store_write(
            self.store.update_loop_state,
            loop_run_id=str(state["loop_run_id"]),
            values={"stop_requested": 1},
        )
        if self.writer is not None:
            self.writer.append_event(
                run_id=str(state["loop_run_id"]),
                event_type=LOOP_STOP_REQUESTED,
                level="WARNING",
                title="stop requested",
                payload={"source": "control"},
                source="runtime",
            )
        return self.get_loop_state(str(state["loop_run_id"]))

    def request_terminate(self, *, loop_run_id: str | None = None) -> dict[str, Any] | None:
        state = self._resolve_target_loop(loop_run_id)
        if state is None:
            return None
        self._critical_store_write(
            self.store.update_loop_state,
            loop_run_id=str(state["loop_run_id"]),
            values={
                "stop_requested": 1,
                "terminate_requested": 1,
                "pause_requested": 0,
                "status": "terminating",
                "heartbeat_at": _now_ts(),
            },
        )
        if self.writer is not None:
            self.writer.append_event(
                run_id=str(state["loop_run_id"]),
                event_type=LOOP_TERMINATE_REQUESTED,
                level="ERROR",
                title="terminate requested",
                payload={"source": "control"},
                source="runtime",
            )
        return self.get_loop_state(str(state["loop_run_id"]))

    def enqueue_proposal(
        self,
        *,
        description: str,
        overrides: list[str],
        requested_by: str | None,
        notes: str | None = None,
        loop_run_id: str | None = None,
    ) -> dict[str, Any]:
        state = self._resolve_target_loop(loop_run_id)
        if state is None:
            raise RuntimeError("No active loop is available to queue a proposal.")
        proposal_id = self._critical_store_write(
            self.store.insert_queued_proposal,
            loop_run_id=str(state["loop_run_id"]),
            status="pending",
            description=description,
            overrides=overrides,
            requested_by=requested_by,
            created_at=_now_ts(),
            claimed_at=None,
            applied_run_id=None,
            notes=notes,
        )
        if self.writer is not None:
            self.writer.append_event(
                run_id=str(state["loop_run_id"]),
                event_type=QUEUE_ENQUEUED,
                level="INFO",
                title=description,
                payload={"proposal_id": proposal_id, "overrides": overrides},
                source="runtime",
            )
        proposals = self.store.fetch_queued_proposals(loop_run_id=str(state["loop_run_id"]))
        selected = next(row for row in proposals if int(row["id"]) == proposal_id)
        return self._normalize_queue_row(selected)

    def claim_next_queued_proposal(self, *, loop_run_id: str) -> dict[str, Any] | None:
        row = self._critical_store_write(
            self.store.claim_next_queued_proposal,
            loop_run_id=loop_run_id,
            claimed_at=_now_ts(),
        )
        if row is None:
            return None
        normalized = self._normalize_queue_row(row)
        if self.writer is not None:
            self.writer.append_event(
                run_id=loop_run_id,
                event_type=QUEUE_CLAIMED,
                level="INFO",
                title=normalized["description"],
                payload={"proposal_id": normalized["id"], "overrides": normalized["overrides"]},
                source="runtime",
            )
        return normalized

    def mark_queue_applied(
        self,
        *,
        proposal_id: int,
        loop_run_id: str,
        applied_run_id: str,
    ) -> None:
        self._critical_store_write(
            self.store.update_queued_proposal,
            proposal_id=proposal_id,
            status="applied",
            applied_run_id=applied_run_id,
        )
        if self.writer is not None:
            self.writer.append_event(
                run_id=loop_run_id,
                event_type=QUEUE_APPLIED,
                level="INFO",
                title="queued proposal applied",
                payload={"proposal_id": proposal_id, "applied_run_id": applied_run_id},
                source="runtime",
            )

    def mark_queue_failed(
        self,
        *,
        proposal_id: int,
        notes: str,
    ) -> None:
        self._critical_store_write(
            self.store.update_queued_proposal,
            proposal_id=proposal_id,
            status="failed",
            notes=notes,
        )

    def list_queued_proposals(self, loop_run_id: str) -> list[dict[str, Any]]:
        return [
            self._normalize_queue_row(row)
            for row in self.store.fetch_queued_proposals(loop_run_id=loop_run_id)
        ]

    def _resolve_target_loop(
        self,
        loop_run_id: str | None,
        *,
        require_paused: bool = False,
    ) -> dict[str, Any] | None:
        state = (
            self.get_loop_state(loop_run_id)
            if loop_run_id is not None
            else self.get_active_loop_state()
        )
        if state is None:
            return None
        if require_paused and state["status"] != "paused":
            return None
        return state

    def _normalize_loop_state(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        normalized = dict(row)
        normalized["pause_requested"] = bool(normalized.get("pause_requested"))
        normalized["stop_requested"] = bool(normalized.get("stop_requested"))
        normalized["terminate_requested"] = bool(normalized.get("terminate_requested"))
        return normalized

    def _normalize_queue_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        normalized["overrides"] = _loads(normalized.pop("overrides_json"))
        return normalized
