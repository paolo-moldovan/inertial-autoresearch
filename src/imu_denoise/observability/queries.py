"""Read models for mission-control TUI and dashboard surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imu_denoise.observability.store import ObservabilityStore


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


class MissionControlQueries:
    """High-level queries over the observability store."""

    def __init__(self, db_path: Path, blob_dir: Path) -> None:
        self.store = ObservabilityStore(db_path=db_path, blob_dir=blob_dir)

    def overview(self) -> dict[str, Any]:
        active_runs = self.store.fetch_one(
            "SELECT COUNT(*) AS count FROM runs WHERE status = 'running'"
        )
        failures = self.store.fetch_one(
            "SELECT COUNT(*) AS count FROM runs WHERE status IN ('failed', 'crash', 'error')"
        )
        experiments = self.store.fetch_one("SELECT COUNT(*) AS count FROM experiments")
        llm_calls = self.store.fetch_one("SELECT COUNT(*) AS count FROM llm_calls")
        latest_metric = self.store.fetch_one(
            """
            SELECT r.name, s.best_metric, s.last_metric, s.heartbeat_at
            FROM status_snapshots s
            JOIN runs r ON r.id = s.run_id
            ORDER BY s.heartbeat_at DESC
            LIMIT 1
            """
        )
        return {
            "active_runs": int(active_runs["count"]) if active_runs is not None else 0,
            "failures": int(failures["count"]) if failures is not None else 0,
            "experiments": int(experiments["count"]) if experiments is not None else 0,
            "llm_calls": int(llm_calls["count"]) if llm_calls is not None else 0,
            "latest_metric": latest_metric,
        }

    def list_active_runs(self) -> list[dict[str, Any]]:
        return self.store.fetch_all(
            """
            SELECT
                r.id,
                r.name,
                r.phase,
                r.dataset,
                r.model,
                r.device,
                r.status,
                r.started_at,
                s.epoch,
                s.best_metric,
                s.last_metric,
                s.heartbeat_at,
                s.message
            FROM runs r
            LEFT JOIN status_snapshots s ON s.run_id = r.id
            WHERE r.status = 'running'
            ORDER BY COALESCE(s.heartbeat_at, r.started_at) DESC
            """
        )

    def list_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.fetch_all(
            """
            SELECT
                r.id,
                r.name,
                r.phase,
                r.dataset,
                r.model,
                r.device,
                r.status,
                r.started_at,
                r.ended_at,
                r.parent_run_id,
                r.iteration,
                r.experiment_id,
                s.epoch,
                s.best_metric,
                s.last_metric,
                s.heartbeat_at,
                s.message
            FROM runs r
            LEFT JOIN status_snapshots s ON s.run_id = r.id
            ORDER BY r.started_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    def list_experiments(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            """
            SELECT
                e.id,
                e.name,
                e.objective_metric,
                e.objective_direction,
                e.overrides_json,
                e.summary_json,
                e.created_at,
                COUNT(r.id) AS run_count
            FROM experiments e
            LEFT JOIN runs r ON r.experiment_id = e.id
            GROUP BY e.id
            ORDER BY e.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            row["overrides"] = _loads(row.pop("overrides_json"))
            row["summary"] = _loads(row.pop("summary_json"))
        return rows

    def list_recent_decisions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            """
            SELECT
                d.id,
                d.run_id,
                r.name AS run_name,
                d.iteration,
                d.proposal_source,
                d.description,
                d.status,
                d.metric_key,
                d.metric_value,
                d.overrides_json,
                d.candidates_json,
                d.reason,
                d.llm_call_id,
                d.created_at,
                d.source
            FROM decisions d
            LEFT JOIN runs r ON r.id = d.run_id
            ORDER BY d.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            row["overrides"] = _loads(row.pop("overrides_json"))
            row["candidates"] = _loads(row.pop("candidates_json"))
        return rows

    def list_recent_llm_calls(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            """
            SELECT
                l.id,
                l.run_id,
                r.name AS run_name,
                l.provider,
                l.model,
                l.base_url,
                l.status,
                l.latency_ms,
                l.parsed_payload_json,
                l.prompt_blob_ref,
                l.response_blob_ref,
                l.stdout_blob_ref,
                l.stderr_blob_ref,
                l.session_id,
                l.reason,
                l.created_at,
                l.source
            FROM llm_calls l
            LEFT JOIN runs r ON r.id = l.run_id
            ORDER BY l.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            row["parsed_payload"] = _loads(row.pop("parsed_payload_json"))
        return rows

    def get_llm_call(self, call_id: str) -> dict[str, Any] | None:
        row = self.store.fetch_one(
            """
            SELECT *
            FROM llm_calls
            WHERE id = ?
            """,
            (call_id,),
        )
        if row is None:
            return None
        row["parsed_payload"] = _loads(row.pop("parsed_payload_json"))
        row["command"] = _loads(row.pop("command_json"))
        for key in ("prompt_blob_ref", "response_blob_ref", "stdout_blob_ref", "stderr_blob_ref"):
            ref = row.get(key)
            if isinstance(ref, str):
                row[key.replace("_blob_ref", "")] = self.store.blobs.read_text(ref)
        return row

    def get_run_detail(self, run_id: str) -> dict[str, Any] | None:
        run = self.store.fetch_one(
            """
            SELECT
                r.*,
                s.epoch,
                s.best_metric,
                s.last_metric,
                s.heartbeat_at,
                s.message AS status_message
            FROM runs r
            LEFT JOIN status_snapshots s ON s.run_id = r.id
            WHERE r.id = ?
            """,
            (run_id,),
        )
        if run is None:
            return None
        experiment = None
        if run.get("experiment_id"):
            experiment = self.store.fetch_one(
                "SELECT * FROM experiments WHERE id = ?",
                (run["experiment_id"],),
            )
            if experiment is not None:
                experiment["config"] = _loads(experiment.pop("config_json"))
                experiment["overrides"] = _loads(experiment.pop("overrides_json"))
                experiment["summary"] = _loads(experiment.pop("summary_json"))

        return {
            "run": run,
            "experiment": experiment,
            "timeline": self.list_events(run_id=run_id, limit=200),
            "artifacts": self.list_artifacts(run_id=run_id),
            "decisions": self.list_decisions_for_run(run_id),
            "llm_calls": self.list_llm_calls_for_run(run_id),
            "tool_calls": self.list_tool_calls(run_id=run_id, limit=200),
            "logs": self.list_logs(run_id, limit=100),
        }

    def list_artifacts(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None:
            rows = self.store.fetch_all("SELECT * FROM artifacts ORDER BY created_at DESC")
        else:
            rows = self.store.fetch_all(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at DESC",
                (run_id,),
            )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def list_events(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        rows = self.store.fetch_all(
            f"""
            SELECT *
            FROM events
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        for row in rows:
            row["payload"] = _loads(row.pop("payload_json"))
        return rows

    def list_logs(self, run_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.list_events(run_id=run_id, limit=limit)

    def list_decisions_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            "SELECT * FROM decisions WHERE run_id = ? ORDER BY created_at DESC",
            (run_id,),
        )
        for row in rows:
            row["overrides"] = _loads(row.pop("overrides_json"))
            row["candidates"] = _loads(row.pop("candidates_json"))
        return rows

    def list_llm_calls_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY created_at DESC",
            (run_id,),
        )
        for row in rows:
            row["parsed_payload"] = _loads(row.pop("parsed_payload_json"))
        return rows

    def list_tool_calls(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        llm_call_id: str | None = None,
        limit: int = 200,
        include_payload: bool = False,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if llm_call_id is not None:
            conditions.append("llm_call_id = ?")
            params.append(llm_call_id)
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        rows = self.store.fetch_all(
            f"""
            SELECT *
            FROM tool_calls
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        if include_payload:
            for row in rows:
                ref = row.get("payload_blob_ref")
                if isinstance(ref, str):
                    row["payload"] = self.store.blobs.read_json(ref)
        return rows

    def list_memory_events(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        return self.store.fetch_all(
            f"""
            SELECT *
            FROM memory_events
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        )

    def list_skill_events(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        rows = self.store.fetch_all(
            f"""
            SELECT *
            FROM skill_events
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        for row in rows:
            row["requested"] = _loads(row.pop("requested_json"))
            row["resolved"] = _loads(row.pop("resolved_json"))
            row["missing"] = _loads(row.pop("missing_json"))
        return rows
