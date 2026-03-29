"""Read models for mission-control TUI and dashboard surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imu_denoise.observability.control import (
    LOOP_PAUSED,
    LOOP_RESUMED,
    LOOP_STOP_REQUESTED,
    LOOP_STOPPED,
    LOOP_TERMINATE_REQUESTED,
    LOOP_TERMINATED,
    QUEUE_APPLIED,
    QUEUE_CLAIMED,
    QUEUE_ENQUEUED,
)
from imu_denoise.observability.events import TRAINING_EPOCH
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

    def get_loop_status(self) -> dict[str, Any] | None:
        row = self.store.fetch_one(
            """
            SELECT
                l.*,
                r.name AS loop_name,
                r.dataset,
                r.model
            FROM loop_state l
            JOIN runs r ON r.id = l.loop_run_id
            ORDER BY l.updated_at DESC
            LIMIT 1
            """
        )
        if row is None:
            return None
        row["pause_requested"] = bool(row.get("pause_requested"))
        return row

    def get_active_loop_state(self) -> dict[str, Any] | None:
        row = self.store.fetch_one(
            """
            SELECT
                l.*,
                r.name AS loop_name,
                r.dataset,
                r.model
            FROM loop_state l
            JOIN runs r ON r.id = l.loop_run_id
            WHERE l.status IN ('running', 'paused', 'terminating')
            ORDER BY l.updated_at DESC
            LIMIT 1
            """
        )
        if row is None:
            return None
        row["pause_requested"] = bool(row.get("pause_requested"))
        return row

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

    def list_runs_by_source(
        self,
        proposal_source: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            """
            SELECT
                r.*,
                d.proposal_source,
                d.status AS decision_status,
                d.metric_key,
                d.metric_value,
                d.created_at AS decision_created_at
            FROM runs r
            JOIN decisions d ON d.run_id = r.id
            WHERE d.proposal_source = ?
            ORDER BY d.created_at DESC
            LIMIT ?
            """,
            (proposal_source, limit),
        )
        return rows

    def list_leaderboard(
        self,
        *,
        limit: int = 10,
        metric_key: str = "val_rmse",
    ) -> list[dict[str, Any]]:
        metric_keys = [metric_key]
        if metric_key == "val_rmse":
            metric_keys.append("rmse")
        rows = self.store.fetch_all(
            """
            WITH ranked_decisions AS (
                SELECT
                    d.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.run_id
                        ORDER BY d.created_at DESC, d.id DESC
                    ) AS row_num
                FROM decisions d
                WHERE d.metric_key IN (?, ?)
            )
            SELECT
                r.id AS run_id,
                r.name AS run_name,
                r.model,
                r.phase,
                r.status AS run_status,
                r.experiment_id,
                rd.proposal_source,
                rd.status AS decision_status,
                rd.metric_key,
                COALESCE(rd.metric_value, s.best_metric) AS metric_value,
                rd.iteration,
                rd.description,
                r.started_at
            FROM runs r
            LEFT JOIN ranked_decisions rd ON rd.run_id = r.id AND rd.row_num = 1
            LEFT JOIN status_snapshots s ON s.run_id = r.id
            WHERE r.phase IN ('training', 'baseline')
              AND r.status = 'completed'
              AND COALESCE(rd.metric_value, s.best_metric) IS NOT NULL
            ORDER BY COALESCE(rd.metric_value, s.best_metric) ASC, r.started_at DESC
            LIMIT ?
            """,
            (metric_keys[0], metric_keys[-1], limit),
        )
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return rows

    def resolve_id_fragment(self, fragment: str) -> dict[str, Any] | None:
        normalized = fragment.strip()
        if not normalized:
            return None
        like_value = f"{normalized}%"
        match_specs = [
            (
                "run",
                "SELECT id, name AS label FROM runs "
                "WHERE id LIKE ? ORDER BY started_at DESC LIMIT 2",
            ),
            (
                "experiment",
                "SELECT id, name AS label FROM experiments "
                "WHERE id LIKE ? ORDER BY created_at DESC LIMIT 2",
            ),
            (
                "llm_call",
                "SELECT id, model AS label FROM llm_calls "
                "WHERE id LIKE ? ORDER BY created_at DESC LIMIT 2",
            ),
        ]
        matches: list[dict[str, Any]] = []
        for entity_type, query in match_specs:
            rows = self.store.fetch_all(query, (like_value,))
            if len(rows) == 1:
                matches.append(
                    {
                        "entity_type": entity_type,
                        "id": rows[0]["id"],
                        "label": rows[0].get("label"),
                    }
                )
            elif len(rows) > 1:
                return None
        if len(matches) == 1:
            return matches[0]
        return None

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
            "identity": self.get_run_identity(run_id),
            "timeline": self.list_events(run_id=run_id, limit=200),
            "artifacts": self.list_artifacts(run_id=run_id),
            "decisions": self.list_decisions_for_run(run_id),
            "llm_calls": self.list_llm_calls_for_run(run_id),
            "tool_calls": self.list_tool_calls(run_id=run_id, limit=200),
            "curves": self.get_run_curves(run_id),
            "links": self.get_traceability_links(run_id),
            "logs": self.list_logs(run_id, limit=100),
        }

    def get_run_identity(self, run_id: str) -> dict[str, Any] | None:
        row = self.store.fetch_one(
            """
            SELECT
                r.id AS run_id,
                r.name AS run_name,
                r.phase,
                r.status AS run_status,
                r.iteration,
                r.experiment_id,
                e.name AS experiment_name
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            WHERE r.id = ?
            """,
            (run_id,),
        )
        if row is None:
            return None
        row["run_id_short"] = str(row["run_id"])[:8]
        experiment_id = row.get("experiment_id")
        row["experiment_id_short"] = str(experiment_id)[:8] if experiment_id else None
        return row

    def get_traceability_links(self, run_id: str) -> dict[str, Any]:
        identity = self.get_run_identity(run_id)
        decisions = self.list_decisions_for_run(run_id)
        llm_calls = self.list_llm_calls_for_run(run_id)
        return {
            "run_id": run_id,
            "experiment_id": None if identity is None else identity.get("experiment_id"),
            "decision_ids": [row["id"] for row in decisions],
            "llm_call_ids": [row["id"] for row in llm_calls],
        }

    def get_run_curves(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            """
            SELECT payload_json
            FROM events
            WHERE run_id = ? AND event_type = ?
            ORDER BY created_at ASC
            """,
            (run_id, TRAINING_EPOCH),
        )
        curves: list[dict[str, Any]] = []
        for row in rows:
            payload = _loads(row["payload_json"])
            if isinstance(payload, dict):
                curves.append(payload)
        if curves:
            return curves

        history_artifact = self.store.fetch_one(
            """
            SELECT path
            FROM artifacts
            WHERE run_id = ? AND artifact_type = 'history'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (run_id,),
        )
        if history_artifact is None:
            return []
        history_path = Path(str(history_artifact["path"]))
        if not history_path.exists():
            return []
        parsed: list[dict[str, Any]] = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                parsed.append(payload)
        return parsed

    def get_mission_control_summary(self, *, limit: int = 10) -> dict[str, Any]:
        loop_state = self.get_active_loop_state()
        leaderboard = self.list_leaderboard(limit=limit)
        best_result = leaderboard[0] if leaderboard else None
        progress = []
        queued: list[dict[str, Any]] = []
        if loop_state is not None:
            progress = self.list_loop_iteration_metrics(str(loop_state["loop_run_id"]))
            queued = self.list_queued_proposals(str(loop_state["loop_run_id"]))
        return {
            "loop_state": loop_state,
            "best_result": best_result,
            "leaderboard": leaderboard,
            "progress": progress,
            "queued_proposals": queued,
            "recent_loop_events": self.list_recent_loop_events(limit=20),
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

    def list_loop_iteration_metrics(self, loop_run_id: str) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            """
            SELECT
                d.iteration,
                d.metric_key,
                d.metric_value,
                d.status,
                r.id AS run_id,
                r.name AS run_name,
                d.description
            FROM decisions d
            JOIN runs r ON r.id = d.run_id
            WHERE r.parent_run_id = ? AND d.metric_value IS NOT NULL
            ORDER BY COALESCE(d.iteration, 999999) ASC, d.created_at ASC
            """,
            (loop_run_id,),
        )
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

    def list_queued_proposals(self, loop_run_id: str) -> list[dict[str, Any]]:
        rows = self.store.fetch_queued_proposals(loop_run_id=loop_run_id)
        for row in rows:
            row["overrides"] = _loads(row.pop("overrides_json"))
        return rows

    def list_recent_loop_events(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            """
            SELECT
                e.*,
                r.name AS run_name
            FROM events e
            LEFT JOIN runs r ON r.id = e.run_id
            WHERE e.event_type IN (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ORDER BY e.created_at DESC
            LIMIT ?
            """,
            (
                LOOP_PAUSED,
                LOOP_RESUMED,
                LOOP_STOP_REQUESTED,
                LOOP_TERMINATE_REQUESTED,
                LOOP_STOPPED,
                LOOP_TERMINATED,
                QUEUE_ENQUEUED,
                QUEUE_CLAIMED,
                QUEUE_APPLIED,
                limit,
            ),
        )
        for row in rows:
            row["payload"] = _loads(row.pop("payload_json"))
        return rows
