"""Reusable Mission Control query helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from autoresearch_core.observability.read_models import (
    build_current_run_summary,
    build_hermes_runtime_summary,
    build_run_policy_context,
)


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


def _fmt_json_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        text = str(value)
    else:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return text if len(text) <= 80 else text[:77] + "..."


class CoreMissionControlQueries:
    """High-level generic queries over a Mission Control store."""

    def __init__(self, *, store: MissionControlQueryStore) -> None:
        self.store = store

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
        return self._normalize_loop_state(row)

    def get_active_loop_state(self) -> dict[str, Any] | None:
        row = self.store.fetch_active_loop_state()
        return self._attach_loop_run_metadata(row)

    def get_latest_loop_state(self) -> dict[str, Any] | None:
        row = self.store.fetch_latest_loop_state()
        return self._attach_loop_run_metadata(row)

    def get_current_loop_state(self) -> dict[str, Any] | None:
        active = self.get_active_loop_state()
        if active is not None:
            return active
        return self.get_latest_loop_state()

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

    def list_loop_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.fetch_all(
            """
            SELECT
                r.id,
                r.name,
                r.dataset,
                r.model,
                r.status,
                r.started_at,
                r.ended_at,
                r.experiment_id
            FROM runs r
            WHERE r.phase = 'autoresearch_loop'
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
        return self.store.fetch_all(
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

    def get_run_objective_metric(self, run_id: str) -> str | None:
        row = self.store.fetch_one(
            """
            SELECT e.objective_metric
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            WHERE r.id = ?
            """,
            (run_id,),
        )
        if row is None or row.get("objective_metric") in {None, ""}:
            return None
        return str(row["objective_metric"])

    def get_run_metric(self, run_id: str, *, metric_key: str = "val_rmse") -> float | None:
        metric_keys = [metric_key]
        if metric_key == "val_rmse":
            metric_keys.append("rmse")
        row = self.store.fetch_one(
            """
            WITH ranked_decisions AS (
                SELECT
                    d.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.run_id
                        ORDER BY d.created_at DESC, d.id DESC
                    ) AS row_num
                FROM decisions d
                WHERE d.run_id = ? AND d.metric_key IN (?, ?)
            )
            SELECT COALESCE(rd.metric_value, s.best_metric) AS metric_value
            FROM runs r
            LEFT JOIN ranked_decisions rd ON rd.run_id = r.id AND rd.row_num = 1
            LEFT JOIN status_snapshots s ON s.run_id = r.id
            WHERE r.id = ?
            """,
            (run_id, metric_keys[0], metric_keys[-1], run_id),
        )
        if row is None or row.get("metric_value") is None:
            return None
        return float(row["metric_value"])

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
        detail = {
            "run": run,
            "experiment": experiment,
            "identity": self.get_run_identity(run_id),
            "change_set": self.get_change_set(run_id),
            "selection_event": self.get_selection_event(run_id),
            "lineage": self.get_run_lineage(run_id),
            "policy_context": self.get_run_policy_context(run_id),
            "change_diff": self.get_run_change_diff(run_id),
            "timeline": self.list_events(run_id=run_id, limit=200),
            "artifacts": self.list_artifacts(run_id=run_id),
            "decisions": self.list_decisions_for_run(run_id),
            "llm_calls": self.list_llm_calls_for_run(run_id),
            "tool_calls": self.list_tool_calls(run_id=run_id, limit=200),
            "curves": self.get_run_curves(run_id),
            "links": self.get_traceability_links(run_id),
            "logs": self.list_logs(run_id, limit=100),
        }
        detail.update(self._get_run_detail_extensions(run_id))
        return detail

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
                e.regime_fingerprint,
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
        regime_fingerprint = row.get("regime_fingerprint")
        row["regime_fingerprint_short"] = (
            str(regime_fingerprint)[:8] if regime_fingerprint else None
        )
        return row

    def get_run_config_payload(self, run_id: str) -> dict[str, Any] | None:
        row = self.store.fetch_one(
            """
            SELECT e.config_json
            FROM runs r
            JOIN experiments e ON e.id = r.experiment_id
            WHERE r.id = ?
            """,
            (run_id,),
        )
        if row is None:
            return None
        payload = _loads(row.get("config_json"))
        if not isinstance(payload, dict):
            return None
        return payload

    def get_run_reference(self, run_id: str | None) -> dict[str, Any] | None:
        if not run_id:
            return None
        row = self.store.fetch_one(
            """
            SELECT
                r.id AS run_id,
                r.name AS run_name,
                r.phase,
                r.status AS run_status,
                r.iteration,
                r.model,
                r.dataset,
                e.regime_fingerprint,
                COALESCE(d.metric_value, s.best_metric) AS metric_value,
                d.metric_key
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            LEFT JOIN status_snapshots s ON s.run_id = r.id
            LEFT JOIN (
                SELECT
                    d1.run_id,
                    d1.metric_key,
                    d1.metric_value
                FROM decisions d1
                JOIN (
                    SELECT run_id, MAX(created_at) AS max_created_at
                    FROM decisions
                    GROUP BY run_id
                ) latest ON latest.run_id = d1.run_id AND latest.max_created_at = d1.created_at
            ) d ON d.run_id = r.id
            WHERE r.id = ?
            """,
            (run_id,),
        )
        if row is None:
            return None
        row["run_id_short"] = str(row["run_id"])[:8]
        regime_fingerprint = row.get("regime_fingerprint")
        row["regime_fingerprint_short"] = (
            str(regime_fingerprint)[:8] if regime_fingerprint else None
        )
        return row

    def get_change_set(self, run_id: str) -> dict[str, Any] | None:
        row = self.store.fetch_one(
            "SELECT * FROM change_sets WHERE run_id = ?",
            (run_id,),
        )
        if row is None:
            return None
        row["overrides"] = _loads(row.pop("overrides_json"))
        row["change_items"] = _loads(row.pop("change_items_json"))
        row["summary"] = _loads(row.pop("summary_json"))
        return row

    def get_selection_event(self, run_id: str) -> dict[str, Any] | None:
        row = self.store.fetch_one(
            "SELECT * FROM selection_events WHERE run_id = ?",
            (run_id,),
        )
        if row is None:
            return None
        row["policy_state"] = _loads(row.pop("policy_state_json"))
        return row

    def get_run_lineage(self, run_id: str) -> dict[str, Any]:
        change_set = self.get_change_set(run_id)
        selection_event = self.get_selection_event(run_id)
        parent_run_id = None if change_set is None else change_set.get("parent_run_id")
        incumbent_run_id = None
        if selection_event is not None and selection_event.get("incumbent_run_id"):
            incumbent_run_id = selection_event.get("incumbent_run_id")
        elif change_set is not None:
            incumbent_run_id = change_set.get("incumbent_run_id")
        return {
            "parent": self.get_run_reference(
                str(parent_run_id) if isinstance(parent_run_id, str) else None
            ),
            "incumbent": self.get_run_reference(
                str(incumbent_run_id) if isinstance(incumbent_run_id, str) else None
            ),
        }

    def get_run_policy_context(self, run_id: str) -> dict[str, Any] | None:
        selection_event = self.get_selection_event(run_id)
        return build_run_policy_context(selection_event)

    def get_run_change_diff(self, run_id: str) -> list[dict[str, Any]]:
        change_set = self.get_change_set(run_id)
        if change_set is None:
            return []
        items = change_set.get("change_items")
        if not isinstance(items, list):
            return []
        return [
            {
                "path": item.get("path"),
                "category": item.get("category"),
                "before": item.get("before"),
                "after": item.get("after"),
                "before_text": _fmt_json_value(item.get("before")),
                "after_text": _fmt_json_value(item.get("after")),
            }
            for item in items
            if isinstance(item, dict)
        ]

    def get_traceability_links(self, run_id: str) -> dict[str, Any]:
        identity = self.get_run_identity(run_id)
        decisions = self.list_decisions_for_run(run_id)
        llm_calls = self.list_llm_calls_for_run(run_id)
        change_set = self.get_change_set(run_id)
        selection_event = self.get_selection_event(run_id)
        return {
            "run_id": run_id,
            "experiment_id": None if identity is None else identity.get("experiment_id"),
            "regime_fingerprint": None if identity is None else identity.get("regime_fingerprint"),
            "decision_ids": [row["id"] for row in decisions],
            "llm_call_ids": [row["id"] for row in llm_calls],
            "change_set_id": None if change_set is None else change_set.get("id"),
            "selection_event_id": None if selection_event is None else selection_event.get("id"),
        }

    def get_run_curves(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.store.fetch_all(
            """
            SELECT payload_json
            FROM events
            WHERE run_id = ? AND event_type = ?
            ORDER BY created_at ASC
            """,
            (run_id, "training_epoch"),
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

    def get_current_run_summary(self, loop_run_id: str | None) -> dict[str, Any] | None:
        if not loop_run_id:
            return None
        loop_state = self.store.fetch_one(
            "SELECT * FROM loop_state WHERE loop_run_id = ?",
            (loop_run_id,),
        )
        current_run_id: str | None = None
        if loop_state is not None and loop_state.get("active_child_run_id"):
            current_run_id = str(loop_state["active_child_run_id"])
        if current_run_id is None:
            latest_child = self.store.fetch_one(
                """
                SELECT id
                FROM runs
                WHERE parent_run_id = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (loop_run_id,),
            )
            if latest_child is not None and latest_child.get("id"):
                current_run_id = str(latest_child["id"])
        if current_run_id is None:
            return None
        detail = self.get_run_detail(current_run_id)
        if detail is None:
            return None
        run = detail["run"]
        latest_decision = detail["decisions"][0] if detail["decisions"] else None
        evaluation_config = (
            detail["experiment"].get("config", {}).get("evaluation")
            if isinstance(detail.get("experiment"), dict)
            else None
        )
        policy_context = detail.get("policy_context")
        return build_current_run_summary(
            current_run_id=current_run_id,
            run=run,
            identity=detail.get("identity"),
            latest_decision=latest_decision,
            evaluation_config=evaluation_config if isinstance(evaluation_config, dict) else None,
            policy_context=policy_context if isinstance(policy_context, dict) else None,
            llm_call_count=len(detail["llm_calls"]),
            artifact_count=len(detail["artifacts"]),
            is_active=bool(
                loop_state is not None and loop_state.get("active_child_run_id") == current_run_id
            ),
        )

    def get_hermes_runtime_summary(self, *, loop_run_id: str | None) -> dict[str, Any] | None:
        if loop_run_id is None:
            return None
        payload = self.get_run_config_payload(loop_run_id)
        if payload is None:
            return None
        autoresearch = payload.get("autoresearch")
        if not isinstance(autoresearch, dict):
            return None
        hermes = autoresearch.get("hermes")
        latest_llm = None
        llm_calls = self.list_recent_llm_calls(limit=1, loop_run_id=loop_run_id)
        if llm_calls:
            latest_llm = llm_calls[0]
        return build_hermes_runtime_summary(
            hermes_config=hermes if isinstance(hermes, dict) else None,
            latest_llm=latest_llm,
        )

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
                e.regime_fingerprint,
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

    def list_recent_decisions(
        self,
        *,
        limit: int = 50,
        loop_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        where_clause = ""
        params: list[Any] = []
        if loop_run_id is not None:
            where_clause = "WHERE (r.parent_run_id = ? OR d.run_id = ?)"
            params.extend([loop_run_id, loop_run_id])
        rows = self.store.fetch_all(
            f"""
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
            {where_clause}
            ORDER BY d.created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        for row in rows:
            row["overrides"] = _loads(row.pop("overrides_json"))
            row["candidates"] = _loads(row.pop("candidates_json"))
        return rows

    def list_recent_llm_calls(
        self,
        *,
        limit: int = 50,
        loop_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        where_clause = ""
        params: list[Any] = []
        if loop_run_id is not None:
            where_clause = "WHERE (l.run_id = ? OR r.parent_run_id = ?)"
            params.extend([loop_run_id, loop_run_id])
        rows = self.store.fetch_all(
            f"""
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
            {where_clause}
            ORDER BY l.created_at DESC
            LIMIT ?
            """,
            (*params, limit),
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

    def list_artifacts(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None:
            return self.store.fetch_all("SELECT * FROM artifacts ORDER BY created_at DESC")
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
        return self.store.fetch_all(
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

    def _attach_loop_run_metadata(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        loop_run_id = str(row["loop_run_id"])
        run = self.store.fetch_one(
            "SELECT name AS loop_name, dataset, model, status AS run_status FROM runs WHERE id = ?",
            (loop_run_id,),
        )
        if run is not None:
            row["loop_name"] = run.get("loop_name")
            row["dataset"] = run.get("dataset")
            row["model"] = run.get("model")
            row["run_status"] = run.get("run_status")
        return self._normalize_loop_state(row)

    def _normalize_loop_state(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        row["pause_requested"] = bool(row.get("pause_requested"))
        row["stop_requested"] = bool(row.get("stop_requested"))
        row["terminate_requested"] = bool(row.get("terminate_requested"))
        return row

    def _get_run_detail_extensions(self, run_id: str) -> dict[str, Any]:
        del run_id
        return {}


class MissionControlQueryStore(Protocol):
    blobs: Any

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]: ...

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None: ...

    def fetch_active_loop_state(self) -> dict[str, Any] | None: ...

    def fetch_latest_loop_state(self) -> dict[str, Any] | None: ...

    def fetch_queued_proposals(self, *, loop_run_id: str) -> list[dict[str, Any]]: ...
