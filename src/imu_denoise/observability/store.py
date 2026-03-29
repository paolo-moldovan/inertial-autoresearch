"""SQLite-backed storage for mission-control observability."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    overrides_json TEXT,
    objective_metric TEXT,
    objective_direction TEXT,
    summary_json TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phase TEXT NOT NULL,
    dataset TEXT,
    model TEXT,
    device TEXT,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    parent_run_id TEXT,
    iteration INTEGER,
    experiment_id TEXT,
    source TEXT NOT NULL,
    FOREIGN KEY (parent_run_id) REFERENCES runs(id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_run_id);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE NOT NULL,
    run_id TEXT,
    iteration INTEGER NOT NULL,
    proposal_source TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    metric_key TEXT NOT NULL,
    metric_value REAL,
    overrides_json TEXT NOT NULL,
    candidates_json TEXT,
    reason TEXT,
    llm_call_id TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    provider TEXT,
    model TEXT,
    base_url TEXT,
    status TEXT NOT NULL,
    latency_ms REAL,
    command_json TEXT,
    parsed_payload_json TEXT,
    prompt_blob_ref TEXT,
    response_blob_ref TEXT,
    stdout_blob_ref TEXT,
    stderr_blob_ref TEXT,
    session_id TEXT,
    reason TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE NOT NULL,
    run_id TEXT,
    llm_call_id TEXT,
    session_id TEXT,
    tool_name TEXT NOT NULL,
    args_summary TEXT,
    result_summary TEXT,
    duration_ms REAL,
    status TEXT NOT NULL,
    payload_blob_ref TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (llm_call_id) REFERENCES llm_calls(id)
);

CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE NOT NULL,
    run_id TEXT,
    session_id TEXT,
    event_type TEXT NOT NULL,
    key_name TEXT,
    item_count INTEGER,
    summary TEXT,
    payload_blob_ref TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS skill_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE NOT NULL,
    run_id TEXT,
    session_id TEXT,
    requested_json TEXT,
    resolved_json TEXT,
    missing_json TEXT,
    status TEXT NOT NULL,
    summary TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    label TEXT,
    metadata_json TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    UNIQUE(run_id, path),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS status_snapshots (
    run_id TEXT PRIMARY KEY,
    phase TEXT,
    epoch INTEGER,
    best_metric REAL,
    last_metric REAL,
    heartbeat_at REAL NOT NULL,
    message TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE NOT NULL,
    run_id TEXT,
    session_id TEXT,
    event_type TEXT NOT NULL,
    level TEXT,
    title TEXT,
    payload_json TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS import_state (
    source_key TEXT PRIMARY KEY,
    cursor_real REAL,
    cursor_text TEXT,
    updated_at REAL NOT NULL
);
"""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now_ts() -> float:
    return time.time()


def _fingerprint(*parts: object) -> str:
    payload = _json_dumps(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BlobStore:
    """Content-addressed gzip blob storage for large raw payloads."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_text(self, content: str, *, extension: str = ".txt.gz") -> str:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        rel_path = Path(digest[:2]) / f"{digest}{extension}"
        abs_path = self.root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if not abs_path.exists():
            with gzip.open(abs_path, "wt", encoding="utf-8") as handle:
                handle.write(content)
        return str(rel_path)

    def write_json(self, payload: Any) -> str:
        return self.write_text(_json_dumps(payload), extension=".json.gz")

    def read_text(self, ref: str) -> str:
        with gzip.open(self.root / ref, "rt", encoding="utf-8") as handle:
            return handle.read()

    def read_json(self, ref: str) -> Any:
        return json.loads(self.read_text(ref))


class ObservabilityStore:
    """Low-level SQLite interface for observability records."""

    def __init__(self, db_path: Path, blob_dir: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(blob_dir)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute("PRAGMA user_version = 1")

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self.fetch_all(query, params)
        if not rows:
            return None
        return rows[0]

    def upsert_experiment(
        self,
        *,
        experiment_id: str,
        name: str,
        config_json: dict[str, Any],
        overrides: list[str],
        objective_metric: str | None,
        objective_direction: str | None,
        summary: dict[str, Any] | None,
        source: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    id, name, config_json, overrides_json, objective_metric,
                    objective_direction, summary_json, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    config_json=excluded.config_json,
                    overrides_json=excluded.overrides_json,
                    objective_metric=excluded.objective_metric,
                    objective_direction=excluded.objective_direction,
                    summary_json=excluded.summary_json,
                    source=excluded.source
                """,
                (
                    experiment_id,
                    name,
                    _json_dumps(config_json),
                    _json_dumps(overrides),
                    objective_metric,
                    objective_direction,
                    None if summary is None else _json_dumps(summary),
                    _now_ts(),
                    source,
                ),
            )

    def upsert_run(
        self,
        *,
        run_id: str,
        name: str,
        phase: str,
        dataset: str | None,
        model: str | None,
        device: str | None,
        status: str,
        started_at: float,
        ended_at: float | None,
        parent_run_id: str | None,
        iteration: int | None,
        experiment_id: str | None,
        source: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, name, phase, dataset, model, device, status,
                    started_at, ended_at, parent_run_id, iteration, experiment_id, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    phase=excluded.phase,
                    dataset=excluded.dataset,
                    model=excluded.model,
                    device=excluded.device,
                    status=excluded.status,
                    started_at=excluded.started_at,
                    ended_at=excluded.ended_at,
                    parent_run_id=excluded.parent_run_id,
                    iteration=excluded.iteration,
                    experiment_id=excluded.experiment_id,
                    source=excluded.source
                """,
                (
                    run_id,
                    name,
                    phase,
                    dataset,
                    model,
                    device,
                    status,
                    started_at,
                    ended_at,
                    parent_run_id,
                    iteration,
                    experiment_id,
                    source,
                ),
            )

    def update_run(
        self,
        *,
        run_id: str,
        status: str | None = None,
        ended_at: float | None = None,
    ) -> None:
        assignments: list[str] = []
        params: list[Any] = []
        if status is not None:
            assignments.append("status = ?")
            params.append(status)
        if ended_at is not None:
            assignments.append("ended_at = ?")
            params.append(ended_at)
        if not assignments:
            return
        params.append(run_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE id = ?", tuple(params))

    def upsert_status_snapshot(
        self,
        *,
        run_id: str,
        phase: str | None,
        epoch: int | None,
        best_metric: float | None,
        last_metric: float | None,
        heartbeat_at: float,
        message: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO status_snapshots (
                    run_id, phase, epoch, best_metric, last_metric, heartbeat_at, message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    phase=excluded.phase,
                    epoch=excluded.epoch,
                    best_metric=excluded.best_metric,
                    last_metric=excluded.last_metric,
                    heartbeat_at=excluded.heartbeat_at,
                    message=excluded.message
                """,
                (run_id, phase, epoch, best_metric, last_metric, heartbeat_at, message),
            )

    def insert_decision(
        self,
        *,
        fingerprint: str | None,
        run_id: str | None,
        iteration: int,
        proposal_source: str,
        description: str,
        status: str,
        metric_key: str,
        metric_value: float | None,
        overrides: list[str],
        candidates: list[dict[str, Any]] | None,
        reason: str | None,
        llm_call_id: str | None,
        created_at: float,
        source: str,
    ) -> None:
        record_fingerprint = fingerprint or _fingerprint(
            run_id,
            iteration,
            proposal_source,
            description,
            status,
            metric_key,
            metric_value,
            overrides,
            source,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO decisions (
                    fingerprint, run_id, iteration, proposal_source, description,
                    status, metric_key, metric_value, overrides_json, candidates_json,
                    reason, llm_call_id, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_fingerprint,
                    run_id,
                    iteration,
                    proposal_source,
                    description,
                    status,
                    metric_key,
                    metric_value,
                    _json_dumps(overrides),
                    None if candidates is None else _json_dumps(candidates),
                    reason,
                    llm_call_id,
                    created_at,
                    source,
                ),
            )

    def insert_llm_call(
        self,
        *,
        call_id: str,
        run_id: str | None,
        provider: str | None,
        model: str | None,
        base_url: str | None,
        status: str,
        latency_ms: float | None,
        command: dict[str, Any] | None,
        parsed_payload: dict[str, Any] | None,
        prompt_blob_ref: str | None,
        response_blob_ref: str | None,
        stdout_blob_ref: str | None,
        stderr_blob_ref: str | None,
        session_id: str | None,
        reason: str | None,
        created_at: float,
        source: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_calls (
                    id, run_id, provider, model, base_url, status, latency_ms,
                    command_json, parsed_payload_json, prompt_blob_ref, response_blob_ref,
                    stdout_blob_ref, stderr_blob_ref, session_id, reason, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    run_id=excluded.run_id,
                    provider=excluded.provider,
                    model=excluded.model,
                    base_url=excluded.base_url,
                    status=excluded.status,
                    latency_ms=excluded.latency_ms,
                    command_json=excluded.command_json,
                    parsed_payload_json=excluded.parsed_payload_json,
                    prompt_blob_ref=excluded.prompt_blob_ref,
                    response_blob_ref=excluded.response_blob_ref,
                    stdout_blob_ref=excluded.stdout_blob_ref,
                    stderr_blob_ref=excluded.stderr_blob_ref,
                    session_id=excluded.session_id,
                    reason=excluded.reason,
                    created_at=excluded.created_at,
                    source=excluded.source
                """,
                (
                    call_id,
                    run_id,
                    provider,
                    model,
                    base_url,
                    status,
                    latency_ms,
                    None if command is None else _json_dumps(command),
                    None if parsed_payload is None else _json_dumps(parsed_payload),
                    prompt_blob_ref,
                    response_blob_ref,
                    stdout_blob_ref,
                    stderr_blob_ref,
                    session_id,
                    reason,
                    created_at,
                    source,
                ),
            )

    def insert_tool_call(
        self,
        *,
        fingerprint: str | None,
        run_id: str | None,
        llm_call_id: str | None,
        session_id: str | None,
        tool_name: str,
        args_summary: str | None,
        result_summary: str | None,
        duration_ms: float | None,
        status: str,
        payload_blob_ref: str | None,
        created_at: float,
        source: str,
    ) -> None:
        record_fingerprint = fingerprint or _fingerprint(
            run_id,
            llm_call_id,
            session_id,
            tool_name,
            args_summary,
            result_summary,
            created_at,
            source,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO tool_calls (
                    fingerprint, run_id, llm_call_id, session_id, tool_name,
                    args_summary, result_summary, duration_ms, status,
                    payload_blob_ref, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_fingerprint,
                    run_id,
                    llm_call_id,
                    session_id,
                    tool_name,
                    args_summary,
                    result_summary,
                    duration_ms,
                    status,
                    payload_blob_ref,
                    created_at,
                    source,
                ),
            )

    def insert_memory_event(
        self,
        *,
        fingerprint: str | None,
        run_id: str | None,
        session_id: str | None,
        event_type: str,
        key_name: str | None,
        item_count: int | None,
        summary: str | None,
        payload_blob_ref: str | None,
        created_at: float,
        source: str,
    ) -> None:
        record_fingerprint = fingerprint or _fingerprint(
            run_id,
            session_id,
            event_type,
            key_name,
            item_count,
            summary,
            created_at,
            source,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_events (
                    fingerprint, run_id, session_id, event_type, key_name,
                    item_count, summary, payload_blob_ref, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_fingerprint,
                    run_id,
                    session_id,
                    event_type,
                    key_name,
                    item_count,
                    summary,
                    payload_blob_ref,
                    created_at,
                    source,
                ),
            )

    def insert_skill_event(
        self,
        *,
        fingerprint: str | None,
        run_id: str | None,
        session_id: str | None,
        requested: list[str] | None,
        resolved: list[str] | None,
        missing: list[str] | None,
        status: str,
        summary: str | None,
        created_at: float,
        source: str,
    ) -> None:
        record_fingerprint = fingerprint or _fingerprint(
            run_id,
            session_id,
            requested,
            resolved,
            missing,
            status,
            summary,
            created_at,
            source,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO skill_events (
                    fingerprint, run_id, session_id, requested_json, resolved_json,
                    missing_json, status, summary, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_fingerprint,
                    run_id,
                    session_id,
                    None if requested is None else _json_dumps(requested),
                    None if resolved is None else _json_dumps(resolved),
                    None if missing is None else _json_dumps(missing),
                    status,
                    summary,
                    created_at,
                    source,
                ),
            )

    def insert_artifact(
        self,
        *,
        run_id: str | None,
        artifact_type: str,
        path: str,
        label: str | None,
        metadata: dict[str, Any] | None,
        created_at: float,
        source: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    run_id, artifact_type, path, label, metadata_json, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, path) DO UPDATE SET
                    artifact_type=excluded.artifact_type,
                    label=excluded.label,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at,
                    source=excluded.source
                """,
                (
                    run_id,
                    artifact_type,
                    path,
                    label,
                    None if metadata is None else _json_dumps(metadata),
                    created_at,
                    source,
                ),
            )

    def insert_event(
        self,
        *,
        fingerprint: str | None,
        run_id: str | None,
        session_id: str | None,
        event_type: str,
        level: str | None,
        title: str | None,
        payload: dict[str, Any] | None,
        created_at: float,
        source: str,
    ) -> None:
        record_fingerprint = fingerprint or _fingerprint(
            run_id,
            session_id,
            event_type,
            level,
            title,
            payload,
            created_at,
            source,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    fingerprint, run_id, session_id, event_type, level, title,
                    payload_json, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_fingerprint,
                    run_id,
                    session_id,
                    event_type,
                    level,
                    title,
                    None if payload is None else _json_dumps(payload),
                    created_at,
                    source,
                ),
            )

    def get_import_cursor(self, source_key: str) -> tuple[float | None, str | None]:
        row = self.fetch_one(
            "SELECT cursor_real, cursor_text FROM import_state WHERE source_key = ?",
            (source_key,),
        )
        if row is None:
            return None, None
        cursor_real = row["cursor_real"]
        cursor_text = row["cursor_text"]
        return (
            float(cursor_real) if cursor_real is not None else None,
            str(cursor_text) if cursor_text is not None else None,
        )

    def set_import_cursor(
        self,
        source_key: str,
        *,
        cursor_real: float | None,
        cursor_text: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO import_state (source_key, cursor_real, cursor_text, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    cursor_real=excluded.cursor_real,
                    cursor_text=excluded.cursor_text,
                    updated_at=excluded.updated_at
                """,
                (source_key, cursor_real, cursor_text, _now_ts()),
            )

