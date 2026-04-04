"""Reusable SQLite-backed storage for Mission Control observability."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
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
    regime_fingerprint TEXT,
    overrides_json TEXT,
    objective_metric TEXT,
    objective_direction TEXT,
    summary_json TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiments_regime
ON experiments(regime_fingerprint, created_at DESC);

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
    iteration INTEGER,
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

CREATE TABLE IF NOT EXISTS change_sets (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    loop_run_id TEXT,
    parent_run_id TEXT,
    incumbent_run_id TEXT,
    reference_kind TEXT NOT NULL,
    proposal_source TEXT NOT NULL,
    description TEXT NOT NULL,
    overrides_json TEXT NOT NULL,
    change_items_json TEXT NOT NULL,
    summary_json TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (loop_run_id) REFERENCES runs(id),
    FOREIGN KEY (parent_run_id) REFERENCES runs(id),
    FOREIGN KEY (incumbent_run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_change_sets_loop ON change_sets(loop_run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS selection_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE NOT NULL,
    run_id TEXT NOT NULL UNIQUE,
    loop_run_id TEXT,
    iteration INTEGER,
    proposal_source TEXT NOT NULL,
    description TEXT NOT NULL,
    incumbent_run_id TEXT,
    candidate_count INTEGER,
    rationale TEXT,
    policy_state_json TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (loop_run_id) REFERENCES runs(id),
    FOREIGN KEY (incumbent_run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_selection_events_loop
ON selection_events(loop_run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS mutation_signatures (
    signature TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    category TEXT NOT NULL,
    path TEXT,
    before_json TEXT,
    after_json TEXT,
    created_at REAL NOT NULL,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mutation_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    loop_run_id TEXT,
    signature TEXT NOT NULL,
    regime_fingerprint TEXT NOT NULL,
    proposal_source TEXT NOT NULL,
    status TEXT NOT NULL,
    metric_key TEXT NOT NULL,
    metric_value REAL,
    incumbent_metric REAL,
    metric_delta REAL,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    UNIQUE(run_id, signature),
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (loop_run_id) REFERENCES runs(id),
    FOREIGN KEY (signature) REFERENCES mutation_signatures(signature)
);

CREATE INDEX IF NOT EXISTS idx_mutation_attempts_lookup
ON mutation_attempts(signature, regime_fingerprint, created_at DESC);

CREATE TABLE IF NOT EXISTS mutation_stats (
    signature TEXT NOT NULL,
    regime_fingerprint TEXT NOT NULL,
    category TEXT NOT NULL,
    path TEXT,
    tries INTEGER NOT NULL,
    keep_count INTEGER NOT NULL,
    discard_count INTEGER NOT NULL,
    crash_count INTEGER NOT NULL,
    avg_metric_delta REAL,
    last_metric_delta REAL,
    last_status TEXT,
    last_run_id TEXT,
    confidence REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (signature, regime_fingerprint),
    FOREIGN KEY (signature) REFERENCES mutation_signatures(signature),
    FOREIGN KEY (last_run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS mutation_lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE NOT NULL,
    run_id TEXT NOT NULL,
    loop_run_id TEXT,
    signature TEXT NOT NULL,
    regime_fingerprint TEXT NOT NULL,
    severity TEXT NOT NULL,
    lesson_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    metric_delta REAL,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (loop_run_id) REFERENCES runs(id),
    FOREIGN KEY (signature) REFERENCES mutation_signatures(signature)
);

CREATE INDEX IF NOT EXISTS idx_mutation_lessons_lookup
ON mutation_lessons(regime_fingerprint, created_at DESC);

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

CREATE TABLE IF NOT EXISTS loop_state (
    loop_run_id TEXT PRIMARY KEY,
    pid INTEGER,
    status TEXT NOT NULL,
    current_iteration INTEGER NOT NULL,
    max_iterations INTEGER NOT NULL,
    batch_size INTEGER,
    pause_after_iteration INTEGER,
    pause_requested INTEGER NOT NULL DEFAULT 0,
    stop_requested INTEGER NOT NULL DEFAULT 0,
    terminate_requested INTEGER NOT NULL DEFAULT 0,
    best_metric REAL,
    best_run_id TEXT,
    active_child_run_id TEXT,
    heartbeat_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (loop_run_id) REFERENCES runs(id),
    FOREIGN KEY (best_run_id) REFERENCES runs(id),
    FOREIGN KEY (active_child_run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_loop_state_status ON loop_state(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS queued_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loop_run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    description TEXT NOT NULL,
    overrides_json TEXT NOT NULL,
    requested_by TEXT,
    created_at REAL NOT NULL,
    claimed_at REAL,
    applied_run_id TEXT,
    notes TEXT,
    FOREIGN KEY (loop_run_id) REFERENCES runs(id),
    FOREIGN KEY (applied_run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_queued_proposals_lookup
ON queued_proposals(loop_run_id, status, created_at ASC);

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

ACTIVE_LOOP_HEARTBEAT_TIMEOUT_SEC = 120.0


def _pid_is_alive(pid: int | None) -> bool:
    """Return whether a PID appears to refer to a live process."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
            self._migrate(conn)
            conn.execute("PRAGMA user_version = 4")

    def _migrate(self, conn: sqlite3.Connection) -> None:
        experiment_columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "regime_fingerprint" not in experiment_columns:
            conn.execute(
                """
                ALTER TABLE experiments
                ADD COLUMN regime_fingerprint TEXT
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_experiments_regime
                ON experiments(regime_fingerprint, created_at DESC)
                """
            )
        self._backfill_experiment_regime_fingerprints(conn)

        decision_info = conn.execute("PRAGMA table_info(decisions)").fetchall()
        needs_decision_migration = False
        for row in decision_info:
            if row["name"] == "iteration" and int(row["notnull"]) == 1:
                needs_decision_migration = True
                break
        if needs_decision_migration:
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.executescript(
                """
                ALTER TABLE decisions RENAME TO decisions_old;

                CREATE TABLE decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT UNIQUE NOT NULL,
                    run_id TEXT,
                    iteration INTEGER,
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

                INSERT INTO decisions (
                    id, fingerprint, run_id, iteration, proposal_source, description,
                    status, metric_key, metric_value, overrides_json, candidates_json,
                    reason, llm_call_id, created_at, source
                )
                SELECT
                    id, fingerprint, run_id, iteration, proposal_source, description,
                    status, metric_key, metric_value, overrides_json, candidates_json,
                    reason, llm_call_id, created_at, source
                FROM decisions_old;

                DROP TABLE decisions_old;
                CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id, created_at DESC);
                """
            )
            conn.execute("PRAGMA foreign_keys=ON")

        loop_state_columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(loop_state)").fetchall()
        }
        if "stop_requested" not in loop_state_columns:
            conn.execute(
                """
                ALTER TABLE loop_state
                ADD COLUMN stop_requested INTEGER NOT NULL DEFAULT 0
                """
            )
        if "terminate_requested" not in loop_state_columns:
            conn.execute(
                """
                ALTER TABLE loop_state
                ADD COLUMN terminate_requested INTEGER NOT NULL DEFAULT 0
                """
            )
        if "pid" not in loop_state_columns:
            conn.execute(
                """
                ALTER TABLE loop_state
                ADD COLUMN pid INTEGER
                """
            )

    def _backfill_experiment_regime_fingerprints(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id, config_json
            FROM experiments
            WHERE regime_fingerprint IS NULL OR regime_fingerprint = ''
            """
        ).fetchall()
        if not rows:
            return

        for row in rows:
            config_json = row["config_json"]
            if not isinstance(config_json, str):
                continue
            payload = json.loads(config_json)
            if not isinstance(payload, dict):
                continue
            regime_fingerprint = self._compute_regime_fingerprint(payload)
            if regime_fingerprint in {None, ""}:
                continue
            conn.execute(
                "UPDATE experiments SET regime_fingerprint = ? WHERE id = ?",
                (regime_fingerprint, str(row["id"])),
            )

    def _compute_regime_fingerprint(self, payload: dict[str, Any]) -> str | None:
        """Compute an optional regime fingerprint for a stored experiment config."""
        return None

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
        regime_fingerprint: str,
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
                    id, name, config_json, regime_fingerprint, overrides_json, objective_metric,
                    objective_direction, summary_json, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    config_json=excluded.config_json,
                    regime_fingerprint=excluded.regime_fingerprint,
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
                    regime_fingerprint,
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
        iteration: int | None,
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

    def upsert_change_set(
        self,
        *,
        change_set_id: str,
        run_id: str,
        loop_run_id: str | None,
        parent_run_id: str | None,
        incumbent_run_id: str | None,
        reference_kind: str,
        proposal_source: str,
        description: str,
        overrides: list[str],
        change_items: list[dict[str, Any]],
        summary: dict[str, Any] | None,
        created_at: float,
        source: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO change_sets (
                    id, run_id, loop_run_id, parent_run_id, incumbent_run_id,
                    reference_kind, proposal_source, description, overrides_json,
                    change_items_json, summary_json, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    id=excluded.id,
                    loop_run_id=excluded.loop_run_id,
                    parent_run_id=excluded.parent_run_id,
                    incumbent_run_id=excluded.incumbent_run_id,
                    reference_kind=excluded.reference_kind,
                    proposal_source=excluded.proposal_source,
                    description=excluded.description,
                    overrides_json=excluded.overrides_json,
                    change_items_json=excluded.change_items_json,
                    summary_json=excluded.summary_json,
                    created_at=excluded.created_at,
                    source=excluded.source
                """,
                (
                    change_set_id,
                    run_id,
                    loop_run_id,
                    parent_run_id,
                    incumbent_run_id,
                    reference_kind,
                    proposal_source,
                    description,
                    _json_dumps(overrides),
                    _json_dumps(change_items),
                    None if summary is None else _json_dumps(summary),
                    created_at,
                    source,
                ),
            )

    def upsert_selection_event(
        self,
        *,
        fingerprint: str | None,
        run_id: str,
        loop_run_id: str | None,
        iteration: int | None,
        proposal_source: str,
        description: str,
        incumbent_run_id: str | None,
        candidate_count: int | None,
        rationale: str | None,
        policy_state: dict[str, Any] | None,
        created_at: float,
        source: str,
    ) -> None:
        record_fingerprint = fingerprint or _fingerprint(
            run_id,
            loop_run_id,
            iteration,
            proposal_source,
            description,
            incumbent_run_id,
            candidate_count,
            rationale,
            source,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO selection_events (
                    fingerprint, run_id, loop_run_id, iteration, proposal_source,
                    description, incumbent_run_id, candidate_count, rationale,
                    policy_state_json, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    fingerprint=excluded.fingerprint,
                    loop_run_id=excluded.loop_run_id,
                    iteration=excluded.iteration,
                    proposal_source=excluded.proposal_source,
                    description=excluded.description,
                    incumbent_run_id=excluded.incumbent_run_id,
                    candidate_count=excluded.candidate_count,
                    rationale=excluded.rationale,
                    policy_state_json=excluded.policy_state_json,
                    created_at=excluded.created_at,
                    source=excluded.source
                """,
                (
                    record_fingerprint,
                    run_id,
                    loop_run_id,
                    iteration,
                    proposal_source,
                    description,
                    incumbent_run_id,
                    candidate_count,
                    rationale,
                    None if policy_state is None else _json_dumps(policy_state),
                    created_at,
                    source,
                ),
            )

    def upsert_mutation_signature(
        self,
        *,
        signature: str,
        display_name: str,
        category: str,
        path: str | None,
        before: Any,
        after: Any,
        created_at: float,
        source: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mutation_signatures (
                    signature, display_name, category, path, before_json, after_json,
                    created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    display_name=excluded.display_name,
                    category=excluded.category,
                    path=excluded.path,
                    before_json=excluded.before_json,
                    after_json=excluded.after_json,
                    source=excluded.source
                """,
                (
                    signature,
                    display_name,
                    category,
                    path,
                    _json_dumps(before),
                    _json_dumps(after),
                    created_at,
                    source,
                ),
            )

    def insert_mutation_attempt(
        self,
        *,
        run_id: str,
        loop_run_id: str | None,
        signature: str,
        regime_fingerprint: str,
        proposal_source: str,
        status: str,
        metric_key: str,
        metric_value: float | None,
        incumbent_metric: float | None,
        metric_delta: float | None,
        created_at: float,
        source: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO mutation_attempts (
                    id, run_id, loop_run_id, signature, regime_fingerprint, proposal_source,
                    status, metric_key, metric_value, incumbent_metric, metric_delta,
                    created_at, source
                )
                VALUES (
                    (SELECT id FROM mutation_attempts WHERE run_id = ? AND signature = ?),
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    signature,
                    run_id,
                    loop_run_id,
                    signature,
                    regime_fingerprint,
                    proposal_source,
                    status,
                    metric_key,
                    metric_value,
                    incumbent_metric,
                    metric_delta,
                    created_at,
                    source,
                ),
            )

    def upsert_mutation_stat(
        self,
        *,
        signature: str,
        regime_fingerprint: str,
        category: str,
        path: str | None,
        tries: int,
        keep_count: int,
        discard_count: int,
        crash_count: int,
        avg_metric_delta: float | None,
        last_metric_delta: float | None,
        last_status: str | None,
        last_run_id: str | None,
        confidence: float,
        updated_at: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mutation_stats (
                    signature, regime_fingerprint, category, path, tries, keep_count,
                    discard_count, crash_count, avg_metric_delta, last_metric_delta,
                    last_status, last_run_id, confidence, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature, regime_fingerprint) DO UPDATE SET
                    category=excluded.category,
                    path=excluded.path,
                    tries=excluded.tries,
                    keep_count=excluded.keep_count,
                    discard_count=excluded.discard_count,
                    crash_count=excluded.crash_count,
                    avg_metric_delta=excluded.avg_metric_delta,
                    last_metric_delta=excluded.last_metric_delta,
                    last_status=excluded.last_status,
                    last_run_id=excluded.last_run_id,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                (
                    signature,
                    regime_fingerprint,
                    category,
                    path,
                    tries,
                    keep_count,
                    discard_count,
                    crash_count,
                    avg_metric_delta,
                    last_metric_delta,
                    last_status,
                    last_run_id,
                    confidence,
                    updated_at,
                ),
            )

    def insert_mutation_lesson(
        self,
        *,
        fingerprint: str | None,
        run_id: str,
        loop_run_id: str | None,
        signature: str,
        regime_fingerprint: str,
        severity: str,
        lesson_type: str,
        summary: str,
        metric_delta: float | None,
        created_at: float,
        source: str,
    ) -> None:
        record_fingerprint = fingerprint or _fingerprint(
            run_id,
            signature,
            regime_fingerprint,
            lesson_type,
            summary,
            source,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO mutation_lessons (
                    fingerprint, run_id, loop_run_id, signature, regime_fingerprint,
                    severity, lesson_type, summary, metric_delta, created_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_fingerprint,
                    run_id,
                    loop_run_id,
                    signature,
                    regime_fingerprint,
                    severity,
                    lesson_type,
                    summary,
                    metric_delta,
                    created_at,
                    source,
                ),
            )

    def upsert_loop_state(
        self,
        *,
        loop_run_id: str,
        pid: int | None,
        status: str,
        current_iteration: int,
        max_iterations: int,
        batch_size: int | None,
        pause_after_iteration: int | None,
        pause_requested: bool,
        stop_requested: bool,
        terminate_requested: bool,
        best_metric: float | None,
        best_run_id: str | None,
        active_child_run_id: str | None,
        heartbeat_at: float,
        updated_at: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO loop_state (
                    loop_run_id, pid, status, current_iteration, max_iterations, batch_size,
                    pause_after_iteration, pause_requested, stop_requested, terminate_requested,
                    best_metric, best_run_id, active_child_run_id, heartbeat_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(loop_run_id) DO UPDATE SET
                    pid=COALESCE(excluded.pid, loop_state.pid),
                    status=excluded.status,
                    current_iteration=excluded.current_iteration,
                    max_iterations=excluded.max_iterations,
                    batch_size=excluded.batch_size,
                    pause_after_iteration=excluded.pause_after_iteration,
                    pause_requested=excluded.pause_requested,
                    stop_requested=excluded.stop_requested,
                    terminate_requested=excluded.terminate_requested,
                    best_metric=excluded.best_metric,
                    best_run_id=excluded.best_run_id,
                    active_child_run_id=excluded.active_child_run_id,
                    heartbeat_at=excluded.heartbeat_at,
                    updated_at=excluded.updated_at
                """,
                (
                    loop_run_id,
                    pid,
                    status,
                    current_iteration,
                    max_iterations,
                    batch_size,
                    pause_after_iteration,
                    1 if pause_requested else 0,
                    1 if stop_requested else 0,
                    1 if terminate_requested else 0,
                    best_metric,
                    best_run_id,
                    active_child_run_id,
                    heartbeat_at,
                    updated_at,
                ),
            )

    def acquire_loop_slot(
        self,
        *,
        loop_run_id: str,
        pid: int | None,
        status: str,
        current_iteration: int,
        max_iterations: int,
        batch_size: int | None,
        pause_after_iteration: int | None,
        pause_requested: bool,
        stop_requested: bool,
        terminate_requested: bool,
        best_metric: float | None,
        best_run_id: str | None,
        active_child_run_id: str | None,
        heartbeat_at: float,
        updated_at: float,
    ) -> str | None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute(
                """
                SELECT l.loop_run_id, l.pid
                FROM loop_state l
                JOIN runs r ON r.id = l.loop_run_id
                WHERE l.loop_run_id != ?
                  AND l.status IN ('running', 'paused', 'terminating')
                  AND l.heartbeat_at >= ?
                  AND r.status = 'running'
                ORDER BY l.heartbeat_at DESC, l.updated_at DESC
                LIMIT 1
                """,
                (loop_run_id, _now_ts() - ACTIVE_LOOP_HEARTBEAT_TIMEOUT_SEC),
            ).fetchone()
            if active is not None:
                active_row = dict(active)
                active_pid = active_row.get("pid")
                if _pid_is_alive(int(active_pid)) if isinstance(active_pid, int) else False:
                    conn.rollback()
                    return str(active["loop_run_id"])
                stale_loop_run_id = str(active["loop_run_id"])
                now = _now_ts()
                conn.execute(
                    """
                    UPDATE loop_state
                    SET status = ?, stop_requested = 1, terminate_requested = 1, updated_at = ?
                    WHERE loop_run_id = ?
                    """,
                    ("terminated", now, stale_loop_run_id),
                )
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, ended_at = COALESCE(ended_at, ?)
                    WHERE id = ? AND status = 'running'
                    """,
                    ("terminated", now, stale_loop_run_id),
                )
            conn.execute(
                """
                INSERT INTO loop_state (
                    loop_run_id, pid, status, current_iteration, max_iterations, batch_size,
                    pause_after_iteration, pause_requested, stop_requested, terminate_requested,
                    best_metric, best_run_id, active_child_run_id, heartbeat_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(loop_run_id) DO UPDATE SET
                    pid=COALESCE(excluded.pid, loop_state.pid),
                    status=excluded.status,
                    current_iteration=excluded.current_iteration,
                    max_iterations=excluded.max_iterations,
                    batch_size=excluded.batch_size,
                    pause_after_iteration=excluded.pause_after_iteration,
                    pause_requested=excluded.pause_requested,
                    stop_requested=excluded.stop_requested,
                    terminate_requested=excluded.terminate_requested,
                    best_metric=excluded.best_metric,
                    best_run_id=excluded.best_run_id,
                    active_child_run_id=excluded.active_child_run_id,
                    heartbeat_at=excluded.heartbeat_at,
                    updated_at=excluded.updated_at
                """,
                (
                    loop_run_id,
                    pid,
                    status,
                    current_iteration,
                    max_iterations,
                    batch_size,
                    pause_after_iteration,
                    1 if pause_requested else 0,
                    1 if stop_requested else 0,
                    1 if terminate_requested else 0,
                    best_metric,
                    best_run_id,
                    active_child_run_id,
                    heartbeat_at,
                    updated_at,
                ),
            )
            conn.commit()
        return None

    def update_loop_state(
        self,
        *,
        loop_run_id: str,
        values: dict[str, Any],
    ) -> None:
        if not values:
            return
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE loop_state SET {assignments}, updated_at = ? WHERE loop_run_id = ?",
                (*values.values(), _now_ts(), loop_run_id),
            )

    def fetch_loop_state(self, loop_run_id: str) -> dict[str, Any] | None:
        return self.fetch_one("SELECT * FROM loop_state WHERE loop_run_id = ?", (loop_run_id,))

    def fetch_active_loop_state(self) -> dict[str, Any] | None:
        return self.fetch_one(
            """
            SELECT l.*
            FROM loop_state l
            WHERE status IN ('running', 'paused', 'terminating')
              AND heartbeat_at >= ?
              AND loop_run_id IN (
                  SELECT id FROM runs WHERE status = 'running'
              )
            ORDER BY heartbeat_at DESC, updated_at DESC
            LIMIT 1
            """,
            (_now_ts() - ACTIVE_LOOP_HEARTBEAT_TIMEOUT_SEC,),
        )

    def fetch_latest_loop_state(self) -> dict[str, Any] | None:
        return self.fetch_one(
            """
            SELECT *
            FROM loop_state
            ORDER BY updated_at DESC
            LIMIT 1
            """
        )

    def insert_queued_proposal(
        self,
        *,
        loop_run_id: str,
        status: str,
        description: str,
        overrides: list[str],
        requested_by: str | None,
        created_at: float,
        claimed_at: float | None,
        applied_run_id: str | None,
        notes: str | None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO queued_proposals (
                    loop_run_id, status, description, overrides_json, requested_by,
                    created_at, claimed_at, applied_run_id, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    loop_run_id,
                    status,
                    description,
                    _json_dumps(overrides),
                    requested_by,
                    created_at,
                    claimed_at,
                    applied_run_id,
                    notes,
                ),
            )
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("Failed to insert queued proposal.")
        return int(row_id)

    def claim_next_queued_proposal(
        self,
        *,
        loop_run_id: str,
        claimed_at: float,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT *
                FROM queued_proposals
                WHERE loop_run_id = ? AND status = 'pending'
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                (loop_run_id,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            conn.execute(
                """
                UPDATE queued_proposals
                SET status = 'claimed', claimed_at = ?
                WHERE id = ?
                """,
                (claimed_at, row["id"]),
            )
            conn.commit()
            payload = dict(row)
            payload["status"] = "claimed"
            payload["claimed_at"] = claimed_at
            return payload

    def update_queued_proposal(
        self,
        *,
        proposal_id: int,
        status: str,
        applied_run_id: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queued_proposals
                SET status = ?,
                    applied_run_id = COALESCE(?, applied_run_id),
                    notes = COALESCE(?, notes)
                WHERE id = ?
                """,
                (status, applied_run_id, notes, proposal_id),
            )

    def fetch_queued_proposals(
        self,
        *,
        loop_run_id: str,
    ) -> list[dict[str, Any]]:
        return self.fetch_all(
            """
            SELECT *
            FROM queued_proposals
            WHERE loop_run_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (loop_run_id,),
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
