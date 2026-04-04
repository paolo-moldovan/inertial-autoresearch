"""Generic import of Hermes session state into Mission Control storage."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from autoresearch_core.observability.writer import CoreObservabilityWriter

HERMES_SESSION_IMPORTED = "hermes_session_imported"
HERMES_TRANSCRIPT_IMPORTED = "hermes_transcript_imported"

_SKILL_PATTERN = re.compile(r'invoked the "([^"]+)" skill', re.IGNORECASE)


def _session_run_id(session_id: str) -> str:
    return f"hermes:session:{session_id}"


def _parse_json_blob(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _transcript_blob(writer: Any, messages: list[dict[str, Any]]) -> str | None:
    return cast(str | None, writer.store_json_blob(messages))


def _detect_skills(system_prompt: str | None, messages: list[dict[str, Any]]) -> list[str]:
    matches: set[str] = set()
    if system_prompt:
        matches.update(_SKILL_PATTERN.findall(system_prompt))
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            matches.update(_SKILL_PATTERN.findall(content))
    return sorted(matches)


def _tool_to_special_event(tool_name: str) -> str | None:
    lowered = tool_name.lower()
    if "memory" in lowered:
        return "memory"
    if "skill" in lowered:
        return "skill"
    return None


def _iso_to_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _import_session(
    *,
    writer: Any,
    session_id: str,
    model: str | None,
    base_url: str | None,
    started_at: float | None,
    ended_at: float | None,
    source_platform: str,
    system_prompt: str | None,
    messages: list[dict[str, Any]],
    origin: str,
    session_event: str,
    transcript_event: str,
) -> None:
    run_id = _session_run_id(session_id)
    writer.start_run(
        name=session_id,
        phase="hermes_session",
        dataset=None,
        model=model,
        device=None,
        source=origin,
        run_id=run_id,
    )
    writer.append_event(
        run_id=run_id,
        session_id=session_id,
        event_type=session_event,
        level="INFO",
        title=f"Hermes session {session_id}",
        payload={
            "platform": source_platform,
            "model": model,
            "base_url": base_url,
            "started_at": started_at,
            "ended_at": ended_at,
        },
        source=origin,
        created_at=started_at,
        fingerprint=CoreObservabilityWriter._fingerprint(origin, session_id, "session"),
    )

    blob_ref = _transcript_blob(writer, messages)
    writer.append_event(
        run_id=run_id,
        session_id=session_id,
        event_type=transcript_event,
        level="INFO",
        title="Hermes transcript imported",
        payload={"message_count": len(messages), "transcript_blob_ref": blob_ref},
        source=origin,
        created_at=started_at,
        fingerprint=CoreObservabilityWriter._fingerprint(origin, session_id, "transcript"),
    )

    prompt = "\n\n".join(
        message["content"]
        for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    )
    response = "\n\n".join(
        message["content"]
        for message in messages
        if message.get("role") == "assistant" and isinstance(message.get("content"), str)
    )
    parsed_payload: dict[str, Any] | None = None
    if response:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                parsed_payload = parsed
        except json.JSONDecodeError:
            parsed_payload = None

    writer.record_llm_call(
        run_id=run_id,
        provider="custom",
        model=model,
        base_url=base_url,
        status="imported",
        latency_ms=None,
        prompt=prompt or None,
        response=response or None,
        parsed_payload=parsed_payload,
        session_id=session_id,
        reason="imported_from_hermes_state",
        source=origin,
        call_id=f"hermes-session:{session_id}",
    )

    skills = _detect_skills(system_prompt, messages)
    if skills:
        writer.record_skill_event(
            run_id=run_id,
            session_id=session_id,
            requested=skills,
            resolved=skills,
            missing=[],
            status="loaded",
            summary="skills detected in Hermes session context",
            source=origin,
            fingerprint=CoreObservabilityWriter._fingerprint(origin, session_id, "skills", skills),
        )

    for message in messages:
        tool_name = message.get("tool_name")
        tool_calls = _parse_json_blob(
            message.get("tool_calls") if isinstance(message.get("tool_calls"), str) else None
        )
        if isinstance(tool_name, str):
            special = _tool_to_special_event(tool_name)
            if special == "memory":
                writer.record_memory_event(
                    run_id=run_id,
                    session_id=session_id,
                    event_type="tool",
                    key_name=tool_name,
                    item_count=None,
                    summary=str(message.get("content") or tool_name),
                    payload=message,
                    source=origin,
                    fingerprint=CoreObservabilityWriter._fingerprint(
                        origin,
                        session_id,
                        tool_name,
                        message.get("timestamp"),
                    ),
                )
            elif special == "skill":
                writer.record_skill_event(
                    run_id=run_id,
                    session_id=session_id,
                    requested=[tool_name],
                    resolved=[tool_name],
                    missing=[],
                    status="called",
                    summary=str(message.get("content") or tool_name),
                    source=origin,
                    fingerprint=CoreObservabilityWriter._fingerprint(
                        origin,
                        session_id,
                        tool_name,
                        message.get("timestamp"),
                    ),
                )
            writer.record_tool_call(
                run_id=run_id,
                llm_call_id=f"hermes-session:{session_id}",
                session_id=session_id,
                tool_name=tool_name,
                args_summary=None,
                result_summary=str(message.get("content") or "")[:200],
                duration_ms=None,
                status="imported",
                payload=message,
                source=origin,
                fingerprint=CoreObservabilityWriter._fingerprint(
                    origin,
                    session_id,
                    tool_name,
                    message.get("timestamp"),
                ),
            )
        if isinstance(tool_calls, list):
            for index, tool_call in enumerate(tool_calls):
                name = None
                if isinstance(tool_call, dict):
                    if isinstance(tool_call.get("function"), dict):
                        maybe_name = tool_call["function"].get("name")
                        if isinstance(maybe_name, str):
                            name = maybe_name
                    if name is None:
                        maybe_name = tool_call.get("name")
                        if isinstance(maybe_name, str):
                            name = maybe_name
                if name:
                    writer.record_tool_call(
                        run_id=run_id,
                        llm_call_id=f"hermes-session:{session_id}",
                        session_id=session_id,
                        tool_name=name,
                        args_summary=None,
                        result_summary=None,
                        duration_ms=None,
                        status="planned",
                        payload=tool_call,
                        source=origin,
                        fingerprint=CoreObservabilityWriter._fingerprint(
                            origin,
                            session_id,
                            name,
                            message.get("timestamp"),
                            index,
                        ),
                    )

    if ended_at is not None:
        writer.finish_run(
            run_id=run_id,
            status="completed",
            summary={"message": "imported session", "ended_at": ended_at},
            source=origin,
        )


def _import_sqlite_state(
    *,
    writer: Any,
    db_path: Path,
    origin: str,
    session_event: str,
    transcript_event: str,
) -> int:
    cursor_real, _ = (
        writer.store.get_import_cursor("hermes_sqlite_sessions")
        if writer.store
        else (None, None)
    )
    imported = 0
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        params: tuple[Any, ...]
        query = """
            SELECT
                id,
                source,
                model,
                model_config,
                system_prompt,
                started_at,
                ended_at
            FROM sessions
        """
        if cursor_real is not None:
            query += " WHERE started_at > ?"
            params = (cursor_real,)
        else:
            params = ()
        query += " ORDER BY started_at ASC"
        session_rows = conn.execute(query, params).fetchall()
        last_started_at = cursor_real
        for row in session_rows:
            session_id = str(row["id"])
            model_config = _parse_json_blob(row["model_config"])
            base_url = model_config.get("base_url") if isinstance(model_config, dict) else None
            message_rows = conn.execute(
                """
                SELECT
                    role,
                    content,
                    tool_name,
                    tool_calls,
                    timestamp,
                    finish_reason,
                    reasoning
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,),
            ).fetchall()
            messages = [dict(message_row) for message_row in message_rows]
            _import_session(
                writer=writer,
                session_id=session_id,
                model=str(row["model"]) if row["model"] is not None else None,
                base_url=str(base_url) if base_url is not None else None,
                started_at=float(row["started_at"]),
                ended_at=float(row["ended_at"]) if row["ended_at"] is not None else None,
                source_platform=str(row["source"]),
                system_prompt=(
                    str(row["system_prompt"]) if row["system_prompt"] is not None else None
                ),
                messages=messages,
                origin=origin,
                session_event=session_event,
                transcript_event=transcript_event,
            )
            imported += 1
            last_started_at = float(row["started_at"])
        if last_started_at is not None and writer.store is not None:
            writer.store.set_import_cursor(
                "hermes_sqlite_sessions",
                cursor_real=last_started_at,
                cursor_text=None,
            )
    return imported


def _import_json_sessions(
    *,
    writer: Any,
    sessions_dir: Path,
    origin: str,
    session_event: str,
    transcript_event: str,
) -> int:
    cursor_real, _ = (
        writer.store.get_import_cursor("hermes_json_sessions")
        if writer.store
        else (None, None)
    )
    imported = 0
    last_mtime = cursor_real
    for session_path in sorted(sessions_dir.glob("*.json")):
        mtime = session_path.stat().st_mtime
        if cursor_real is not None and mtime <= cursor_real:
            continue
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        session_id = str(payload["session_id"])
        messages = list(payload.get("messages", []))
        _import_session(
            writer=writer,
            session_id=session_id,
            model=str(payload.get("model")) if payload.get("model") is not None else None,
            base_url=str(payload.get("base_url")) if payload.get("base_url") is not None else None,
            started_at=_iso_to_timestamp(payload.get("session_start")),
            ended_at=_iso_to_timestamp(payload.get("last_updated")),
            source_platform=str(payload.get("platform") or "cli"),
            system_prompt=(
                str(payload.get("system_prompt")) if payload.get("system_prompt") else None
            ),
            messages=messages if isinstance(messages, list) else [],
            origin=origin,
            session_event=session_event,
            transcript_event=transcript_event,
        )
        imported += 1
        last_mtime = mtime
    if last_mtime is not None and writer.store is not None:
        writer.store.set_import_cursor(
            "hermes_json_sessions",
            cursor_real=last_mtime,
            cursor_text=None,
        )
    return imported


def import_hermes_state(
    *,
    writer: Any,
    hermes_home: Path,
    origin: str = "hermes_import",
    session_event: str = HERMES_SESSION_IMPORTED,
    transcript_event: str = HERMES_TRANSCRIPT_IMPORTED,
) -> dict[str, int]:
    """Import Hermes SQLite and JSON session state into Mission Control storage."""
    counts = {"sqlite_sessions": 0, "json_sessions": 0}
    state_db = hermes_home / "state.db"
    sessions_dir = hermes_home / "sessions"

    if state_db.exists():
        counts["sqlite_sessions"] = _import_sqlite_state(
            writer=writer,
            db_path=state_db,
            origin=origin,
            session_event=session_event,
            transcript_event=transcript_event,
        )
    elif sessions_dir.exists():
        counts["json_sessions"] = _import_json_sessions(
            writer=writer,
            sessions_dir=sessions_dir,
            origin=origin,
            session_event=session_event,
            transcript_event=transcript_event,
        )
        return counts

    if sessions_dir.exists():
        counts["json_sessions"] = _import_json_sessions(
            writer=writer,
            sessions_dir=sessions_dir,
            origin=origin,
            session_event=session_event,
            transcript_event=transcript_event,
        )
    return counts
