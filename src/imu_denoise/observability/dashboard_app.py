"""Streamlit dashboard for mission-control observability."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from imu_denoise.observability.queries import MissionControlQueries


def render_dashboard(*, db_path: Path, blob_dir: Path) -> None:
    """Render the read-only Streamlit dashboard."""
    try:
        import pandas as pd  # type: ignore[import-untyped]
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Streamlit is not installed. Install `imu-denoise[monitor]` to use imu-dashboard."
        ) from exc

    st.set_page_config(page_title="IMU Mission Control", layout="wide")
    st.title("Mission Control")

    queries = MissionControlQueries(db_path=db_path, blob_dir=blob_dir)
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Experiments", "Run Detail", "LLM Traces", "Memories and Skills", "Artifacts"],
    )

    if page == "Overview":
        _render_overview(st, pd, queries)
    elif page == "Experiments":
        _render_experiments(st, pd, queries)
    elif page == "Run Detail":
        _render_run_detail(st, pd, queries)
    elif page == "LLM Traces":
        _render_llm_traces(st, queries)
    elif page == "Memories and Skills":
        _render_memories_and_skills(st, pd, queries)
    else:
        _render_artifacts(st, pd, queries)


def _render_overview(st: Any, pd: Any, queries: MissionControlQueries) -> None:
    overview = queries.overview()
    cols = st.columns(4)
    cols[0].metric("Active Runs", overview["active_runs"])
    cols[1].metric("Failures", overview["failures"])
    cols[2].metric("Experiments", overview["experiments"])
    cols[3].metric("LLM Calls", overview["llm_calls"])
    st.subheader("Active Runs")
    st.dataframe(pd.DataFrame(queries.list_active_runs()), use_container_width=True)
    st.subheader("Recent Decisions")
    st.dataframe(pd.DataFrame(queries.list_recent_decisions(limit=20)), use_container_width=True)


def _render_experiments(st: Any, pd: Any, queries: MissionControlQueries) -> None:
    experiments = queries.list_experiments(limit=200)
    st.subheader("Experiments")
    st.dataframe(pd.DataFrame(experiments), use_container_width=True)
    runs = queries.list_runs(limit=200)
    st.subheader("Runs")
    st.dataframe(pd.DataFrame(runs), use_container_width=True)


def _render_run_detail(st: Any, pd: Any, queries: MissionControlQueries) -> None:
    runs = queries.list_runs(limit=200)
    if not runs:
        st.info("No runs available.")
        return
    labels = {f"{row['name']} [{row['phase']}]": row["id"] for row in runs}
    selected = st.selectbox("Run", list(labels.keys()))
    detail = queries.get_run_detail(labels[selected])
    if detail is None:
        st.warning("Run detail not found.")
        return
    st.json(detail["run"])
    if detail["experiment"] is not None:
        st.subheader("Config Snapshot")
        st.json(detail["experiment"]["config"])
    st.subheader("Timeline")
    st.dataframe(pd.DataFrame(detail["timeline"]), use_container_width=True)
    st.subheader("Decisions")
    st.dataframe(pd.DataFrame(detail["decisions"]), use_container_width=True)
    st.subheader("LLM Calls")
    st.dataframe(pd.DataFrame(detail["llm_calls"]), use_container_width=True)
    st.subheader("Tool Calls")
    st.dataframe(pd.DataFrame(detail["tool_calls"]), use_container_width=True)
    st.subheader("Artifacts")
    _artifact_gallery(st, pd, detail["artifacts"])
    st.subheader("Logs")
    st.dataframe(pd.DataFrame(detail["logs"]), use_container_width=True)


def _render_llm_traces(st: Any, queries: MissionControlQueries) -> None:
    llm_calls = queries.list_recent_llm_calls(limit=200)
    if not llm_calls:
        st.info("No LLM traces available.")
        return
    labels = {
        f"{row['id']} | {row.get('model') or 'unknown'} | {row['status']}": row["id"]
        for row in llm_calls
    }
    selected = st.selectbox("Trace", list(labels.keys()))
    detail = queries.get_llm_call(labels[selected])
    if detail is None:
        st.warning("Trace not found.")
        return
    trace_summary = {
        key: value
        for key, value in detail.items()
        if not key.endswith(("prompt", "response"))
    }
    st.json(trace_summary)
    if detail.get("prompt"):
        st.subheader("Prompt")
        st.code(detail["prompt"])
    if detail.get("response"):
        st.subheader("Response")
        st.code(detail["response"])
    if detail.get("stdout"):
        st.subheader("Stdout")
        st.code(detail["stdout"])
    if detail.get("stderr"):
        st.subheader("Stderr")
        st.code(detail["stderr"])
    tool_calls = queries.list_tool_calls(
        llm_call_id=labels[selected],
        limit=50,
        include_payload=True,
    )
    if tool_calls:
        st.subheader("Tool Calls")
        st.dataframe(tool_calls, use_container_width=True)


def _render_memories_and_skills(st: Any, pd: Any, queries: MissionControlQueries) -> None:
    st.subheader("Memory Events")
    st.dataframe(pd.DataFrame(queries.list_memory_events(limit=200)), use_container_width=True)
    st.subheader("Skill Events")
    st.dataframe(pd.DataFrame(queries.list_skill_events(limit=200)), use_container_width=True)


def _render_artifacts(st: Any, pd: Any, queries: MissionControlQueries) -> None:
    artifacts = queries.list_artifacts()
    st.subheader("Artifacts")
    _artifact_gallery(st, pd, artifacts)


def _artifact_gallery(st: Any, pd: Any, artifacts: list[dict[str, Any]]) -> None:
    if not artifacts:
        st.info("No artifacts registered.")
        return
    st.dataframe(pd.DataFrame(artifacts), use_container_width=True)
    for artifact in artifacts:
        path = Path(artifact["path"])
        if path.suffix.lower() != ".png" or not path.exists():
            continue
        st.image(
            str(path),
            caption=f"{artifact['artifact_type']} - {artifact.get('label') or path.name}",
        )


def main() -> None:
    db_path = Path(
        os.environ.get("IMU_DASHBOARD_DB_PATH", "artifacts/observability/mission_control.db")
    )
    blob_dir = Path(
        os.environ.get("IMU_DASHBOARD_BLOB_DIR", "artifacts/observability/blobs")
    )
    render_dashboard(db_path=db_path, blob_dir=blob_dir)


if __name__ == "__main__":  # pragma: no cover - streamlit bootstrap path
    main()
