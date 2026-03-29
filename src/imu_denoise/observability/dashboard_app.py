"""Streamlit dashboard for mission-control observability."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from imu_denoise.config.schema import ObservabilityConfig
from imu_denoise.observability import LoopController, MissionControlQueries, ObservabilityStore
from imu_denoise.observability.writer import ObservabilityWriter


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
    queries = MissionControlQueries(db_path=db_path, blob_dir=blob_dir)
    store = ObservabilityStore(db_path=db_path, blob_dir=blob_dir)
    writer = ObservabilityWriter(
        config=ObservabilityConfig(
            enabled=True,
            db_path=str(db_path),
            blob_dir=str(blob_dir),
        ),
        store=store,
    )
    controller = LoopController(store=store, writer=writer)

    st.title("Mission Control")
    _render_sidebar(st, queries)
    pages = ["Mission Control", "Runs", "LLM Traces"]
    current_page = _query_param_value(st, "page")
    default_index = pages.index(current_page) if current_page in pages else 0
    page = st.sidebar.radio("Page", pages, index=default_index)
    st.query_params["page"] = page

    if page == "Mission Control":
        _render_mission_control(st, pd, queries, controller)
    elif page == "Runs":
        _render_runs_page(st, pd, queries)
    else:
        _render_llm_traces(st, pd, queries)


def _render_sidebar(st: Any, queries: MissionControlQueries) -> None:
    st.sidebar.subheader("Jump")
    fragment = st.sidebar.text_input("Search by ID", value="")
    if fragment:
        match = queries.resolve_id_fragment(fragment)
        if match is None:
            st.sidebar.warning("No unique ID match.")
        elif match["entity_type"] == "run":
            st.query_params["run_id"] = str(match["id"])
            st.query_params["page"] = "Runs"
            st.sidebar.success(f"Run {str(match['id'])[:8]}")
        elif match["entity_type"] == "experiment":
            st.query_params["experiment_id"] = str(match["id"])
            st.query_params["page"] = "Runs"
            st.sidebar.success(f"Experiment {str(match['id'])[:8]}")
        else:
            st.query_params["llm_call_id"] = str(match["id"])
            st.query_params["page"] = "LLM Traces"
            st.sidebar.success(f"LLM trace {str(match['id'])[:8]}")


def _render_mission_control(
    st: Any,
    pd: Any,
    queries: MissionControlQueries,
    controller: LoopController,
) -> None:
    summary = queries.get_mission_control_summary(limit=10)
    loop_state = summary["loop_state"]
    best_result = summary["best_result"]
    leaderboard = summary["leaderboard"]
    progress = summary["progress"]

    _render_status_bar(st, loop_state, controller)

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Best Result")
        if best_result is None:
            st.info("No completed training or baseline runs yet.")
        else:
            st.metric("RMSE", f"{float(best_result['metric_value']):.6f}")
            st.write(f"Model: `{best_result['model']}`")
            st.write(f"Run: `{best_result['run_name']}`")
            st.write(f"Run ID: `{str(best_result['run_id'])[:8]}`")
            if best_result.get("iteration") is not None:
                st.write(f"Iteration: `{best_result['iteration']}`")

    with right:
        st.subheader("Progress")
        if progress:
            progress_df = pd.DataFrame(progress)
            chart_df = progress_df[["iteration", "metric_value"]].rename(
                columns={"metric_value": "val_rmse"}
            )
            st.line_chart(chart_df.set_index("iteration"))
        else:
            st.info("No loop progress available yet.")

    st.subheader("Leaderboard")
    if not leaderboard:
        st.info("No leaderboard entries yet.")
        return
    leaderboard_df = pd.DataFrame(leaderboard)
    selected_run_id = _render_selectable_dataframe(
        st,
        leaderboard_df[
            [
                "rank",
                "run_name",
                "model",
                "metric_value",
                "decision_status",
                "proposal_source",
                "run_id",
                "experiment_id",
            ]
        ],
        id_column="run_id",
        key="leaderboard",
    )
    if selected_run_id:
        st.query_params["run_id"] = selected_run_id
        st.query_params["page"] = "Runs"
        st.rerun()


def _render_status_bar(
    st: Any,
    loop_state: dict[str, Any] | None,
    controller: LoopController,
) -> None:
    st.subheader("Status")
    if loop_state is None:
        st.info("No active loop.")
        return

    cols = st.columns([2, 1, 1, 1, 1])
    cols[0].write(
        f"Loop `{str(loop_state['loop_run_id'])[:8]}` is **{loop_state['status']}**"
    )
    cols[1].metric("Completed", int(loop_state["current_iteration"]))
    cols[2].metric("Max", int(loop_state["max_iterations"]))
    best_metric = loop_state.get("best_metric")
    cols[3].metric("Best", "" if best_metric is None else f"{float(best_metric):.6f}")
    cols[4].write(
        f"Heartbeat: `{int(float(loop_state['heartbeat_at']))}`"
        if loop_state.get("heartbeat_at") is not None
        else "Heartbeat: n/a"
    )

    controls = st.columns(2)
    if controls[0].button("Pause", disabled=loop_state["status"] != "running"):
        controller.request_pause(loop_run_id=str(loop_state["loop_run_id"]))
        st.rerun()
    if controls[1].button("Resume", disabled=loop_state["status"] != "paused"):
        controller.resume_loop(loop_run_id=str(loop_state["loop_run_id"]))
        st.rerun()


def _render_runs_page(st: Any, pd: Any, queries: MissionControlQueries) -> None:
    runs = queries.list_runs(limit=200)
    if not runs:
        st.info("No runs available.")
        return

    query_run_id = _query_param_value(st, "run_id")
    query_experiment_id = _query_param_value(st, "experiment_id")
    labels = {f"{row['name']} [{row['phase']}]": row["id"] for row in runs}
    default_index = 0
    if query_run_id in labels.values():
        default_index = list(labels.values()).index(query_run_id)
    elif query_experiment_id:
        for index, row in enumerate(runs):
            if row.get("experiment_id") == query_experiment_id:
                default_index = index
                break
    selected_label = st.selectbox("Run", list(labels.keys()), index=default_index)
    run_id = str(labels[selected_label])
    st.query_params["run_id"] = run_id

    detail = queries.get_run_detail(run_id)
    if detail is None:
        st.warning("Run detail not found.")
        return
    identity = detail["identity"] or {}

    st.subheader(f"Run: {identity.get('run_name') or detail['run']['name']}")
    header_cols = st.columns(4)
    header_cols[0].code(f"run {identity.get('run_id_short') or run_id[:8]}")
    experiment_id_short = identity.get("experiment_id_short")
    header_cols[1].code(
        "exp " + experiment_id_short if experiment_id_short else "exp n/a"
    )
    header_cols[2].write(f"Iteration: `{identity.get('iteration', '')}`")
    decisions = detail["decisions"]
    latest_status = decisions[0]["status"] if decisions else detail["run"]["status"]
    header_cols[3].write(f"Decision: `{latest_status}`")

    with st.expander("Training Curves", expanded=True):
        curves = detail["curves"]
        if curves:
            curves_df = pd.DataFrame(curves)
            available_columns = [
                column
                for column in ("train_loss", "val_loss", "val_rmse")
                if column in curves_df.columns
            ]
            if available_columns:
                st.line_chart(curves_df.set_index("epoch")[available_columns])
            else:
                st.dataframe(curves_df, use_container_width=True)
        else:
            st.info("No training curves available.")

    with st.expander("Config Snapshot", expanded=False):
        if detail["experiment"] is not None:
            st.json(detail["experiment"]["config"])
        else:
            st.info("No config snapshot attached.")

    with st.expander("Decision", expanded=False):
        if decisions:
            st.json(decisions[0])
        else:
            st.info("No decision record attached.")

    with st.expander("Artifacts", expanded=False):
        _artifact_gallery(st, pd, detail["artifacts"])

    with st.expander("LLM Trace", expanded=False):
        llm_calls = detail["llm_calls"]
        if not llm_calls:
            st.info("No LLM trace for this run.")
        else:
            first_call = queries.get_llm_call(str(llm_calls[0]["id"]))
            if first_call is not None:
                st.json(
                    {
                        key: value
                        for key, value in first_call.items()
                        if key not in {"prompt", "response", "stdout", "stderr"}
                    }
                )
                if first_call.get("prompt"):
                    st.code(first_call["prompt"])
                if first_call.get("response"):
                    st.code(first_call["response"])


def _render_llm_traces(st: Any, pd: Any, queries: MissionControlQueries) -> None:
    llm_calls = queries.list_recent_llm_calls(limit=200)
    if not llm_calls:
        st.info("No LLM traces available.")
        return

    query_llm_id = _query_param_value(st, "llm_call_id")
    labels = {
        f"{row['id'][:8]} | {row.get('model') or 'unknown'} | {row['status']}": row["id"]
        for row in llm_calls
    }
    default_index = 0
    if query_llm_id in labels.values():
        default_index = list(labels.values()).index(query_llm_id)
    selected_label = st.selectbox("Trace", list(labels.keys()), index=default_index)
    llm_call_id = str(labels[selected_label])
    st.query_params["llm_call_id"] = llm_call_id

    detail = queries.get_llm_call(llm_call_id)
    if detail is None:
        st.warning("Trace not found.")
        return
    run_id = detail.get("run_id")
    header_cols = st.columns(3)
    header_cols[0].code(f"llm {llm_call_id[:8]}")
    header_cols[1].write(f"Run: `{str(run_id)[:8] if run_id else 'n/a'}`")
    header_cols[2].write(f"Latency: `{detail.get('latency_ms') or 'n/a'}` ms")
    st.json(
        {
            key: value
            for key, value in detail.items()
            if key not in {"prompt", "response", "stdout", "stderr"}
        }
    )
    if detail.get("prompt"):
        st.subheader("Prompt")
        st.code(detail["prompt"])
    if detail.get("response"):
        st.subheader("Response")
        st.code(detail["response"])
    tool_calls = queries.list_tool_calls(llm_call_id=llm_call_id, limit=50, include_payload=True)
    if tool_calls:
        st.subheader("Tool Calls")
        st.dataframe(pd.DataFrame(tool_calls), use_container_width=True)


def _query_param_value(st: Any, key: str) -> str | None:
    value = st.query_params.get(key)
    if isinstance(value, list):
        if not value:
            return None
        first = value[0]
        return str(first) if first is not None else None
    if value is None:
        return None
    return str(value)


def _render_selectable_dataframe(
    st: Any,
    dataframe: Any,
    *,
    id_column: str,
    key: str,
) -> str | None:
    try:
        event = st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=key,
        )
        selected_rows = getattr(getattr(event, "selection", None), "rows", [])
        if selected_rows:
            return str(dataframe.iloc[selected_rows[0]][id_column])
    except TypeError:
        pass

    options = {
        f"{row['run_name']} ({row['model']})": row[id_column]
        for row in dataframe.to_dict(orient="records")
        if "run_name" in row
    }
    if not options:
        st.dataframe(dataframe, use_container_width=True, hide_index=True)
        return None
    selected = st.selectbox("Open run", list(options.keys()), key=f"{key}-fallback")
    return str(options[selected])


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
