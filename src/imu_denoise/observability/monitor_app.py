"""Textual live monitor for mission-control observability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imu_denoise.config.schema import ObservabilityConfig
from imu_denoise.observability import LoopController, MissionControlQueries, ObservabilityStore
from imu_denoise.observability.writer import ObservabilityWriter


def run_monitor(*, db_path: Path, blob_dir: Path, refresh_hz: int) -> None:
    """Start the Textual operator console."""
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Vertical
        from textual.widgets import DataTable, Footer, Header, Static
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Textual is not installed. Install `imu-denoise[monitor]` to use imu-monitor."
        ) from exc

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

    class FocusableStatic(Static):
        can_focus = True

    class MissionControlApp(App[None]):
        """Minimal live console for status, live run, and recent decisions."""

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("d", "detail", "Detail"),
            ("p", "pause", "Pause"),
            ("r", "resume", "Resume"),
            ("tab", "cycle_sections", "Cycle"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._detail_open = False
            self._focus_order = ["#status", "#live_run", "#recent_decisions", "#detail"]
            self._focus_index = 0
            self._decision_rows: list[dict[str, Any]] = []

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical():
                yield FocusableStatic(id="status")
                yield FocusableStatic(id="live_run")
                yield DataTable(id="recent_decisions")
                yield FocusableStatic(id="detail")
            yield Footer()

        def on_mount(self) -> None:
            decisions = self.query_one("#recent_decisions", DataTable)
            decisions.add_columns("iter", "source", "run", "decision", "status", "metric")
            self.set_interval(max(1.0 / max(refresh_hz, 1), 0.25), self.refresh_data)
            self.refresh_data()

        def action_detail(self) -> None:
            self._detail_open = not self._detail_open
            self.refresh_data()

        def action_pause(self) -> None:
            controller.request_pause()
            self.refresh_data()

        def action_resume(self) -> None:
            controller.resume_loop()
            self.refresh_data()

        def action_cycle_sections(self) -> None:
            self._focus_index = (self._focus_index + 1) % len(self._focus_order)
            self.query_one(self._focus_order[self._focus_index]).focus()

        def refresh_data(self) -> None:
            summary = queries.get_mission_control_summary(limit=10)
            loop_state = summary["loop_state"]
            best_result = summary["best_result"]

            status_widget = self.query_one("#status", Static)
            if loop_state is None:
                status_widget.update("STATUS\nNo active loop.")
            else:
                best_text = (
                    f"{float(best_result['metric_value']):.6f} ({best_result['run_name']})"
                    if isinstance(best_result, dict)
                    and isinstance(best_result.get("metric_value"), (int, float))
                    else "n/a"
                )
                status_widget.update(
                    "STATUS\n"
                    f"Loop: {loop_state['status']}  |  "
                    "Iteration "
                    f"{loop_state['current_iteration']}/{loop_state['max_iterations']}  |  "
                    f"Best: {best_text}"
                )

            live_widget = self.query_one("#live_run", Static)
            active_runs = queries.list_active_runs()
            if active_runs:
                current = active_runs[0]
                progress_ratio = 0.0
                max_iterations = float(loop_state["max_iterations"]) if loop_state else 0.0
                if max_iterations > 0 and loop_state is not None:
                    progress_ratio = min(
                        float(loop_state["current_iteration"]) / max_iterations,
                        1.0,
                    )
                live_widget.update(
                    "LIVE RUN\n"
                    f"{current['name']}  |  epoch {current.get('epoch') or '-'}  |  "
                    f"val_rmse {_fmt_metric(current.get('last_metric'))}\n"
                    f"{_progress_bar(progress_ratio)}"
                )
            else:
                live_widget.update("LIVE RUN\nNo active experiment run.")

            decisions_widget = self.query_one("#recent_decisions", DataTable)
            decisions_widget.clear()
            self._decision_rows = queries.list_recent_decisions(limit=12)
            for row in self._decision_rows:
                decisions_widget.add_row(
                    str(row.get("iteration") or ""),
                    str(row["proposal_source"]),
                    str(row.get("run_name") or row.get("run_id") or ""),
                    str(row["description"]),
                    str(row["status"]),
                    _fmt_metric(row.get("metric_value")),
                )

            detail_widget = self.query_one("#detail", Static)
            if not self._detail_open:
                detail_widget.update("DETAIL\nPress `d` to toggle detail.")
                return

            if best_result is None:
                detail_widget.update("DETAIL\nNo leaderboard entry available.")
                return
            selected_run_id = str(best_result["run_id"])
            cursor_row = getattr(decisions_widget, "cursor_row", 0)
            if (
                isinstance(cursor_row, int)
                and 0 <= cursor_row < len(self._decision_rows)
                and self._decision_rows[cursor_row].get("run_id")
            ):
                selected_run_id = str(self._decision_rows[cursor_row]["run_id"])
            detail = queries.get_run_detail(selected_run_id)
            if detail is None:
                detail_widget.update("DETAIL\nBest run detail not found.")
                return
            identity = detail["identity"] or {}
            detail_widget.update(
                "DETAIL\n"
                f"Run: {identity.get('run_name')}\n"
                f"Run ID: {identity.get('run_id_short')}\n"
                f"Experiment ID: {identity.get('experiment_id_short') or 'n/a'}\n"
                f"Curves: {len(detail['curves'])} points\n"
                f"Artifacts: {len(detail['artifacts'])}\n"
                f"LLM Calls: {len(detail['llm_calls'])}"
            )

    MissionControlApp().run()


def _fmt_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return ""


def _progress_bar(ratio: float, width: int = 24) -> str:
    filled = int(width * max(0.0, min(ratio, 1.0)))
    return "█" * filled + "░" * (width - filled)
