"""Textual live monitor for mission-control observability."""

from __future__ import annotations

import os
import subprocess
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
        """Minimal live console for status, current run, and recent decisions."""

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("d", "detail", "Detail"),
            ("p", "pause", "Pause"),
            ("r", "resume", "Resume"),
            ("s", "stop", "Stop"),
            ("t", "terminate", "Terminate"),
            ("tab", "cycle_sections", "Cycle"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._detail_open = False
            self._focus_order = [
                "#status",
                "#live_run",
                "#hermes",
                "#current_candidates",
                "#recent_decisions",
                "#detail",
            ]
            self._focus_index = 0
            self._decision_rows: list[dict[str, Any]] = []

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical():
                yield FocusableStatic(id="status")
                yield FocusableStatic(id="live_run")
                yield FocusableStatic(id="hermes")
                yield FocusableStatic(id="current_candidates")
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

        async def action_quit(self) -> None:
            _kill_mission_control_tmux_session()
            self.exit()

        def action_pause(self) -> None:
            controller.request_pause()
            self.refresh_data()

        def action_resume(self) -> None:
            controller.resume_loop()
            self.refresh_data()

        def action_stop(self) -> None:
            controller.request_stop()
            self.refresh_data()

        def action_terminate(self) -> None:
            controller.request_terminate()
            self.refresh_data()

        def action_cycle_sections(self) -> None:
            self._focus_index = (self._focus_index + 1) % len(self._focus_order)
            self.query_one(self._focus_order[self._focus_index]).focus()

        def refresh_data(self) -> None:
            summary = queries.get_mission_control_summary(limit=10)
            loop_state = summary["loop_state"]
            current_run = summary.get("current_run")
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
                    f"Best: {best_text}  |  "
                    "Flags "
                    f"pause={loop_state.get('pause_requested')} "
                    f"stop={loop_state.get('stop_requested')} "
                    f"term={loop_state.get('terminate_requested')}"
                )

            live_widget = self.query_one("#live_run", Static)
            if current_run is not None:
                progress_ratio = 0.0
                max_iterations = float(loop_state["max_iterations"]) if loop_state else 0.0
                if max_iterations > 0 and loop_state is not None:
                    progress_ratio = min(
                        float(loop_state["current_iteration"]) / max_iterations,
                        1.0,
                    )
                current_metric = _fmt_metric(
                    current_run.get("last_metric") or current_run.get("metric_value")
                )
                live_widget.update(
                    "CURRENT RUN\n"
                    f"{current_run['run_name']} ({current_run.get('model') or 'n/a'})  |  "
                    f"epoch {current_run.get('epoch') or '-'}  |  "
                    f"metric {current_metric}\n"
                    "causality="
                    f"{_fmt_causal(current_run.get('causal'))}  "
                    "realtime="
                    f"{_fmt_bool(current_run.get('realtime_mode'))}  "
                    "reconstruction="
                    f"{current_run.get('reconstruction') or 'n/a'}\n"
                    f"status={current_run.get('status') or 'n/a'}  "
                    f"decision={current_run.get('decision_status') or 'n/a'}\n"
                    f"{_progress_bar(progress_ratio)}"
                )
            else:
                live_widget.update("CURRENT RUN\nNo current experiment run.")

            hermes_widget = self.query_one("#hermes", Static)
            hermes_runtime = summary.get("hermes_runtime")
            if hermes_runtime is None:
                hermes_widget.update("HERMES\nNo Hermes runtime data.")
            else:
                toolsets = ", ".join(
                    str(item) for item in hermes_runtime.get("toolsets") or []
                ) or "n/a"
                skills = ", ".join(
                    str(item) for item in hermes_runtime.get("skills") or []
                ) or "n/a"
                hermes_widget.update(
                    "HERMES\n"
                    f"Model: {hermes_runtime.get('model') or 'n/a'}  |  "
                    f"Provider: {hermes_runtime.get('provider') or 'n/a'}\n"
                    f"Toolsets: {toolsets}\n"
                    f"Skills: {skills}\n"
                    "Session: "
                    f"{str(hermes_runtime.get('latest_session_id') or 'n/a')[:8]}  |  "
                    "Last: "
                    f"{hermes_runtime.get('latest_status') or 'n/a'} "
                    f"{_fmt_latency(hermes_runtime.get('latest_latency_ms'))}"
                )

            candidates_widget = self.query_one("#current_candidates", Static)
            candidate_pool = summary.get("current_candidate_pool")
            if not candidate_pool or not candidate_pool.get("candidates"):
                candidates_widget.update("CURRENT CANDIDATE POOL\nNo current candidate pool.")
            else:
                lines = [
                    "CURRENT CANDIDATE POOL",
                    f"Run: {candidate_pool.get('run_name') or 'n/a'}",
                    "Source: "
                    f"{candidate_pool.get('proposal_source') or 'n/a'}  |  "
                    f"Mode: {candidate_pool.get('policy_mode') or 'n/a'}",
                    "Selected: "
                    f"{_fmt_candidate_index(candidate_pool.get('selected_candidate_index'))}  |  "
                    "Preferred: "
                    f"{_fmt_candidate_index(candidate_pool.get('preferred_candidate_index'))}",
                    f"Why: {candidate_pool.get('selection_rationale') or 'n/a'}",
                ]
                if candidate_pool.get("hermes_status") or candidate_pool.get("hermes_reason"):
                    lines.append(
                        "Hermes: "
                        f"{candidate_pool.get('hermes_status') or 'n/a'}  |  "
                        f"{candidate_pool.get('hermes_reason') or 'n/a'}"
                    )
                blocked = dict(candidate_pool.get("blocked_candidates") or {})
                if blocked:
                    lines.append(f"Blocked: {len(blocked)} candidate(s)")
                for index, item in enumerate(list(candidate_pool.get("candidates") or [])[:5]):
                    marker = ""
                    if index == candidate_pool.get("selected_candidate_index"):
                        marker += "*"
                    if item.get("hermes_preferred"):
                        marker += "h"
                    marker = marker or "-"
                    lines.append(
                        f"{marker} {item.get('description') or 'n/a'}  "
                        f"score={_fmt_score(item.get('total_score'))}  "
                        f"tries={item.get('total_tries') or 0}  "
                        f"keep={item.get('keep_count') or 0}/"
                        f"discard={item.get('discard_count') or 0}/"
                        f"crash={item.get('crash_count') or 0}"
                    )
                    reasons = ", ".join(str(reason) for reason in (item.get("reasons") or [])[:3])
                    if reasons:
                        lines.append(f"    {reasons}")
                for description, reasons in list(blocked.items())[:3]:
                    rendered = ", ".join(str(reason) for reason in reasons[:3])
                    lines.append(f"x {description}  {rendered}")
                candidates_widget.update("\n".join(lines))

            decisions_widget = self.query_one("#recent_decisions", DataTable)
            decisions_widget.clear()
            self._decision_rows = list(summary.get("recent_decisions") or [])
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
            lineage = detail.get("lineage") or {}
            policy_context = detail.get("policy_context") or {}
            parent = lineage.get("parent") or {}
            incumbent = lineage.get("incumbent") or {}
            related_lessons = detail.get("related_lessons") or []
            change_diff = detail.get("change_diff") or []
            detail_widget.update(
                "DETAIL\n"
                f"Run: {identity.get('run_name')}\n"
                f"Run ID: {identity.get('run_id_short')}\n"
                f"Experiment ID: {identity.get('experiment_id_short') or 'n/a'}\n"
                f"Regime: {identity.get('regime_fingerprint_short') or 'n/a'}\n"
                f"Causality: {_fmt_causal(identity.get('causal'))}\n"
                "Parent: "
                f"{parent.get('run_name') or 'n/a'} "
                f"({parent.get('run_id_short') or 'n/a'})\n"
                "Incumbent: "
                f"{incumbent.get('run_name') or 'n/a'} ({incumbent.get('run_id_short') or 'n/a'})\n"
                f"Policy: {policy_context.get('policy_mode') or 'n/a'}\n"
                f"Why: {policy_context.get('rationale') or 'n/a'}\n"
                f"Diffs: {len(change_diff)}  |  Curves: {len(detail['curves'])}\n"
                f"Lessons: {len(related_lessons)}  |  LLM Calls: {len(detail['llm_calls'])}"
            )

    MissionControlApp().run()


def _fmt_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return ""


def _fmt_latency(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"({float(value):.1f} ms)"
    return ""


def _fmt_causal(value: Any) -> str:
    if value is True:
        return "causal"
    if value is False:
        return "non-causal"
    return "n/a"


def _fmt_candidate_index(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return "n/a"


def _fmt_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "n/a"


def _kill_mission_control_tmux_session() -> bool:
    """Kill the enclosing Mission Control tmux session when running inside it."""
    if not os.getenv("TMUX"):
        return False
    try:
        result = subprocess.run(
            ["tmux", "display-message", "-p", "#S"],
            check=False,
            capture_output=True,
            text=True,
        )
        session_name = result.stdout.strip()
        if result.returncode != 0 or not session_name.startswith("imu-mission-control"):
            return False
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False)
        return True
    except Exception:
        return False


def _fmt_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "n/a"


def _progress_bar(ratio: float, width: int = 24) -> str:
    filled = int(width * max(0.0, min(ratio, 1.0)))
    return "█" * filled + "░" * (width - filled)
