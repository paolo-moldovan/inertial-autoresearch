"""Textual live monitor for mission-control observability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imu_denoise.observability.queries import MissionControlQueries


def run_monitor(*, db_path: Path, blob_dir: Path, refresh_hz: int) -> None:
    """Start the Textual read-only monitor."""
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import (
            DataTable,
            Footer,
            Header,
            Static,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Textual is not installed. Install `imu-denoise[monitor]` to use imu-monitor."
        ) from exc

    queries = MissionControlQueries(db_path=db_path, blob_dir=blob_dir)

    class MissionControlApp(App[None]):
        """Minimal live console for run, decision, and trace status."""

        BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh")]

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical():
                with Horizontal():
                    yield DataTable(id="active_runs")
                    yield DataTable(id="decisions")
                with Horizontal():
                    yield DataTable(id="llm_calls")
                    yield Static(id="detail")
                yield DataTable(id="artifacts")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#active_runs", DataTable).add_columns(
                "run",
                "phase",
                "status",
                "epoch",
                "best",
                "last",
                "heartbeat",
            )
            self.query_one("#decisions", DataTable).add_columns(
                "run",
                "iter",
                "source",
                "decision",
                "status",
                "metric",
            )
            self.query_one("#llm_calls", DataTable).add_columns(
                "run",
                "model",
                "status",
                "latency_ms",
                "reason",
            )
            self.query_one("#artifacts", DataTable).add_columns(
                "run",
                "type",
                "label",
                "path",
            )
            self.set_interval(max(1.0 / max(refresh_hz, 1), 0.25), self.refresh_data)
            self.refresh_data()

        def action_refresh(self) -> None:
            self.refresh_data()

        def refresh_data(self) -> None:
            active_runs = queries.list_active_runs()
            decisions = queries.list_recent_decisions(limit=12)
            llm_calls = queries.list_recent_llm_calls(limit=12)
            artifacts = queries.list_artifacts()[:12]

            active_table = self.query_one("#active_runs", DataTable)
            active_table.clear()
            for row in active_runs:
                active_table.add_row(
                    row["name"],
                    row["phase"],
                    row["status"],
                    str(row.get("epoch") or ""),
                    _fmt_metric(row.get("best_metric")),
                    _fmt_metric(row.get("last_metric")),
                    _fmt_float(row.get("heartbeat_at")),
                )

            decision_table = self.query_one("#decisions", DataTable)
            decision_table.clear()
            for row in decisions:
                decision_table.add_row(
                    str(row.get("run_name") or row.get("run_id") or ""),
                    str(row["iteration"]),
                    row["proposal_source"],
                    row["description"],
                    row["status"],
                    _fmt_metric(row.get("metric_value")),
                )

            llm_table = self.query_one("#llm_calls", DataTable)
            llm_table.clear()
            for row in llm_calls:
                llm_table.add_row(
                    str(row.get("run_name") or row.get("run_id") or ""),
                    str(row.get("model") or ""),
                    row["status"],
                    _fmt_metric(row.get("latency_ms")),
                    str(row.get("reason") or ""),
                )

            artifact_table = self.query_one("#artifacts", DataTable)
            artifact_table.clear()
            for row in artifacts:
                artifact_table.add_row(
                    str(row.get("run_id") or ""),
                    row["artifact_type"],
                    str(row.get("label") or ""),
                    row["path"],
                )

            detail = {
                "overview": queries.overview(),
                "latest_decision": decisions[0] if decisions else None,
                "latest_llm_call": llm_calls[0] if llm_calls else None,
            }
            self.query_one("#detail", Static).update(json.dumps(detail, indent=2, default=str))

    MissionControlApp().run()


def _fmt_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return ""


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.0f}"
    return ""
