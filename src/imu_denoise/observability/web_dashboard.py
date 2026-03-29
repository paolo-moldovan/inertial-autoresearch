"""Lightweight Mission Control web dashboard with in-place polling updates."""

# ruff: noqa: E501

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from imu_denoise.config.schema import ObservabilityConfig
from imu_denoise.observability import LoopController, MissionControlQueries, ObservabilityStore
from imu_denoise.observability.writer import ObservabilityWriter

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IMU Mission Control</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --fg: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --good: #10b981;
      --warn: #f59e0b;
      --bad: #ef4444;
      --border: #334155;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: linear-gradient(180deg, #0b1220 0%, var(--bg) 100%);
      color: var(--fg);
    }
    header, section, aside {
      background: rgba(17, 24, 39, 0.94);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
    }
    .shell {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 16px;
      padding: 16px;
      min-height: 100vh;
    }
    .left, .right { display: grid; gap: 16px; align-content: start; }
    .row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .metric { background: var(--panel-2); border-radius: 12px; padding: 12px; }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    button {
      border: 1px solid var(--border);
      background: #0b1220;
      color: var(--fg);
      padding: 10px 12px;
      border-radius: 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button.danger { border-color: #7f1d1d; color: #fecaca; }
    button.warn { border-color: #78350f; color: #fde68a; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--border); font-size: 14px; }
    tr:hover { background: rgba(56, 189, 248, 0.08); }
    .clickable { cursor: pointer; }
    .muted { color: var(--muted); }
    .good { color: var(--good); }
    .warn-text { color: var(--warn); }
    .bad { color: var(--bad); }
    code, pre { background: #0b1220; border-radius: 10px; padding: 10px; overflow: auto; }
    pre { white-space: pre-wrap; }
    .toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
    input {
      width: 100%;
      background: #0b1220;
      color: var(--fg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
    }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .tag { font-size: 12px; color: var(--muted); }
    .status-line { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  </style>
</head>
<body>
  <div class="shell">
    <div class="left">
      <header>
        <div class="toolbar">
          <input id="search-id" placeholder="Search by run / experiment / LLM ID prefix">
          <button id="search-btn">Jump</button>
        </div>
        <div id="status-line" class="status-line muted">Loading Mission Control…</div>
        <div class="controls">
          <button id="pause-btn">Pause</button>
          <button id="resume-btn">Resume</button>
          <button id="stop-btn" class="warn">Stop After Current Iteration</button>
          <button id="terminate-btn" class="danger">Terminate Now</button>
        </div>
      </header>
      <section>
        <h2>Leaderboard</h2>
        <table>
          <thead><tr><th>#</th><th>Run</th><th>Model</th><th>Metric</th><th>Status</th><th>Source</th></tr></thead>
          <tbody id="leaderboard-body"></tbody>
        </table>
      </section>
      <section>
        <div class="split">
          <div>
            <h2>Queued Proposals</h2>
            <table>
              <thead><tr><th>ID</th><th>Status</th><th>Description</th></tr></thead>
              <tbody id="queue-body"></tbody>
            </table>
          </div>
          <div>
            <h2>Recent Loop Events</h2>
            <table>
              <thead><tr><th>Type</th><th>Title</th><th>Run</th></tr></thead>
              <tbody id="events-body"></tbody>
            </table>
          </div>
        </div>
      </section>
      <section>
        <h2>Run Detail</h2>
        <div id="run-detail" class="muted">Select a run from the leaderboard.</div>
      </section>
    </div>
    <div class="right">
      <aside>
        <h2>Recent Decisions</h2>
        <table>
          <thead><tr><th>Iter</th><th>Source</th><th>Run</th><th>Status</th><th>Metric</th></tr></thead>
          <tbody id="decisions-body"></tbody>
        </table>
      </aside>
      <aside>
        <h2>Recent LLM Traces</h2>
        <table>
          <thead><tr><th>ID</th><th>Model</th><th>Status</th><th>Latency</th></tr></thead>
          <tbody id="llm-body"></tbody>
        </table>
      </aside>
    </div>
  </div>
  <script>
    let selectedRunId = null;

    async function getJson(path, options) {
      const response = await fetch(path, options);
      if (!response.ok) throw new Error(await response.text());
      return await response.json();
    }

    function shortId(value) {
      return value ? String(value).slice(0, 8) : "n/a";
    }

    function metric(value) {
      return (typeof value === "number" && Number.isFinite(value)) ? value.toFixed(6) : "";
    }

    function rowHtml(cells) {
      return "<tr>" + cells.map((cell) => `<td>${cell}</td>`).join("") + "</tr>";
    }

    async function refreshSummary() {
      const data = await getJson("/api/summary");
      const loop = data.loop_state;
      const best = data.best_result;
      const statusLine = document.getElementById("status-line");
      if (!loop) {
        statusLine.textContent = "No active loop.";
      } else {
        statusLine.innerHTML = `
          <span>Loop <strong>${shortId(loop.loop_run_id)}</strong> is <strong>${loop.status}</strong></span>
          <span>Iteration <strong>${loop.current_iteration}/${loop.max_iterations}</strong></span>
          <span>Best <strong>${best ? metric(best.metric_value) : "n/a"}</strong></span>
          <span>Flags: pause=${loop.pause_requested} stop=${loop.stop_requested} terminate=${loop.terminate_requested}</span>
        `;
      }

      document.getElementById("leaderboard-body").innerHTML = data.leaderboard.map((row) => `
        <tr class="clickable" data-run-id="${row.run_id}">
          <td>${row.rank}</td>
          <td>${row.run_name}<div class="tag">${shortId(row.run_id)}</div></td>
          <td>${row.model}</td>
          <td>${metric(row.metric_value)}</td>
          <td>${row.decision_status}</td>
          <td>${row.proposal_source}</td>
        </tr>
      `).join("");

      document.getElementById("queue-body").innerHTML = data.queued_proposals.map((row) =>
        rowHtml([row.id, row.status, row.description])
      ).join("") || rowHtml(["", "<span class='muted'>empty</span>", ""]);

      document.getElementById("events-body").innerHTML = data.recent_loop_events.map((row) =>
        rowHtml([row.event_type, row.title || "", shortId(row.run_id)])
      ).join("") || rowHtml(["", "<span class='muted'>no events</span>", ""]);
    }

    async function refreshDecisions() {
      const data = await getJson("/api/decisions");
      document.getElementById("decisions-body").innerHTML = data.map((row) =>
        rowHtml([
          row.iteration ?? "",
          row.proposal_source,
          `${row.run_name || ""}<div class="tag">${shortId(row.run_id)}</div>`,
          row.status,
          metric(row.metric_value),
        ])
      ).join("");
    }

    async function refreshLlm() {
      const data = await getJson("/api/llm");
      document.getElementById("llm-body").innerHTML = data.map((row) =>
        rowHtml([
          shortId(row.id),
          row.model || "unknown",
          row.status,
          typeof row.latency_ms === "number" ? `${row.latency_ms.toFixed(1)} ms` : "",
        ])
      ).join("");
    }

    async function refreshRunDetail() {
      if (!selectedRunId) return;
      const detail = await getJson(`/api/run?run_id=${encodeURIComponent(selectedRunId)}`);
      const container = document.getElementById("run-detail");
      if (!detail || !detail.run) {
        container.innerHTML = "<span class='muted'>Run detail not found.</span>";
        return;
      }
      const decisions = detail.decisions || [];
      const artifacts = detail.artifacts || [];
      const curves = detail.curves || [];
      const llmCalls = detail.llm_calls || [];
      const latestDecision = decisions[0];
      container.innerHTML = `
        <div class="row">
          <div class="metric"><div class="label">Run</div><div class="value">${detail.run.name}</div><div class="tag">${shortId(detail.run.id)}</div></div>
          <div class="metric"><div class="label">Phase</div><div class="value">${detail.run.phase}</div></div>
          <div class="metric"><div class="label">Status</div><div class="value">${detail.run.status}</div></div>
          <div class="metric"><div class="label">Experiment</div><div class="value">${detail.identity?.experiment_id_short || "n/a"}</div></div>
        </div>
        <div class="controls">
          <button id="rerun-btn">Queue Rerun</button>
        </div>
        <div class="split">
          <div>
            <h3>Decision</h3>
            <pre>${latestDecision ? JSON.stringify(latestDecision, null, 2) : "No decision record."}</pre>
            <h3>Curves</h3>
            <pre>${curves.length ? JSON.stringify(curves.slice(-10), null, 2) : "No curves."}</pre>
          </div>
          <div>
            <h3>Artifacts</h3>
            <pre>${artifacts.length ? JSON.stringify(artifacts, null, 2) : "No artifacts."}</pre>
            <h3>LLM Calls</h3>
            <pre>${llmCalls.length ? JSON.stringify(llmCalls.slice(0, 3), null, 2) : "No LLM calls."}</pre>
          </div>
        </div>
      `;
      document.getElementById("rerun-btn").onclick = async () => {
        await getJson("/api/control/rerun", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({run_id: selectedRunId}),
        });
        await refreshAll();
      };
    }

    async function refreshAll() {
      await Promise.all([refreshSummary(), refreshDecisions(), refreshLlm()]);
      await refreshRunDetail();
      document.querySelectorAll("[data-run-id]").forEach((row) => {
        row.onclick = async () => {
          selectedRunId = row.getAttribute("data-run-id");
          history.replaceState({}, "", `?run_id=${encodeURIComponent(selectedRunId)}`);
          await refreshRunDetail();
        };
      });
    }

    async function control(path) {
      await getJson(path, {method: "POST"});
      await refreshAll();
    }

    document.getElementById("pause-btn").onclick = () => control("/api/control/pause");
    document.getElementById("resume-btn").onclick = () => control("/api/control/resume");
    document.getElementById("stop-btn").onclick = () => control("/api/control/stop");
    document.getElementById("terminate-btn").onclick = () => control("/api/control/terminate");
    document.getElementById("search-btn").onclick = async () => {
      const fragment = document.getElementById("search-id").value.trim();
      if (!fragment) return;
      const match = await getJson(`/api/resolve?id=${encodeURIComponent(fragment)}`);
      if (match.entity_type === "run") {
        selectedRunId = match.id;
        await refreshRunDetail();
      }
    };

    const params = new URLSearchParams(window.location.search);
    selectedRunId = params.get("run_id");
    refreshAll();
    setInterval(refreshAll, 1000);
  </script>
</body>
</html>
"""


def run_web_dashboard(*, db_path: Path, blob_dir: Path, host: str, port: int) -> None:
    """Serve the Mission Control dashboard over HTTP without Streamlit reruns."""
    store = ObservabilityStore(db_path=db_path, blob_dir=blob_dir)
    writer = ObservabilityWriter(
        config=ObservabilityConfig(enabled=True, db_path=str(db_path), blob_dir=str(blob_dir)),
        store=store,
    )
    controller = LoopController(store=store, writer=writer)
    queries = MissionControlQueries(db_path=db_path, blob_dir=blob_dir)

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(HTML)
                return
            if parsed.path == "/api/summary":
                self._send_json(queries.get_mission_control_summary(limit=15))
                return
            if parsed.path == "/api/decisions":
                self._send_json(queries.list_recent_decisions(limit=20))
                return
            if parsed.path == "/api/llm":
                self._send_json(queries.list_recent_llm_calls(limit=20))
                return
            if parsed.path == "/api/resolve":
                fragment = parse_qs(parsed.query).get("id", [""])[0]
                self._send_json(queries.resolve_id_fragment(fragment) or {})
                return
            if parsed.path == "/api/run":
                run_id = parse_qs(parsed.query).get("run_id", [""])[0]
                self._send_json(queries.get_run_detail(run_id) or {})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/control/pause":
                self._send_json(controller.request_pause() or {})
                return
            if parsed.path == "/api/control/resume":
                self._send_json(controller.resume_loop() or {})
                return
            if parsed.path == "/api/control/stop":
                self._send_json(controller.request_stop() or {})
                return
            if parsed.path == "/api/control/terminate":
                self._send_json(controller.request_terminate() or {})
                return
            if parsed.path == "/api/control/rerun":
                payload = self._read_json_body()
                run_id = str(payload.get("run_id") or "")
                detail = queries.get_run_detail(run_id)
                if detail is None or not detail.get("decisions"):
                    self.send_error(HTTPStatus.BAD_REQUEST, "Run cannot be requeued.")
                    return
                decision = detail["decisions"][0]
                queued = controller.enqueue_proposal(
                    description=f"rerun {detail['run']['name']}",
                    overrides=list(decision.get("overrides") or []),
                    requested_by="dashboard",
                    notes=f"requeued from run {run_id}",
                )
                self._send_json(queued)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else {}

        def _send_html(self, payload: str) -> None:
            body = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Mission Control dashboard listening on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
