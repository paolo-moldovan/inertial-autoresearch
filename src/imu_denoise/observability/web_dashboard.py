"""Lightweight Mission Control web dashboard with in-place polling updates."""

# ruff: noqa: E501

from __future__ import annotations

import json
import mimetypes
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
    .detail-grid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 16px; align-items: start; }
    .stack { display: grid; gap: 16px; }
    .mini-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
    .figure-card {
      background: #0b1220;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }
    .figure-card img {
      display: block;
      width: 100%;
      height: auto;
      background: #020617;
    }
    .figure-meta {
      padding: 10px 12px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      font-size: 13px;
    }
    .artifact-list { display: grid; gap: 8px; }
    .artifact-link {
      display: block;
      background: #0b1220;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      color: var(--fg);
      text-decoration: none;
    }
    .artifact-link:hover { border-color: var(--accent); }
    .panel {
      background: #0b1220;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
    }
    .panel h3, .panel h4 { margin-top: 0; }
    .kv { display: grid; grid-template-columns: 140px 1fr; gap: 6px 12px; font-size: 14px; }
    .pill {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      font-size: 12px;
      color: var(--fg);
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
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
        <h2>Current Run</h2>
        <div id="current-run" class="muted">No current experiment run.</div>
      </section>
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
        <h2>Hermes Runtime</h2>
        <div id="hermes-runtime" class="muted">No Hermes runtime data.</div>
      </aside>
      <aside>
        <h2>Current Candidate Pool</h2>
        <div id="candidate-pool" class="muted">No current candidate pool.</div>
      </aside>
      <aside>
        <h2>Recent Decisions</h2>
        <table>
          <thead><tr><th>Iter</th><th>Source</th><th>Run</th><th>Status</th><th>Metric</th></tr></thead>
          <tbody id="decisions-body"></tbody>
        </table>
      </aside>
      <aside>
        <h2>Mutation Memory</h2>
        <table>
          <thead><tr><th>Mutation</th><th>Stats</th><th>Confidence</th></tr></thead>
          <tbody id="mutation-body"></tbody>
        </table>
      </aside>
      <aside>
        <h2>Recent Lessons</h2>
        <table>
          <thead><tr><th>Severity</th><th>Mutation</th><th>Lesson</th></tr></thead>
          <tbody id="lessons-body"></tbody>
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
    let pollTimer = null;
    let refreshInFlight = false;
    let currentPollMs = 1500;
    const renderState = {
      summaryHash: "",
      detailHash: "",
    };

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

    function valueText(value) {
      if (value === null || value === undefined) return "n/a";
      if (typeof value === "boolean") return value ? "true" : "false";
      if (typeof value === "number") return Number.isFinite(value) ? String(value) : "n/a";
      if (typeof value === "string") return value;
      const text = JSON.stringify(value);
      return text.length <= 80 ? text : `${text.slice(0, 77)}...`;
    }

    function rowHtml(cells) {
      return "<tr>" + cells.map((cell) => `<td>${cell}</td>`).join("") + "</tr>";
    }

    function stableHash(value) {
      return JSON.stringify(value ?? null);
    }

    function hasActiveSelection() {
      const selection = window.getSelection();
      return !!selection && !selection.isCollapsed && selection.toString().trim().length > 0;
    }

    function userIsInteracting() {
      if (document.hidden || hasActiveSelection()) return true;
      const active = document.activeElement;
      if (!active) return false;
      const tag = active.tagName;
      return tag === "INPUT" || tag === "TEXTAREA";
    }

    function scheduleRefresh(nextMs = currentPollMs) {
      if (pollTimer) window.clearTimeout(pollTimer);
      pollTimer = window.setTimeout(refreshAll, nextMs);
    }

    async function refreshSummary() {
      const data = await getJson("/api/summary");
      currentPollMs = data.loop_state && ["running", "paused", "terminating"].includes(data.loop_state.status)
        ? 1500
        : 5000;
      const summaryHash = stableHash({
        loop_state: data.loop_state,
        best_result: data.best_result,
        current_run: data.current_run,
        leaderboard: data.leaderboard,
        queued_proposals: data.queued_proposals,
        recent_loop_events: data.recent_loop_events,
        recent_decisions: data.recent_decisions,
        recent_llm_calls: data.recent_llm_calls,
        regime_fingerprint: data.regime_fingerprint,
        mutation_leaderboard: data.mutation_leaderboard,
        recent_mutation_lessons: data.recent_mutation_lessons,
        hermes_runtime: data.hermes_runtime,
        current_candidate_pool: data.current_candidate_pool,
      });
      if (summaryHash === renderState.summaryHash) {
        return data;
      }
      renderState.summaryHash = summaryHash;

      const loop = data.loop_state;
      const best = data.best_result;
      const statusLine = document.getElementById("status-line");
      const currentRun = data.current_run;
      document.getElementById("pause-btn").disabled = !loop || loop.status !== "running";
      document.getElementById("resume-btn").disabled = !loop || loop.status !== "paused";
      document.getElementById("stop-btn").disabled = !loop || !["running", "paused"].includes(loop.status);
      document.getElementById("terminate-btn").disabled = !loop || !["running", "paused", "terminating"].includes(loop.status);
      if (!loop) {
        statusLine.textContent = "No current loop.";
      } else {
        const loopLabel = loop.loop_name || shortId(loop.loop_run_id);
        statusLine.innerHTML = `
          <span>Loop <strong>${loopLabel}</strong> is <strong>${loop.status}</strong></span>
          <span class="tag">id ${shortId(loop.loop_run_id)}</span>
          <span>Iteration <strong>${loop.current_iteration}/${loop.max_iterations}</strong></span>
          <span>Best <strong>${best ? metric(best.metric_value) : "n/a"}</strong></span>
          <span class="tag">regime ${shortId(data.regime_fingerprint)}</span>
          <span>Flags pause=${loop.pause_requested} stop=${loop.stop_requested} terminate=${loop.terminate_requested}</span>
        `;
      }

      document.getElementById("current-run").innerHTML = !currentRun ? "<span class='muted'>No current experiment run.</span>" : `
        <div class="row">
          <div class="metric">
            <div class="label">Run</div>
            <div class="value">${currentRun.run_name}</div>
            <div class="tag">${shortId(currentRun.run_id)} ${currentRun.is_active ? "active" : "latest"}</div>
          </div>
          <div class="metric">
            <div class="label">Model</div>
            <div class="value">${currentRun.model || "n/a"}</div>
            <div class="tag">${currentRun.phase || "n/a"} · ${currentRun.causal === true ? "causal" : currentRun.causal === false ? "non-causal" : "causality n/a"}</div>
          </div>
          <div class="metric">
            <div class="label">Epoch</div>
            <div class="value">${valueText(currentRun.epoch)}</div>
            <div class="tag">${currentRun.status || "n/a"}</div>
          </div>
          <div class="metric">
            <div class="label">Metric</div>
            <div class="value">${metric(currentRun.last_metric ?? currentRun.metric_value)}</div>
            <div class="tag">${currentRun.metric_key || "last_metric"}</div>
          </div>
        </div>
        <div class="panel" style="margin-top:12px;">
          <div class="kv">
            <div class="muted">Decision</div>
            <div>${currentRun.decision_description || "n/a"}</div>
            <div class="muted">Decision Status</div>
            <div>${currentRun.decision_status || "n/a"}</div>
            <div class="muted">Heartbeat</div>
            <div>${currentRun.heartbeat_at || "n/a"}</div>
            <div class="muted">Artifacts</div>
            <div>${valueText(currentRun.artifact_count)}</div>
            <div class="muted">LLM Calls</div>
            <div>${valueText(currentRun.llm_call_count)}</div>
            <div class="muted">Realtime Mode</div>
            <div>${valueText(currentRun.realtime_mode)}</div>
            <div class="muted">Reconstruction</div>
            <div>${currentRun.reconstruction || "n/a"}</div>
            <div class="muted">Eval Metrics</div>
            <div>${(currentRun.evaluation_metrics || []).join(", ") || "n/a"}</div>
          </div>
        </div>
      `;

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

      document.getElementById("decisions-body").innerHTML = (data.recent_decisions || []).map((row) =>
        rowHtml([
          row.iteration ?? "",
          row.proposal_source,
          `${row.run_name || ""}<div class="tag">${shortId(row.run_id)}</div>`,
          row.status,
          metric(row.metric_value),
        ])
      ).join("");

      document.getElementById("mutation-body").innerHTML = (data.mutation_leaderboard || []).map((row) =>
        rowHtml([
          `${row.display_name}<div class="tag">${row.category || ""}</div>`,
          `tries ${row.tries} | keep ${row.keep_count} | discard ${row.discard_count} | crash ${row.crash_count}`,
          typeof row.confidence === "number" ? row.confidence.toFixed(2) : "",
        ])
      ).join("") || rowHtml(["<span class='muted'>no mutation stats</span>", "", ""]);

      document.getElementById("lessons-body").innerHTML = (data.recent_mutation_lessons || []).map((row) =>
        rowHtml([
          row.severity,
          `${row.display_name}<div class="tag">${shortId(row.run_id)}</div>`,
          row.summary,
        ])
      ).join("") || rowHtml(["", "", "<span class='muted'>no lessons yet</span>"]);

      document.getElementById("llm-body").innerHTML = (data.recent_llm_calls || []).map((row) =>
        rowHtml([
          shortId(row.id),
          row.model || "unknown",
          row.status,
          typeof row.latency_ms === "number" ? `${row.latency_ms.toFixed(1)} ms` : "",
        ])
      ).join("");

      const hermes = data.hermes_runtime;
      document.getElementById("hermes-runtime").innerHTML = !hermes ? "<span class='muted'>No Hermes runtime data.</span>" : `
        <div class="kv">
          <div class="muted">Provider</div>
          <div>${hermes.provider || "n/a"}</div>
          <div class="muted">Model</div>
          <div>${hermes.model || "n/a"}</div>
          <div class="muted">Toolsets</div>
          <div>${(hermes.toolsets || []).map((item) => `<span class="pill">${item}</span>`).join(" ") || "n/a"}</div>
          <div class="muted">Skills</div>
          <div>${(hermes.skills || []).map((item) => `<span class="pill">${item}</span>`).join(" ") || "n/a"}</div>
          <div class="muted">Pass Session</div>
          <div>${valueText(hermes.pass_session_id)}</div>
          <div class="muted">Session</div>
          <div><span class="mono">${shortId(hermes.latest_session_id)}</span></div>
          <div class="muted">Last Call</div>
          <div>${hermes.latest_status || "n/a"} ${typeof hermes.latest_latency_ms === "number" ? `(${hermes.latest_latency_ms.toFixed(1)} ms)` : ""}</div>
          <div class="muted">Reason</div>
          <div>${hermes.latest_reason || "n/a"}</div>
        </div>
      `;

      const candidatePool = data.current_candidate_pool;
      document.getElementById("candidate-pool").innerHTML = !candidatePool || !(candidatePool.candidates || []).length
        ? "<span class='muted'>No current candidate pool.</span>"
        : `
        <div class="kv" style="margin-bottom:12px;">
          <div class="muted">Run</div>
          <div>${candidatePool.run_name || "n/a"} <span class="tag">${shortId(candidatePool.run_id)}</span></div>
          <div class="muted">Source</div>
          <div>${candidatePool.proposal_source || "n/a"}</div>
          <div class="muted">Policy Mode</div>
          <div>${candidatePool.policy_mode || "n/a"}</div>
          <div class="muted">Selected</div>
          <div>${valueText(candidatePool.selected_candidate_index)}</div>
          <div class="muted">Preferred</div>
          <div>${valueText(candidatePool.preferred_candidate_index)}${candidatePool.preferred_candidate_description ? ` · ${candidatePool.preferred_candidate_description}` : ""}</div>
          <div class="muted">Hermes</div>
          <div>${candidatePool.hermes_status || "n/a"}${candidatePool.hermes_reason ? ` · ${candidatePool.hermes_reason}` : ""}</div>
          <div class="muted">Blocked</div>
          <div>${Object.keys(candidatePool.blocked_candidates || {}).length}</div>
          <div class="muted">Why</div>
          <div>${candidatePool.selection_rationale || "n/a"}</div>
        </div>
        <table>
          <thead><tr><th>#</th><th>Candidate</th><th>Score</th><th>Signals</th></tr></thead>
          <tbody>${(candidatePool.candidates || []).map((item, index) => `
            <tr>
              <td>${index === candidatePool.selected_candidate_index ? "<strong>*</strong>" : item.hermes_preferred ? "h" : ""}${index}</td>
              <td>${item.description || "n/a"}<div class="tag">${item.hermes_preferred ? "Hermes preferred" : "local"}${item.regime_compatible === false ? " · regime blocked" : ""}</div></td>
              <td>${typeof item.total_score === "number" ? item.total_score.toFixed(3) : ""}</td>
              <td>${(item.reasons || []).slice(0, 3).join(", ") || "n/a"}</td>
            </tr>
          `).join("")}</tbody>
        </table>
        ${Object.keys(candidatePool.blocked_candidates || {}).length ? `
          <div class="panel" style="margin-top:12px;">
            <h3>Blocked Candidates</h3>
            <table>
              <thead><tr><th>Candidate</th><th>Reasons</th></tr></thead>
              <tbody>${Object.entries(candidatePool.blocked_candidates || {}).map(([description, reasons]) => `
                <tr>
                  <td>${description}</td>
                  <td>${(Array.isArray(reasons) ? reasons : []).join(", ") || "n/a"}</td>
                </tr>
              `).join("")}</tbody>
            </table>
          </div>
        ` : ""}
      `;
      return data;
    }

    async function refreshRunDetail() {
      if (!selectedRunId) return;
      const detail = await getJson(`/api/run?run_id=${encodeURIComponent(selectedRunId)}`);
      const detailHash = stableHash(detail);
      if (detailHash === renderState.detailHash) {
        return;
      }
      renderState.detailHash = detailHash;
      const container = document.getElementById("run-detail");
      if (!detail || !detail.run) {
        container.innerHTML = "<span class='muted'>Run detail not found.</span>";
        return;
      }
      const decisions = detail.decisions || [];
      const artifacts = detail.artifacts || [];
      const curves = detail.curves || [];
      const llmCalls = detail.llm_calls || [];
      const mutationAttempts = detail.mutation_attempts || [];
      const lineage = detail.lineage || {};
      const parentRun = lineage.parent;
      const incumbentRun = lineage.incumbent;
      const policyContext = detail.policy_context || {};
      const relatedLessons = detail.related_lessons || [];
      const changeDiff = detail.change_diff || [];
      const latestDecision = decisions[0];
      const figureArtifacts = artifacts.filter((item) => item.artifact_type === "figure");
      const otherArtifacts = artifacts.filter((item) => item.artifact_type !== "figure");
      const curvePreview = curves.slice(-10);
      container.innerHTML = `
        <div class="row">
          <div class="metric"><div class="label">Run</div><div class="value">${detail.run.name}</div><div class="tag">${shortId(detail.run.id)}</div></div>
          <div class="metric"><div class="label">Phase</div><div class="value">${detail.run.phase}</div></div>
          <div class="metric"><div class="label">Status</div><div class="value">${detail.run.status}</div></div>
          <div class="metric"><div class="label">Experiment</div><div class="value">${detail.identity?.experiment_id_short || "n/a"}</div><div class="tag">regime ${detail.identity?.regime_fingerprint_short || "n/a"} · ${detail.identity?.causal === true ? "causal" : detail.identity?.causal === false ? "non-causal" : "causality n/a"}</div></div>
        </div>
        <div class="controls">
          <button id="rerun-btn">Queue Rerun</button>
        </div>
        <div class="stack">
          <div class="mini-grid">
            <div class="panel">
              <h3>Lineage</h3>
              <div class="kv">
                <div class="muted">Parent</div>
                <div>${parentRun ? `${parentRun.run_name} <span class="tag">${shortId(parentRun.run_id)}</span>` : "n/a"}</div>
                <div class="muted">Incumbent</div>
                <div>${incumbentRun ? `${incumbentRun.run_name} <span class="tag">${shortId(incumbentRun.run_id)}</span>` : "n/a"}</div>
                <div class="muted">Reference</div>
                <div>${detail.change_set?.reference_kind || "n/a"}</div>
                <div class="muted">Compared metric</div>
                <div>${incumbentRun ? `${incumbentRun.metric_key || "metric"} ${metric(incumbentRun.metric_value)}` : "n/a"}</div>
              </div>
            </div>
            <div class="panel">
              <h3>Why Selected</h3>
              <div class="kv">
                <div class="muted">Source</div>
                <div>${policyContext.proposal_source || detail.selection_event?.proposal_source || "n/a"}</div>
                <div class="muted">Rationale</div>
                <div>${policyContext.rationale || detail.selection_event?.rationale || "n/a"}</div>
                <div class="muted">Policy mode</div>
                <div>${policyContext.policy_mode || "n/a"}</div>
                <div class="muted">Stagnating</div>
                <div>${valueText(policyContext.stagnating)}</div>
                <div class="muted">Explore p</div>
                <div>${valueText(policyContext.explore_probability)}</div>
                <div class="muted">Preferred</div>
                <div>${policyContext.preferred_candidate_description || "n/a"}</div>
              </div>
            </div>
            <div class="panel">
              <h3>Evaluation Context</h3>
              <div class="kv">
                <div class="muted">Realtime Mode</div>
                <div>${valueText(detail.experiment?.config?.evaluation?.realtime_mode)}</div>
                <div class="muted">Reconstruction</div>
                <div>${detail.experiment?.config?.evaluation?.reconstruction || "n/a"}</div>
                <div class="muted">Metrics</div>
                <div>${(detail.experiment?.config?.evaluation?.metrics || []).join(", ") || "n/a"}</div>
                <div class="muted">Causality</div>
                <div>${detail.identity?.causal === true ? "causal" : detail.identity?.causal === false ? "non-causal" : "n/a"}</div>
              </div>
            </div>
          </div>
          <div class="mini-grid">
            <div class="panel">
              <h3>Hermes Context</h3>
              <div class="kv">
                <div class="muted">Source</div>
                <div>${detail.selection_event?.proposal_source || "n/a"}</div>
                <div class="muted">LLM Calls</div>
                <div>${llmCalls.length}</div>
                <div class="muted">Last Session</div>
                <div>${llmCalls.length ? `<span class="mono">${shortId(llmCalls[0].session_id)}</span>` : "n/a"}</div>
                <div class="muted">Skills</div>
                <div>${((window.__latestSummary?.hermes_runtime?.skills) || []).map((item) => `<span class="pill">${item}</span>`).join(" ") || "n/a"}</div>
                <div class="muted">Toolsets</div>
                <div>${((window.__latestSummary?.hermes_runtime?.toolsets) || []).map((item) => `<span class="pill">${item}</span>`).join(" ") || "n/a"}</div>
              </div>
            </div>
            <div class="panel">
              <h3>Change Diff</h3>
              ${
                changeDiff.length
                  ? `<table><thead><tr><th>Path</th><th>Before</th><th>After</th></tr></thead><tbody>${changeDiff.map((item) => `
                      <tr>
                        <td><span class="mono">${item.path || ""}</span><div class="tag">${item.category || ""}</div></td>
                        <td>${item.before_text}</td>
                        <td>${item.after_text}</td>
                      </tr>
                    `).join("")}</tbody></table>`
                  : "<div class='muted'>No structured config diff.</div>"
              }
            </div>
            <div class="panel">
              <h3>Policy Candidates</h3>
              ${
                (policyContext.policy_candidates || []).length
                  ? `<table><thead><tr><th>Candidate</th><th>Score</th><th>Notes</th></tr></thead><tbody>${policyContext.policy_candidates.map((item) => `
                      <tr>
                        <td>${item.description || ""}<div class="tag">${item.hermes_preferred ? "Hermes preferred" : "local"}</div></td>
                        <td>${typeof item.total_score === "number" ? item.total_score.toFixed(3) : ""}</td>
                        <td>${(item.reasons || []).slice(0, 3).join(", ")}</td>
                      </tr>
                    `).join("")}</tbody></table>`
                  : "<div class='muted'>No policy candidate breakdown for this run.</div>"
              }
            </div>
          </div>
          <div>
            <h3>Figures</h3>
            ${
              figureArtifacts.length
                ? `<div class="figure-grid">${figureArtifacts.map((artifact) => `
                    <div class="figure-card">
                      <a href="/artifact?path=${encodeURIComponent(artifact.path)}" target="_blank" rel="noreferrer">
                        <img src="/artifact?path=${encodeURIComponent(artifact.path)}" alt="${artifact.label || artifact.artifact_type}">
                      </a>
                      <div class="figure-meta">
                        <strong>${artifact.label || artifact.artifact_type}</strong>
                        <span class="tag">${shortId(artifact.id)}</span>
                      </div>
                    </div>
                  `).join("")}</div>`
                : "<div class='muted'>No figure artifacts for this run yet.</div>"
            }
          </div>
          <div class="detail-grid">
            <div class="stack">
              <div class="panel">
                <h3>Training Curves</h3>
                <pre>${curvePreview.length ? JSON.stringify(curvePreview, null, 2) : "No curves."}</pre>
              </div>
              <div class="panel">
                <h3>Decision Snapshot</h3>
                <div class="kv">
                  <div class="muted">Status</div>
                  <div>${latestDecision?.status || "n/a"}</div>
                  <div class="muted">Metric</div>
                  <div>${latestDecision ? `${latestDecision.metric_key} ${metric(latestDecision.metric_value)}` : "n/a"}</div>
                  <div class="muted">Description</div>
                  <div>${latestDecision?.description || "n/a"}</div>
                </div>
              </div>
              <div class="panel">
                <h3>Mutation Memory</h3>
                ${
                  mutationAttempts.length
                    ? `<table><thead><tr><th>Signature</th><th>Delta</th><th>Status</th></tr></thead><tbody>${mutationAttempts.slice(0, 6).map((item) => `
                        <tr>
                          <td>${item.display_name || item.signature}</td>
                          <td>${typeof item.metric_delta === "number" ? item.metric_delta.toFixed(4) : ""}</td>
                          <td>${item.status}</td>
                        </tr>
                      `).join("")}</tbody></table>`
                    : "<div class='muted'>No mutation attempts recorded.</div>"
                }
              </div>
              <div class="panel">
                <h3>Related Lessons</h3>
                ${
                  relatedLessons.length
                    ? `<table><thead><tr><th>Severity</th><th>Mutation</th><th>Lesson</th></tr></thead><tbody>${relatedLessons.map((item) => `
                        <tr>
                          <td>${item.severity}</td>
                          <td>${item.display_name}</td>
                          <td>${item.summary}</td>
                        </tr>
                      `).join("")}</tbody></table>`
                    : "<div class='muted'>No related mutation lessons yet.</div>"
                }
              </div>
            </div>
            <div class="stack">
              <div class="panel">
                <h3>Artifacts</h3>
                ${
                  otherArtifacts.length
                    ? `<div class="artifact-list">${otherArtifacts.map((artifact) => `
                        <a class="artifact-link" href="/artifact?path=${encodeURIComponent(artifact.path)}" target="_blank" rel="noreferrer">
                          <strong>${artifact.label || artifact.artifact_type}</strong>
                          <div class="tag">${artifact.artifact_type}</div>
                        </a>
                      `).join("")}</div>`
                    : "<div class='muted'>No non-figure artifacts.</div>"
                }
              </div>
              <div class="panel">
                <h3>LLM Calls</h3>
                <pre>${llmCalls.length ? JSON.stringify(llmCalls.slice(0, 3), null, 2) : "No LLM calls."}</pre>
              </div>
            </div>
          </div>
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
      if (refreshInFlight) return;
      if (userIsInteracting()) {
        scheduleRefresh(750);
        return;
      }
      refreshInFlight = true;
      try {
        window.__latestSummary = await refreshSummary();
        await refreshRunDetail();
        document.querySelectorAll("[data-run-id]").forEach((row) => {
          row.onclick = async () => {
            selectedRunId = row.getAttribute("data-run-id");
            renderState.detailHash = "";
            history.replaceState({}, "", `?run_id=${encodeURIComponent(selectedRunId)}`);
            await refreshRunDetail();
          };
        });
      } finally {
        refreshInFlight = false;
        scheduleRefresh();
      }
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
    artifacts_root = db_path.resolve().parent.parent

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
            if parsed.path == "/artifact":
                path_value = parse_qs(parsed.query).get("path", [""])[0]
                artifact_path = Path(path_value).expanduser().resolve()
                if not self._is_allowed_artifact_path(artifact_path):
                    self.send_error(HTTPStatus.FORBIDDEN, "artifact path is outside the allowed roots")
                    return
                if not artifact_path.exists() or not artifact_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "artifact not found")
                    return
                self._send_file(artifact_path)
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

        def _send_file(self, path: Path) -> None:
            body = path.read_bytes()
            mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _is_allowed_artifact_path(self, path: Path) -> bool:
            allowed_roots = [artifacts_root, blob_dir.resolve()]
            return any(path.is_relative_to(root) for root in allowed_roots)

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Mission Control dashboard listening on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
