"""Hermes + Ollama integration for config-first autoresearch."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

from autoresearch_loop.mutations import MutationProposal
from imu_denoise.config import HermesConfig

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_PROJECT_SKILLS_DIR = Path(__file__).resolve().parent / "hermes_skills"


@dataclass(frozen=True)
class HermesQueryTrace:
    """Captured Hermes query metadata for observability."""

    prompt: str
    command: dict[str, Any]
    status: str
    latency_ms: float | None
    stdout: str | None
    stderr: str | None
    parsed_payload: dict[str, Any] | None
    session_id: str | None
    reason: str | None


class HermesProposalError(RuntimeError):
    """Raised when Hermes cannot provide a valid mutation proposal."""

    def __init__(self, message: str, *, trace: HermesQueryTrace | None = None) -> None:
        super().__init__(message)
        self.trace = trace


def hermes_backend_ready(config: HermesConfig, *, root: Path) -> bool:
    """Return whether Hermes and the configured local model endpoint look reachable."""
    python_bin = root / config.python_bin
    cli_path = root / config.cli_path
    if not python_bin.exists() or not cli_path.exists():
        return False

    for url in _healthcheck_urls(config.base_url):
        try:
            with urlopen(url, timeout=config.healthcheck_timeout_sec):
                return True
        except (HTTPError, URLError, TimeoutError, ValueError):
            continue
    return False


def choose_mutation_proposal(
    *,
    config: HermesConfig,
    iteration: int,
    metric_key: str,
    metric_direction: str,
    history: list[dict[str, object]],
    candidates: list[MutationProposal],
    incumbent: dict[str, object] | None,
    search_space: dict[str, object] | None,
    mutation_lessons: list[dict[str, object]] | None,
    root: Path,
) -> MutationProposal:
    """Use Hermes to choose the next config mutation from a bounded candidate set."""
    proposal, _trace = choose_mutation_proposal_with_trace(
        config=config,
        iteration=iteration,
        metric_key=metric_key,
        metric_direction=metric_direction,
        history=history,
        candidates=candidates,
        incumbent=incumbent,
        search_space=search_space,
        mutation_lessons=mutation_lessons,
        root=root,
    )
    return proposal


def choose_mutation_proposal_with_trace(
    *,
    config: HermesConfig,
    iteration: int,
    metric_key: str,
    metric_direction: str,
    history: list[dict[str, object]],
    candidates: list[MutationProposal],
    incumbent: dict[str, object] | None,
    search_space: dict[str, object] | None,
    mutation_lessons: list[dict[str, object]] | None,
    root: Path,
) -> tuple[MutationProposal, HermesQueryTrace]:
    """Return the selected proposal plus the underlying Hermes trace."""
    if not candidates:
        raise HermesProposalError("No mutation candidates were provided to Hermes.")

    prompt = _build_prompt(
        iteration=iteration,
        metric_key=metric_key,
        metric_direction=metric_direction,
        history=history,
        candidates=candidates,
        incumbent=incumbent,
        search_space=search_space,
        mutation_lessons=mutation_lessons,
    )
    trace = _run_hermes_query(prompt=prompt, config=config, root=root)
    if trace.stdout is None:
        raise HermesProposalError("Hermes did not produce stdout.", trace=trace)
    try:
        payload = _extract_json_payload(trace.stdout)
    except HermesProposalError as exc:
        if exc.trace is not None:
            raise
        raise HermesProposalError(str(exc), trace=trace) from exc
    index = payload.get("candidate_index")
    if not isinstance(index, int):
        raise HermesProposalError(
            "Hermes response did not include an integer candidate_index.",
            trace=trace,
        )
    if index < 0 or index >= len(candidates):
        max_index = len(candidates) - 1
        raise HermesProposalError(
            f"Hermes selected candidate_index={index}, outside the valid range 0-{max_index}.",
            trace=trace,
        )
    hydrated_trace = HermesQueryTrace(
        prompt=trace.prompt,
        command=trace.command,
        status=trace.status,
        latency_ms=trace.latency_ms,
        stdout=trace.stdout,
        stderr=trace.stderr,
        parsed_payload=payload,
        session_id=trace.session_id,
        reason=payload.get("reason") if isinstance(payload.get("reason"), str) else None,
    )
    return candidates[index], hydrated_trace


def _run_hermes_query(*, prompt: str, config: HermesConfig, root: Path) -> HermesQueryTrace:
    python_bin = root / config.python_bin
    cli_path = root / config.cli_path
    if not python_bin.exists():
        raise HermesProposalError(f"Hermes python not found: {python_bin}")
    if not cli_path.exists():
        raise HermesProposalError(f"Hermes CLI not found: {cli_path}")

    command = [
        str(python_bin),
        str(cli_path),
        "--quiet",
        "-q",
        prompt,
        "--provider",
        config.provider,
        "--base_url",
        config.base_url,
        "--model",
        config.model,
        "--max_turns",
        str(config.max_turns),
    ]
    if config.toolsets:
        command.extend(["--toolsets", ",".join(config.toolsets)])
    api_key = config.api_key or _default_custom_api_key(config)
    if api_key:
        command.extend(["--api_key", api_key])

    env = os.environ.copy()
    hermes_home = (root / config.home_dir).resolve()
    env["HERMES_HOME"] = str(hermes_home)
    _sync_project_skills(hermes_home)

    if config.skills:
        command.extend(["--skills", ",".join(config.skills)])
    if config.pass_session_id:
        command.append("--pass_session_id")

    command_payload = {
        "argv": command,
        "cwd": str(root),
        "env": {"HERMES_HOME": str(hermes_home)},
    }
    start = time.perf_counter()
    wall_clock_start = time.time()

    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        trace = HermesQueryTrace(
            prompt=prompt,
            command=command_payload,
            status="timeout",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            stdout=_decode_maybe_bytes(exc.stdout),
            stderr=_decode_maybe_bytes(exc.stderr),
            parsed_payload=None,
            session_id=_latest_session_id(hermes_home, started_at=wall_clock_start),
            reason="timeout",
        )
        raise HermesProposalError(
            f"Hermes timed out after {config.timeout_sec} seconds.",
            trace=trace,
        ) from exc

    failure_reason = _detect_hermes_failure(stdout=completed.stdout, stderr=completed.stderr)
    trace = HermesQueryTrace(
        prompt=prompt,
        command=command_payload,
        status="ok" if completed.returncode == 0 and failure_reason is None else "error",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        stdout=completed.stdout,
        stderr=completed.stderr,
        parsed_payload=None,
        session_id=_latest_session_id(hermes_home, started_at=wall_clock_start),
        reason=failure_reason,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise HermesProposalError(
            f"Hermes exited with code {completed.returncode}: {stderr}",
            trace=trace,
        )
    if failure_reason is not None:
        raise HermesProposalError(
            f"Hermes did not produce a usable proposal: {failure_reason}",
            trace=trace,
        )

    return trace


def _default_custom_api_key(config: HermesConfig) -> str | None:
    """Provide a harmless dummy key for OpenAI-compatible local servers when needed."""
    if config.provider != "custom":
        return None
    normalized = (config.base_url or "").lower()
    if not normalized:
        return None
    if "11434" in normalized:
        return "ollama"
    return "EMPTY"


def _detect_hermes_failure(*, stdout: str, stderr: str) -> str | None:
    """Detect Hermes CLI failures that still exit with code 0."""
    combined = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part).strip()
    if not combined:
        return None

    patterns = [
        r"API call failed after \d+ retries:\s*(.+)",
        r"Final error:\s*(.+)",
        r"Error:\s*(HTTP \d+:.+)",
        r"error\"\s*:\s*\"([^\"]+)\"",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    lowered = combined.lower()
    if "max retries" in lowered and "giving up" in lowered:
        lines = [line.strip() for line in combined.splitlines() if line.strip()]
        return lines[-1] if lines else "Hermes exhausted retries"
    return None


def _sync_project_skills(hermes_home: Path) -> None:
    if not _PROJECT_SKILLS_DIR.exists():
        return
    skills_root = hermes_home / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    for skill_dir in _PROJECT_SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        destination = skills_root / skill_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(skill_dir, destination)


def _build_prompt(
    *,
    iteration: int,
    metric_key: str,
    metric_direction: str,
    history: list[dict[str, object]],
    candidates: list[MutationProposal],
    incumbent: dict[str, object] | None,
    search_space: dict[str, object] | None,
    mutation_lessons: list[dict[str, object]] | None,
) -> str:
    history_lines = [
        json.dumps(record, separators=(",", ":"), sort_keys=True)
        for record in history[-8:]
    ]
    if not history_lines:
        history_lines = ['{"note":"no prior completed runs"}']

    candidate_lines = [
        (
            f"{index}. description={proposal.description!r} "
            f"overrides={json.dumps(proposal.overrides, separators=(',', ':'))} "
            f"groups={json.dumps(list(proposal.groups), separators=(',', ':'))} "
            f"architecture_change={json.dumps(proposal.architecture_change)}"
        )
        for index, proposal in enumerate(candidates)
    ]
    incumbent_block = (
        json.dumps(incumbent, separators=(",", ":"), sort_keys=True)
        if incumbent is not None
        else '{"note":"no incumbent yet"}'
    )
    search_space_block = (
        json.dumps(search_space, separators=(",", ":"), sort_keys=True)
        if search_space is not None
        else '{"note":"no explicit search-space constraints"}'
    )
    lesson_lines = [
        json.dumps(record, separators=(",", ":"), sort_keys=True)
        for record in (mutation_lessons or [])[:6]
    ]
    if not lesson_lines:
        lesson_lines = ['{"note":"no prior mutation lessons"}']

    return "\n".join(
        [
            "You are selecting the next IMU denoising experiment mutation.",
            f"Iteration: {iteration}",
            f"Objective: {metric_direction} {metric_key}",
            "Current incumbent:",
            incumbent_block,
            "Search-space contract from the local controller:",
            search_space_block,
            "Recent mutation lessons from accepted/rejected prior runs:",
            *lesson_lines,
            "Constraints:",
            "- Choose exactly one candidate from the provided list.",
            "- Do not invent new overrides.",
            "- Respect the search-space contract and incumbent context.",
            (
                "- Prefer candidates that build on the current incumbent "
                "unless branching is explicitly allowed."
            ),
            (
                "- Prefer candidates that are likely to improve validation quality "
                "while keeping the search diverse."
            ),
            (
                'Return ONLY compact JSON in the form '
                '{"candidate_index": <int>, "reason": "<short reason>"}.'
            ),
            "Recent results:",
            *history_lines,
            "Candidates:",
            *candidate_lines,
        ]
    )


def _extract_json_payload(output: str) -> dict[str, Any]:
    clean_output = _ANSI_RE.sub("", output).strip()

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", clean_output, flags=re.DOTALL)
    candidates = fenced if fenced else re.findall(r"\{.*?\}", clean_output, flags=re.DOTALL)
    for raw in reversed(candidates):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise HermesProposalError(f"Hermes did not return parseable JSON. Output was:\n{clean_output}")


def _healthcheck_urls(base_url: str) -> list[str]:
    normalized = base_url.rstrip("/")
    urls = [f"{normalized}/models"]

    parsed = urlparse(normalized)
    if parsed.path.endswith("/v1"):
        api_tags_path = parsed.path[: -len("/v1")] + "/api/tags"
        urls.append(
            urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    api_tags_path,
                    "",
                    "",
                    "",
                )
            )
        )
    return urls


def _decode_maybe_bytes(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _latest_session_id(hermes_home: Path, *, started_at: float) -> str | None:
    sessions_dir = hermes_home / "sessions"
    if not sessions_dir.exists():
        return None
    candidates = sorted(
        sessions_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            if path.stat().st_mtime + 2.0 < started_at:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        session_id = payload.get("session_id")
        if isinstance(session_id, str):
            return session_id
    return None
