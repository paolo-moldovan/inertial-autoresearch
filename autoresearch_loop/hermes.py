"""Hermes + Ollama integration for config-first autoresearch."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

from autoresearch_loop.mutations import MutationProposal
from imu_denoise.config import HermesConfig

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class HermesProposalError(RuntimeError):
    """Raised when Hermes cannot provide a valid mutation proposal."""


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
    root: Path,
) -> MutationProposal:
    """Use Hermes to choose the next config mutation from a bounded candidate set."""
    if not candidates:
        raise HermesProposalError("No mutation candidates were provided to Hermes.")

    prompt = _build_prompt(
        iteration=iteration,
        metric_key=metric_key,
        metric_direction=metric_direction,
        history=history,
        candidates=candidates,
    )
    response = _run_hermes_query(prompt=prompt, config=config, root=root)
    payload = _extract_json_payload(response)
    index = payload.get("candidate_index")
    if not isinstance(index, int):
        raise HermesProposalError("Hermes response did not include an integer candidate_index.")
    if index < 0 or index >= len(candidates):
        max_index = len(candidates) - 1
        raise HermesProposalError(
            f"Hermes selected candidate_index={index}, outside the valid range 0-{max_index}."
        )
    return candidates[index]


def _run_hermes_query(*, prompt: str, config: HermesConfig, root: Path) -> str:
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
    if config.api_key:
        command.extend(["--api_key", config.api_key])

    env = os.environ.copy()
    env["HERMES_HOME"] = str((root / config.home_dir).resolve())

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
        raise HermesProposalError(
            f"Hermes timed out after {config.timeout_sec} seconds."
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise HermesProposalError(f"Hermes exited with code {completed.returncode}: {stderr}")

    return completed.stdout


def _build_prompt(
    *,
    iteration: int,
    metric_key: str,
    metric_direction: str,
    history: list[dict[str, object]],
    candidates: list[MutationProposal],
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
            f"overrides={json.dumps(proposal.overrides, separators=(',', ':'))}"
        )
        for index, proposal in enumerate(candidates)
    ]

    return "\n".join(
        [
            "You are selecting the next IMU denoising experiment mutation.",
            f"Iteration: {iteration}",
            f"Objective: {metric_direction} {metric_key}",
            "Constraints:",
            "- Choose exactly one candidate from the provided list.",
            "- Do not invent new overrides.",
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
