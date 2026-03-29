"""Launch the mission-control tmuxinator session."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

PROFILE_TO_PROJECT = {
    "smoke": "mission-control.yml",
    "euroc": "mission-control-euroc.yml",
}


def _project_session_name(project_file: Path) -> str | None:
    for line in project_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            _, value = stripped.split(":", 1)
            session_name = value.strip()
            return session_name or None
    return None


def _tmux_session_exists(session_name: str) -> bool:
    tmux_bin = shutil.which("tmux")
    if tmux_bin is None:
        return False
    completed = subprocess.run(
        [tmux_bin, "has-session", "-t", session_name],
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the mission-control tmuxinator session.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_TO_PROJECT),
        default="smoke",
        help="Named Mission Control profile to launch.",
    )
    parser.add_argument(
        "--project-file",
        type=str,
        default="",
        help="Override the tmuxinator project file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the tmuxinator command without starting it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    project_file = (
        Path(args.project_file)
        if args.project_file
        else repo_root / ".tmuxinator" / PROFILE_TO_PROJECT[args.profile]
    )
    if not project_file.exists():
        raise FileNotFoundError(f"tmuxinator project file not found: {project_file}")

    session_name = _project_session_name(project_file)
    if session_name and _tmux_session_exists(session_name):
        print(f"Mission Control session already exists: {session_name}")
        return 1

    tmuxinator_bin = shutil.which("tmuxinator")
    if tmuxinator_bin is None:
        raise RuntimeError(
            "tmuxinator is not installed or not on PATH. Install it first, then rerun."
        )

    command = [tmuxinator_bin, "start", "-p", str(project_file.resolve())]
    if args.dry_run:
        print(" ".join(command))
        return 0

    env = os.environ.copy()
    env["IMU_AUTORESEARCH_ROOT"] = str(repo_root)
    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
