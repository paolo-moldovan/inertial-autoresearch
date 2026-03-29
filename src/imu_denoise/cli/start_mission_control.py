"""Launch the mission-control tmuxinator session."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the mission-control tmuxinator session.")
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
        else repo_root / ".tmuxinator" / "mission-control.yml"
    )
    if not project_file.exists():
        raise FileNotFoundError(f"tmuxinator project file not found: {project_file}")

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

