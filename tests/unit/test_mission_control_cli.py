"""Tests for the tmuxinator mission-control launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from _pytest.monkeypatch import MonkeyPatch

from imu_denoise.cli.start_mission_control import main


def test_start_mission_control_dry_run_prints_command(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The launcher should print the tmuxinator command in dry-run mode."""

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/tmuxinator")
    monkeypatch.setattr("sys.argv", ["imu-mission-control", "--dry-run"])

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "tmuxinator start -p" in captured.out


def test_start_mission_control_runs_tmuxinator(monkeypatch: MonkeyPatch) -> None:
    """The launcher should invoke tmuxinator with the repo-local project file."""

    seen: dict[str, Any] = {}

    def _run(
        command: list[str],
        env: dict[str, str],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen["env"] = env
        seen["check"] = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/tmuxinator")
    monkeypatch.setattr("subprocess.run", _run)
    monkeypatch.setattr("sys.argv", ["imu-mission-control"])

    exit_code = main()

    assert exit_code == 0
    assert seen["command"][:3] == [
        "/usr/local/bin/tmuxinator",
        "start",
        "-p",
    ]
    assert Path(seen["env"]["IMU_AUTORESEARCH_ROOT"]).name == "inertial-autoresearch"


def test_start_mission_control_uses_euroc_profile(monkeypatch: MonkeyPatch) -> None:
    """The launcher should switch project files when the EuRoC profile is requested."""
    seen: dict[str, Any] = {}

    def _run(
        command: list[str],
        env: dict[str, str],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen["env"] = env
        seen["check"] = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/tmuxinator")
    monkeypatch.setattr("subprocess.run", _run)
    monkeypatch.setattr("sys.argv", ["imu-mission-control", "--profile", "euroc"])

    exit_code = main()

    assert exit_code == 0
    assert seen["command"][-1].endswith(".tmuxinator/mission-control-euroc.yml")
