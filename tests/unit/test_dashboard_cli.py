"""Tests for the Mission Control web dashboard CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _pytest.monkeypatch import MonkeyPatch

from imu_denoise.cli.dashboard import build_parser, run_command


def test_dashboard_cli_uses_web_dashboard_runner(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """The dashboard CLI should launch the lightweight web dashboard server."""
    seen: dict[str, Any] = {}

    def _fake_run_web_dashboard(*, db_path: Path, blob_dir: Path, host: str, port: int) -> None:
        seen["db_path"] = db_path
        seen["blob_dir"] = blob_dir
        seen["host"] = host
        seen["port"] = port

    monkeypatch.setattr("imu_denoise.cli.dashboard.run_web_dashboard", _fake_run_web_dashboard)

    args = build_parser().parse_args(
        [
            "--config",
            "configs/training/quick.yaml",
            "--set",
            f"observability.db_path={tmp_path / 'mission_control.db'}",
            "--set",
            f"observability.blob_dir={tmp_path / 'blobs'}",
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
        ]
    )

    assert run_command(args) == 0
    assert seen["db_path"] == (tmp_path / "mission_control.db").resolve()
    assert seen["blob_dir"] == (tmp_path / "blobs").resolve()
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 9999
