"""CLI entrypoint for the Streamlit mission-control dashboard."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from imu_denoise.cli.common import add_common_config_arguments, resolve_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the mission-control Streamlit dashboard.")
    add_common_config_arguments(parser)
    parser.add_argument(
        "--db-path",
        type=str,
        default="",
        help="Override observability SQLite path.",
    )
    parser.add_argument(
        "--blob-dir",
        type=str,
        default="",
        help="Override observability blob path.",
    )
    parser.add_argument("--port", type=int, default=0, help="Override Streamlit port.")
    return parser


def run_command(args: Any) -> int:
    config = resolve_config(args.config, args.overrides)
    db_path = Path(args.db_path or config.observability.db_path).resolve()
    blob_dir = Path(args.blob_dir or config.observability.blob_dir).resolve()
    port = args.port or config.observability.streamlit_port
    os.environ["IMU_DASHBOARD_DB_PATH"] = str(db_path)
    os.environ["IMU_DASHBOARD_BLOB_DIR"] = str(blob_dir)

    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Streamlit is not installed. Install `imu-denoise[monitor]` to use imu-dashboard."
        ) from exc

    app_path = Path(__file__).resolve().parents[1] / "observability" / "dashboard_app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    result = stcli.main()
    return 0 if result is None else int(result)


def main() -> int:
    return run_command(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
