"""CLI entrypoint for the Mission Control web dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.observability.web_dashboard import run_web_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the mission-control web dashboard.")
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
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=0, help="Override dashboard port.")
    return parser


def run_command(args: Any) -> int:
    config = resolve_config(args.config, args.overrides)
    db_path = Path(args.db_path or config.observability.db_path).resolve()
    blob_dir = Path(args.blob_dir or config.observability.blob_dir).resolve()
    port = args.port or config.observability.streamlit_port
    run_web_dashboard(
        db_path=db_path,
        blob_dir=blob_dir,
        host=args.host,
        port=port,
    )
    return 0


def main() -> int:
    return run_command(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
