"""CLI entrypoint for the Textual mission-control monitor."""

from __future__ import annotations

import argparse
from pathlib import Path

from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.observability.monitor_app import run_monitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the mission-control TUI.")
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
    parser.add_argument("--refresh-hz", type=int, default=0, help="Override refresh rate.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = resolve_config(args.config, args.overrides)
    db_path = Path(args.db_path or config.observability.db_path)
    blob_dir = Path(args.blob_dir or config.observability.blob_dir)
    refresh_hz = args.refresh_hz or config.observability.tui_refresh_hz
    run_monitor(db_path=db_path, blob_dir=blob_dir, refresh_hz=refresh_hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
