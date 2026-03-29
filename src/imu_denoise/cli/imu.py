"""Unified `imu` command with subcommands for the core workflow."""

from __future__ import annotations

import argparse
from typing import Any

from imu_denoise.cli.common import add_common_config_arguments
from imu_denoise.cli.dashboard import run_command as run_dashboard_command
from imu_denoise.cli.evaluate import run_command as run_evaluate_command
from imu_denoise.cli.loop_control import (
    add_loop_arguments,
    add_queue_arguments,
    add_status_arguments,
    run_loop_command,
    run_queue_command,
    run_status_command,
)
from imu_denoise.cli.monitor import run_command as run_monitor_command
from imu_denoise.cli.run_baseline import run_command as run_baseline_command
from imu_denoise.cli.train import run_command as run_train_command


def _attach_name_override(args: Any) -> None:
    if getattr(args, "name", ""):
        args.overrides = [*list(args.overrides), f"name={args.name}"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified IMU denoising workflow CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Train a single experiment.")
    add_common_config_arguments(run_parser)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--name", type=str, default="", help="Optional experiment name.")
    run_parser.set_defaults(handler=run_train_command, pre_handler=_attach_name_override)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint.")
    add_common_config_arguments(eval_parser)
    eval_parser.add_argument("--checkpoint", type=str, default="")
    eval_parser.add_argument("--name", type=str, default="", help="Optional evaluation run name.")
    eval_parser.set_defaults(handler=run_evaluate_command, pre_handler=_attach_name_override)

    baseline_parser = subparsers.add_parser("baseline", help="Run a classical baseline.")
    add_common_config_arguments(baseline_parser)
    baseline_parser.add_argument(
        "--baseline",
        choices=["kalman", "complementary"],
        default="kalman",
    )
    baseline_parser.add_argument("--name", type=str, default="", help="Optional run name.")
    baseline_parser.set_defaults(handler=run_baseline_command, pre_handler=_attach_name_override)

    loop_parser = subparsers.add_parser("loop", help="Run or resume the autoresearch loop.")
    add_loop_arguments(loop_parser)
    loop_parser.set_defaults(handler=run_loop_command, pre_handler=None)

    queue_parser = subparsers.add_parser(
        "queue",
        help="Queue a human proposal for the active loop.",
    )
    add_queue_arguments(queue_parser)
    queue_parser.set_defaults(handler=run_queue_command, pre_handler=None)

    status_parser = subparsers.add_parser("status", help="Show active loop status.")
    add_status_arguments(status_parser)
    status_parser.set_defaults(handler=run_status_command, pre_handler=None)

    dashboard_parser = subparsers.add_parser("dashboard", help="Run the Streamlit dashboard.")
    add_common_config_arguments(dashboard_parser)
    dashboard_parser.add_argument("--db-path", type=str, default="")
    dashboard_parser.add_argument("--blob-dir", type=str, default="")
    dashboard_parser.add_argument("--port", type=int, default=0)
    dashboard_parser.set_defaults(handler=run_dashboard_command, pre_handler=None)

    monitor_parser = subparsers.add_parser("monitor", help="Run the Textual TUI.")
    add_common_config_arguments(monitor_parser)
    monitor_parser.add_argument("--db-path", type=str, default="")
    monitor_parser.add_argument("--blob-dir", type=str, default="")
    monitor_parser.add_argument("--refresh-hz", type=int, default=0)
    monitor_parser.set_defaults(handler=run_monitor_command, pre_handler=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pre_handler = getattr(args, "pre_handler", None)
    if callable(pre_handler):
        pre_handler(args)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
