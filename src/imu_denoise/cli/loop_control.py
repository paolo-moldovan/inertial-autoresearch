"""Unified CLI helpers for autoresearch loop control."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from autoresearch_loop.loop import run_autoresearch
from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.observability import LoopController, MissionControlQueries, ObservabilityWriter


def add_loop_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_config_arguments(parser)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override autoresearch.max_iterations.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Pause after this many completed runs when --pause is enabled.",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Enable batch pause mode for review and resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the current paused loop instead of starting a new one.",
    )


def run_loop_command(args: Any) -> int:
    config = resolve_config(args.config, args.overrides)
    writer = ObservabilityWriter.from_experiment_config(config)
    controller = LoopController.from_experiment_config(config, writer=writer)
    if args.resume:
        resumed = controller.resume_loop()
        if resumed is None:
            print("No paused loop is available to resume.")
            return 1
        print(
            f"Resumed loop {str(resumed['loop_run_id'])[:8]} "
            f"at iteration {resumed['current_iteration']}."
        )
        return 0

    results = run_autoresearch(
        config_paths=list(args.config),
        base_overrides=list(args.overrides),
        max_iterations=args.max_iterations,
        batch_size=args.batch or None,
        pause_enabled=bool(args.pause),
    )
    print(f"Completed {len(results)} autoresearch runs.")
    return 0


def add_queue_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional YAML config file(s) used to resolve observability paths.",
    )
    parser.add_argument(
        "--set",
        dest="proposal_overrides",
        action="append",
        default=[],
        help="Queued proposal override in dotted.key=value form. May be passed multiple times.",
    )
    parser.add_argument("description", type=str, help="Human description for the queued proposal.")
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional note stored alongside the queued proposal.",
    )


def run_queue_command(args: Any) -> int:
    config = resolve_config(args.config, [])
    writer = ObservabilityWriter.from_experiment_config(config)
    controller = LoopController.from_experiment_config(config, writer=writer)
    proposal = controller.enqueue_proposal(
        description=args.description,
        overrides=list(args.proposal_overrides),
        requested_by=os.environ.get("USER") or "human",
        notes=args.notes or None,
    )
    print(
        f"Queued proposal #{proposal['id']} for loop {str(proposal['loop_run_id'])[:8]}: "
        f"{proposal['description']}"
    )
    return 0


def add_status_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_config_arguments(parser)


def run_status_command(args: Any) -> int:
    config = resolve_config(args.config, args.overrides)
    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )
    loop_state = queries.get_loop_status()
    if loop_state is None:
        print("No active loop.")
        return 0
    summary = queries.get_mission_control_summary(limit=1)
    best = summary["best_result"]
    best_text = (
        f"{best['metric_value']:.6f} ({best['run_name']})"
        if isinstance(best, dict) and isinstance(best.get("metric_value"), (int, float))
        else "n/a"
    )
    print(
        f"Loop {loop_state['status']}, iteration {loop_state['current_iteration']}/"
        f"{loop_state['max_iterations']}, best={best_text}"
    )
    return 0
