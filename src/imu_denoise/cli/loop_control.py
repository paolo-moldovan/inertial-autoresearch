"""Unified CLI helpers for autoresearch loop control."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from autoresearch_loop.loop import run_autoresearch
from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.observability import LoopAlreadyRunningError, build_mission_control_services


def _build_services(config: Any) -> Any:
    return build_mission_control_services(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )


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
    services = _build_services(config)
    facade = services.facade
    if args.resume:
        resumed = facade.resume_loop()
        if resumed is None:
            print("No paused loop is available to resume.")
            return 1
        print(
            f"Resumed loop {str(resumed['loop_run_id'])[:8]} "
            f"at iteration {resumed['current_iteration']}."
        )
        return 0

    try:
        results = run_autoresearch(
            config_paths=list(args.config),
            base_overrides=list(args.overrides),
            max_iterations=args.max_iterations,
            batch_size=args.batch or None,
            pause_enabled=bool(args.pause),
        )
    except LoopAlreadyRunningError as exc:
        print(
            "Another loop is already active. "
            f"Blocking loop: {exc.blocking_loop_run_id[:8]}"
        )
        return 1
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
    facade = _build_services(config).facade
    proposal = facade.enqueue_proposal(
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
    facade = _build_services(config).facade
    loop_state = facade.get_loop_status()
    if loop_state is None:
        print("No active loop.")
        return 0
    summary = facade.get_summary(limit=1)
    best = summary["best_result"]
    best_text = (
        f"{best['metric_value']:.6f} ({best['run_name']})"
        if isinstance(best, dict) and isinstance(best.get("metric_value"), (int, float))
        else "n/a"
    )
    print(
        f"Loop {loop_state['status']}, iteration {loop_state['current_iteration']}/"
        f"{loop_state['max_iterations']}, best={best_text}, "
        f"flags=pause:{int(bool(loop_state.get('pause_requested')))} "
        f"stop:{int(bool(loop_state.get('stop_requested')))} "
        f"terminate:{int(bool(loop_state.get('terminate_requested')))}"
    )
    return 0


def add_stop_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_config_arguments(parser)
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Force-terminate the active loop and interrupt the current training run.",
    )


def run_stop_command(args: Any) -> int:
    config = resolve_config(args.config, args.overrides)
    facade = _build_services(config).facade
    state = (
        facade.request_terminate()
        if args.terminate
        else facade.request_stop()
    )
    if state is None:
        print("No active loop is available.")
        return 1
    action = "Terminate" if args.terminate else "Stop"
    print(f"{action} requested for loop {str(state['loop_run_id'])[:8]}.")
    return 0


def add_rerun_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_config_arguments(parser)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID or unique ID prefix to requeue for the active loop.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional note stored with the queued rerun.",
    )


def run_rerun_command(args: Any) -> int:
    config = resolve_config(args.config, args.overrides)
    facade = _build_services(config).facade
    match = facade.search_entity(args.run_id)
    if match is None or match["entity_type"] != "run":
        print(f"Could not resolve run ID: {args.run_id}")
        return 1
    run_id = str(match["id"])
    detail = facade.get_run_detail(run_id)
    if detail is None:
        print(f"Run not found: {run_id}")
        return 1
    decisions = detail["decisions"]
    if not decisions:
        print(f"Run {run_id[:8]} has no decision payload to rerun.")
        return 1
    decision = decisions[0]
    proposal = facade.enqueue_proposal(
        description=f"rerun {detail['run']['name']}",
        overrides=list(decision.get("overrides") or []),
        requested_by=os.environ.get("USER") or "human",
        notes=args.notes or f"requeued from run {run_id}",
    )
    print(
        f"Queued rerun #{proposal['id']} for loop {str(proposal['loop_run_id'])[:8]} "
        f"from run {run_id[:8]}."
    )
    return 0
