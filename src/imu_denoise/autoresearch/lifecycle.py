"""Loop lifecycle helpers for the IMU autoresearch runtime."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from imu_denoise.observability import LoopController, MissionControlQueries


@dataclass(frozen=True)
class BaselineReference:
    """Resolved baseline policy for a loop run."""

    include_baseline_run: bool
    metric_value: float | None
    run_id: str | None
    description: str


class SupportsMetricResult(Protocol):
    """Structural protocol for result objects that expose an objective metric."""

    @property
    def metric_value(self) -> float | None: ...


def best_metric_from_results(
    results: Sequence[SupportsMetricResult],
    direction: str,
) -> float | None:
    valid = [result.metric_value for result in results if result.metric_value is not None]
    if not valid:
        return None
    return max(valid) if direction == "maximize" else min(valid)


def resolve_baseline_reference(
    *,
    base_config: Any,
    queries: MissionControlQueries,
) -> BaselineReference:
    policy = base_config.autoresearch.baseline.mode

    if policy == "per_loop":
        return BaselineReference(
            include_baseline_run=True,
            metric_value=None,
            run_id=None,
            description="per-loop baseline",
        )

    if policy == "global":
        baseline = queries.find_best_global_incumbent(
            metric_key=base_config.autoresearch.metric_key,
            dataset=base_config.data.dataset,
            direction=base_config.autoresearch.metric_direction,
            reference_config=base_config,
        )
        if baseline is None:
            return BaselineReference(
                include_baseline_run=True,
                metric_value=None,
                run_id=None,
                description="global incumbent not found; falling back to per-loop baseline",
            )
        return BaselineReference(
            include_baseline_run=False,
            metric_value=float(baseline["metric_value"]),
            run_id=str(baseline["run_id"]),
            description=f"global incumbent {str(baseline['run_id'])[:8]}",
        )

    if policy == "manual":
        configured_run_id = base_config.autoresearch.baseline.run_id.strip()
        if not configured_run_id:
            raise ValueError("autoresearch.baseline.run_id is required when mode=manual")
        match = queries.resolve_id_fragment(configured_run_id)
        if match is None or match["entity_type"] != "run":
            run_id = configured_run_id
        else:
            run_id = str(match["id"])
        metric_value = queries.get_run_metric(
            run_id,
            metric_key=base_config.autoresearch.metric_key,
        )
        if metric_value is None:
            raise ValueError(
                f"Could not resolve baseline metric for manual baseline run: {configured_run_id}"
            )
        return BaselineReference(
            include_baseline_run=False,
            metric_value=metric_value,
            run_id=run_id,
            description=f"manual baseline {run_id[:8]}",
        )

    raise ValueError(f"Unsupported autoresearch baseline mode: {policy}")


def wait_while_paused(
    *,
    loop_controller: LoopController,
    loop_run_id: str,
    total_iterations: int,
    batch_size: int | None,
    current_iteration: int,
    best_metric: float | None,
    best_run_id: str | None,
) -> dict[str, Any]:
    while True:
        loop_state = loop_controller.get_loop_state(loop_run_id)
        if loop_state is None:
            raise RuntimeError("Loop state disappeared while waiting for resume.")
        if bool(loop_state.get("stop_requested")) or bool(loop_state.get("terminate_requested")):
            return loop_state
        if loop_state["status"] != "paused":
            return loop_state
        loop_controller.heartbeat(
            loop_run_id=loop_run_id,
            current_iteration=current_iteration,
            max_iterations=total_iterations,
            batch_size=batch_size,
            pause_after_iteration=loop_state.get("pause_after_iteration"),
            pause_requested=bool(loop_state.get("pause_requested")),
            stop_requested=bool(loop_state.get("stop_requested")),
            terminate_requested=bool(loop_state.get("terminate_requested")),
            best_metric=best_metric,
            best_run_id=best_run_id,
            active_child_run_id=None,
            status="paused",
        )
        time.sleep(0.25)


def finish_loop_with_status(
    *,
    observability: Any,
    loop_controller: LoopController,
    loop_run_id: str,
    current_iteration: int,
    max_iterations: int,
    batch_size: int | None,
    best_metric: float | None,
    best_run_id: str | None,
    status: str,
    message: str,
) -> None:
    observability.finish_run(
        run_id=loop_run_id,
        status=status,
        summary={"message": message},
        source="runtime",
    )
    loop_controller.complete_loop(
        loop_run_id=loop_run_id,
        current_iteration=current_iteration,
        max_iterations=max_iterations,
        batch_size=batch_size,
        best_metric=best_metric,
        best_run_id=best_run_id,
        status=status,
    )
