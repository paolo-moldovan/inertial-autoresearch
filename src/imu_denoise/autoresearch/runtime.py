"""Compatibility wrapper for the IMU-domain autoresearch runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch_core import recent_policy_results, result_snapshot

from .artifacts import (
    append_result,
    ensure_results_file,
    resolve_results_file,
    safe_update_run_manifest,
)
from .execution import (
    metric_from_summary,
    resolve_iteration_config,
    run_single_experiment,
)
from .iteration import build_run_overrides
from .lifecycle import (
    BaselineReference,
    best_metric_from_results,
    finish_loop_with_status,
    resolve_baseline_reference,
    wait_while_paused,
)
from .runner import LoopResult, build_parser, main, run_autoresearch
from .selection import select_mutation_proposal

ROOT = Path(__file__).resolve().parents[3]


def _metric_from_summary(summary: Any, metric_key: str) -> float:
    return metric_from_summary(summary, metric_key)


def _resolve_iteration_config(*args: Any, **kwargs: Any) -> Any:
    return resolve_iteration_config(*args, **kwargs)


def _run_single_experiment(*args: Any, **kwargs: Any) -> Any:
    return run_single_experiment(*args, **kwargs)


def _ensure_results_file(*args: Any, **kwargs: Any) -> Any:
    return ensure_results_file(*args, **kwargs)


def _append_result(*args: Any, **kwargs: Any) -> Any:
    return append_result(*args, **kwargs)


def _build_run_overrides(*args: Any, **kwargs: Any) -> Any:
    return build_run_overrides(*args, **kwargs)


def _result_snapshot(*args: Any, **kwargs: Any) -> Any:
    return result_snapshot(*args, **kwargs)


def _recent_policy_results(*args: Any, **kwargs: Any) -> Any:
    return recent_policy_results(*args, **kwargs)


def _select_mutation_proposal(
    *,
    resolve_iteration_config_fn: Any | None = None,
    **kwargs: Any,
) -> Any:
    return select_mutation_proposal(
        root=ROOT,
        resolve_iteration_config_fn=resolve_iteration_config_fn or _resolve_iteration_config,
        **kwargs,
    )


def _best_metric_from_results(*args: Any, **kwargs: Any) -> Any:
    return best_metric_from_results(*args, **kwargs)


def _resolve_baseline_reference(*args: Any, **kwargs: Any) -> Any:
    return resolve_baseline_reference(*args, **kwargs)


def _wait_while_paused(*args: Any, **kwargs: Any) -> Any:
    return wait_while_paused(*args, **kwargs)


def _finish_loop_with_status(*args: Any, **kwargs: Any) -> Any:
    return finish_loop_with_status(*args, **kwargs)


def _resolve_results_file(*args: Any, **kwargs: Any) -> Any:
    return resolve_results_file(*args, **kwargs)


def _safe_update_run_manifest(*args: Any, **kwargs: Any) -> Any:
    return safe_update_run_manifest(*args, **kwargs)

__all__ = [
    "BaselineReference",
    "LoopResult",
    "_append_result",
    "_best_metric_from_results",
    "_build_run_overrides",
    "_ensure_results_file",
    "_finish_loop_with_status",
    "_metric_from_summary",
    "_recent_policy_results",
    "_resolve_baseline_reference",
    "_resolve_iteration_config",
    "_resolve_results_file",
    "_result_snapshot",
    "_run_single_experiment",
    "_safe_update_run_manifest",
    "_select_mutation_proposal",
    "_wait_while_paused",
    "build_parser",
    "main",
    "run_autoresearch",
]


if __name__ == "__main__":
    raise SystemExit(main())
