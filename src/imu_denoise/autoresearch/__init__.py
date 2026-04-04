"""IMU-specific adapter layer on top of the reusable autoresearch core."""

from imu_denoise.autoresearch.adapter import IMUProjectAdapter
from imu_denoise.autoresearch.mutations import build_mutation_schedule, default_mutation_pool
from imu_denoise.autoresearch.runtime import (
    _metric_from_summary,
    _resolve_iteration_config,
    _select_mutation_proposal,
    build_parser,
    run_autoresearch,
)

__all__ = [
    "IMUProjectAdapter",
    "_metric_from_summary",
    "_resolve_iteration_config",
    "_select_mutation_proposal",
    "build_mutation_schedule",
    "build_parser",
    "default_mutation_pool",
    "run_autoresearch",
]
