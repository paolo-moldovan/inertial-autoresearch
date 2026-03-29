"""Shared CLI helpers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from imu_denoise.config import ExperimentConfig, load_config

DEFAULT_CONFIG_PATHS = (
    Path("configs/base.yaml"),
    Path("configs/device.yaml"),
    Path("configs/training/default.yaml"),
    Path("configs/autoresearch.yaml"),
)


def add_common_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared config arguments used by package CLI commands."""
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional YAML config file(s) merged after the defaults.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="CLI override in dotted.key=value form. May be passed multiple times.",
    )


def resolve_config(cli_config_paths: Sequence[str], overrides: Sequence[str]) -> ExperimentConfig:
    """Load the experiment config from defaults plus CLI additions."""
    config_paths = [*DEFAULT_CONFIG_PATHS, *(Path(path) for path in cli_config_paths)]
    return load_config(*config_paths, overrides=list(overrides))
