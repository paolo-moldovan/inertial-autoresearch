"""Shared CLI helpers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from imu_denoise.config import ExperimentConfig, load_config
from imu_denoise.models import get_model

DEFAULT_CONFIG_PATHS = (
    Path("configs/base.yaml"),
    Path("configs/device.yaml"),
    Path("configs/training/default.yaml"),
    Path("configs/autoresearch.yaml"),
)


def _has_explicit_model_config(config_paths: Sequence[Path], model_name: str) -> bool:
    target = (Path("configs/models") / f"{model_name}.yaml").resolve()
    return any(path.resolve() == target for path in config_paths if path.exists())


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
    explicit_paths = [Path(path) for path in cli_config_paths]
    probe_config = load_config(
        *DEFAULT_CONFIG_PATHS,
        *explicit_paths,
        overrides=list(overrides),
    )
    selected_model = probe_config.model.name
    auto_model_path = Path("configs/models") / f"{selected_model}.yaml"
    config_paths = list(DEFAULT_CONFIG_PATHS)
    if auto_model_path.exists() and not _has_explicit_model_config(explicit_paths, selected_model):
        config_paths.append(auto_model_path)
    config_paths.extend(explicit_paths)
    return load_config(*config_paths, overrides=list(overrides))


def build_model(config: ExperimentConfig) -> Any:
    """Instantiate a built-in model from experiment config."""
    kwargs: dict[str, object] = {
        "hidden_dim": config.model.hidden_dim,
        "num_layers": config.model.num_layers,
        "dropout": config.model.dropout,
    }
    if config.model.name == "lstm":
        kwargs["bidirectional"] = config.model.bidirectional
    elif config.model.name == "transformer":
        kwargs["num_heads"] = config.model.num_heads
    elif config.model.name == "conv1d":
        kwargs["kernel_size"] = config.model.kernel_size
        kwargs["dilation_base"] = config.model.dilation_base
    kwargs.update(config.model.extra)
    return get_model(config.model.name, **kwargs)
