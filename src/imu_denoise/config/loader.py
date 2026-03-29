"""YAML config loading with hierarchical merge and CLI overrides."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from imu_denoise.config.schema import ExperimentConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_cli_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted CLI overrides like 'training.lr=0.0001'."""
    result = copy.deepcopy(data)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override!r}. Expected 'key=value'.")
        key, value = override.split("=", 1)
        parts = key.split(".")

        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Auto-convert types
        target[parts[-1]] = _parse_value(value)
    return result


def _parse_value(value: str) -> Any:
    """Parse a CLI value string into the appropriate Python type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a frozen dataclass instance."""
    if not is_dataclass(cls):
        return data

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        # Resolve nested dataclass fields by their top-level config name.
        field_type = _resolve_type(cls, f.name)
        if field_type is not None and is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _resolve_type(cls: type, field_name: str) -> type | None:
    """Resolve dataclass field types from annotations."""
    from imu_denoise.config.schema import (
        AutoResearchConfig,
        DataConfig,
        DeviceConfig,
        HermesConfig,
        ModelConfig,
        ObservabilityConfig,
        TrainingConfig,
    )

    if cls is ExperimentConfig:
        type_map: dict[str, type] = {
            "device": DeviceConfig,
            "data": DataConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "autoresearch": AutoResearchConfig,
            "observability": ObservabilityConfig,
        }
        return type_map.get(field_name)

    if cls is AutoResearchConfig:
        nested_type_map: dict[str, type] = {
            "hermes": HermesConfig,
        }
        return nested_type_map.get(field_name)

    return None


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_config(
    *config_paths: str | Path,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load and merge multiple YAML configs, then apply CLI overrides.

    Args:
        *config_paths: YAML files to merge in order (later overrides earlier).
        overrides: CLI overrides in "dotted.key=value" format.

    Returns:
        Fully resolved ExperimentConfig.
    """
    merged: dict[str, Any] = {}
    for path in config_paths:
        path = Path(path)
        if path.exists():
            data = load_yaml(path)
            merged = _deep_merge(merged, data)

    if overrides:
        merged = _apply_cli_overrides(merged, overrides)

    return cast(ExperimentConfig, _dict_to_dataclass(ExperimentConfig, merged))
