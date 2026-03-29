"""Tests for config loading and CLI override behavior."""

from __future__ import annotations

from pathlib import Path

from imu_denoise.config import ExperimentConfig, load_config


def test_load_config_merges_project_defaults() -> None:
    """Project config fragments should merge into one typed experiment config."""
    config = load_config(
        Path("configs/base.yaml"),
        Path("configs/device.yaml"),
        Path("configs/models/lstm.yaml"),
        Path("configs/training/quick.yaml"),
        Path("configs/autoresearch.yaml"),
    )

    assert isinstance(config, ExperimentConfig)
    assert config.model.name == "lstm"
    assert config.training.epochs == 2
    assert config.data.dataset == "synthetic"
    assert config.device.preferred == "auto"


def test_load_config_applies_cli_overrides() -> None:
    """Dotted overrides should update nested config values with parsed types."""
    config = load_config(
        Path("configs/base.yaml"),
        overrides=[
            "name=phase1-check",
            "training.lr=0.0005",
            "training.epochs=3",
            "device.compile=true",
        ],
    )

    assert config.name == "phase1-check"
    assert config.training.lr == 0.0005
    assert config.training.epochs == 3
    assert config.device.compile is True


def test_checkpoint_and_figure_paths_are_derived() -> None:
    """Derived output paths should stay consistent with the experiment name."""
    config = load_config(
        Path("configs/base.yaml"),
        overrides=["name=test-run", "output_dir=custom_artifacts"],
    )

    assert config.checkpoint_dir == Path("custom_artifacts/checkpoints/test-run")
    assert config.figures_dir == Path("custom_artifacts/figures/test-run")
