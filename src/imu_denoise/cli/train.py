"""Training CLI preflight."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.device import DeviceContext
from imu_denoise.models import get_model


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the train CLI."""
    parser = argparse.ArgumentParser(description="IMU denoising training preflight.")
    add_common_config_arguments(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and model construction without starting training.",
    )
    return parser


def _model_kwargs(model_name: str, config: Any) -> dict[str, object]:
    """Build constructor kwargs for the selected built-in model."""
    common: dict[str, object] = {
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    }
    if model_name == "lstm":
        common["bidirectional"] = config.bidirectional
    elif model_name == "transformer":
        common["num_heads"] = config.num_heads
    elif model_name == "conv1d":
        common["kernel_size"] = config.kernel_size
        common["dilation_base"] = config.dilation_base
    common.update(config.extra)
    return common


def main() -> int:
    """Validate configuration, device selection, and model creation."""
    args = build_parser().parse_args()
    config = resolve_config(args.config, args.overrides)
    device_ctx = DeviceContext.from_config(config.device)
    model = get_model(config.model.name, **_model_kwargs(config.model.name, config.model))

    checkpoint_dir = Path(config.checkpoint_dir)
    figures_dir = Path(config.figures_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Resolved experiment configuration:")
    print(f"  name: {config.name}")
    print(f"  dataset: {config.data.dataset}")
    print(f"  model: {config.model.name}")
    print(f"  device: {device_ctx.device.type}")
    print(f"  dtype: {device_ctx.dtype}")
    print(f"  amp_enabled: {device_ctx.amp_enabled}")
    print(f"  pin_memory: {device_ctx.pin_memory}")
    print(f"  checkpoint_dir: {checkpoint_dir}")
    print(f"  figures_dir: {figures_dir}")
    print(f"  model_class: {model.__class__.__name__}")

    if not args.dry_run:
        raise SystemExit(
            "Training loop is not implemented yet. Use --dry-run during Phase 1 foundation work."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
