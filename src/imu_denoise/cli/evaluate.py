"""Evaluation CLI preflight."""

from __future__ import annotations

import argparse

from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.device import DeviceContext


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the evaluate CLI."""
    parser = argparse.ArgumentParser(description="IMU denoising evaluation preflight.")
    add_common_config_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to validate.",
    )
    return parser


def main() -> int:
    """Validate evaluation configuration and runtime environment."""
    args = build_parser().parse_args()
    config = resolve_config(args.config, args.overrides)
    device_ctx = DeviceContext.from_config(config.device)

    print("Evaluation preflight:")
    print(f"  name: {config.name}")
    print(f"  dataset: {config.data.dataset}")
    print(f"  device: {device_ctx.device.type}")
    print(f"  checkpoint: {args.checkpoint or '<not provided>'}")
    print("Evaluator wiring lands in Phase 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
