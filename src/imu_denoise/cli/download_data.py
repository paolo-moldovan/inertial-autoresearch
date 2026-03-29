"""Dataset download CLI preflight."""

from __future__ import annotations

import argparse

from imu_denoise.cli.common import add_common_config_arguments, resolve_config


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for data download tasks."""
    parser = argparse.ArgumentParser(description="IMU dataset download preflight.")
    add_common_config_arguments(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Override dataset name for download planning.",
    )
    return parser


def main() -> int:
    """Show planned dataset download inputs without mutating vendor data."""
    args = build_parser().parse_args()
    config = resolve_config(args.config, args.overrides)
    dataset_name = args.dataset or config.data.dataset

    print("Data download preflight:")
    print(f"  dataset: {dataset_name}")
    print(f"  data_dir: {config.data.data_dir}")
    print("Concrete dataset download commands land in Phase 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
