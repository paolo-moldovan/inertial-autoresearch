"""Dataset download CLI preflight."""

from __future__ import annotations

import argparse

from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.data.blackbird.constants import SPEEDS
from imu_denoise.data.blackbird.download import download_all, download_blackbird_sequence


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
    parser.add_argument(
        "--sequence",
        dest="sequences",
        action="append",
        default=[],
        help="Dataset sequence or trajectory to download. Can be repeated.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist.",
    )
    return parser


def main() -> int:
    """Download supported datasets into the project data directory."""
    args = build_parser().parse_args()
    config = resolve_config(args.config, args.overrides)
    dataset_name = args.dataset or config.data.dataset

    if dataset_name == "blackbird":
        requested = args.sequences or config.data.sequences
        if not requested:
            downloaded = download_all(data_dir=config.data.data_dir, overwrite=args.overwrite)
            print(f"Downloaded {len(downloaded)} Blackbird sequence directories.")
            return 0

        count = 0
        for sequence_name in requested:
            if "_" in sequence_name:
                trajectory, speed = sequence_name.rsplit("_", 1)
                download_blackbird_sequence(
                    trajectory,
                    speed,
                    data_dir=config.data.data_dir,
                    overwrite=args.overwrite,
                )
                count += 1
                continue

            for speed in SPEEDS:
                download_blackbird_sequence(
                    sequence_name,
                    speed,
                    data_dir=config.data.data_dir,
                    overwrite=args.overwrite,
                )
                count += 1
        print(f"Downloaded {count} Blackbird sequence directories.")
        return 0

    if dataset_name == "euroc":
        print("EuRoC download is not automated in-repo yet.")
        print("Place raw files under data/raw/euroc/<sequence>/mav0/... then run preprocess_data.")
        return 0

    if dataset_name == "synthetic":
        print("Synthetic data is generated on demand and does not need downloading.")
        return 0

    raise SystemExit(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    raise SystemExit(main())
