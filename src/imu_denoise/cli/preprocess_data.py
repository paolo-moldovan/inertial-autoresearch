"""Dataset preprocessing CLI."""

from __future__ import annotations

import argparse

from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.data.blackbird.constants import SPEEDS, TRAJECTORIES
from imu_denoise.data.blackbird.preprocess import preprocess_and_save as preprocess_blackbird
from imu_denoise.data.euroc.constants import SEQUENCES
from imu_denoise.data.euroc.preprocess import preprocess_and_save as preprocess_euroc


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for data preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw IMU datasets into cached `.npz` files."
    )
    add_common_config_arguments(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Override dataset name (`euroc`, `blackbird`, or `synthetic`).",
    )
    parser.add_argument(
        "--sequence",
        dest="sequences",
        action="append",
        default=[],
        help="Sequence name to preprocess. Can be repeated.",
    )
    return parser


def _requested_names_from_config(
    dataset_name: str,
    config_sequences: list[str],
    split_names: list[str],
) -> list[str]:
    """Derive requested sequence names from config when CLI filters are absent."""
    requested = config_sequences or split_names
    if requested:
        return requested
    if dataset_name == "euroc":
        return SEQUENCES
    if dataset_name == "blackbird":
        return TRAJECTORIES
    return []


def main() -> int:
    """Preprocess dataset files into the shared processed-cache format."""
    args = build_parser().parse_args()
    config = resolve_config(args.config, args.overrides)
    dataset_name = args.dataset or config.data.dataset

    if dataset_name == "synthetic":
        print("Synthetic data is generated on demand and does not require preprocessing.")
        return 0

    split_names = (
        config.data.train_sequences + config.data.val_sequences + config.data.test_sequences
    )
    requested_names = args.sequences or _requested_names_from_config(
        dataset_name,
        config.data.sequences,
        split_names,
    )

    if dataset_name == "euroc":
        for sequence_name in requested_names:
            path = preprocess_euroc(sequence_name, data_dir=config.data.data_dir)
            print(path)
        return 0

    if dataset_name == "blackbird":
        for sequence_name in requested_names:
            if "_" in sequence_name:
                trajectory, speed = sequence_name.rsplit("_", 1)
                print(preprocess_blackbird(trajectory, speed, data_dir=config.data.data_dir))
                continue

            for speed in SPEEDS:
                print(preprocess_blackbird(sequence_name, speed, data_dir=config.data.data_dir))
        return 0

    raise SystemExit(f"Unsupported dataset for preprocessing: {dataset_name}")


if __name__ == "__main__":
    raise SystemExit(main())
