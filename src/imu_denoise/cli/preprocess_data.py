"""Dataset preprocessing CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.data.blackbird.constants import SPEEDS, TRAJECTORIES
from imu_denoise.data.blackbird.preprocess import preprocess_and_save as preprocess_blackbird
from imu_denoise.data.euroc.constants import SEQUENCES
from imu_denoise.data.euroc.preprocess import preprocess_and_save as preprocess_euroc
from imu_denoise.observability import ObservabilityWriter
from imu_denoise.observability.lineage import data_regime_fingerprint
from imu_denoise.utils.paths import build_run_paths, write_run_manifest


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
    observability = ObservabilityWriter.from_experiment_config(config)
    run_name = f"preprocess-{dataset_name}"
    run_id = observability.make_run_id(name=run_name, phase="preprocess")
    run_paths = build_run_paths(config.output_dir, run_name=run_name, run_id=run_id)
    run_id = observability.start_run(
        name=run_name,
        phase="preprocess",
        dataset=dataset_name,
        model="preprocess",
        device="cpu",
        config=config,
        source="runtime",
        run_id=run_id,
    )
    write_run_manifest(
        run_paths,
        {
            "run_id": run_id,
            "name": run_name,
            "phase": "preprocess",
            "regime_fingerprint": data_regime_fingerprint(config),
        },
    )
    processed_count = 0
    diagnostics_count = 0

    def _register_processed_artifacts(path: Path) -> None:
        nonlocal processed_count, diagnostics_count
        processed_count += 1
        observability.register_artifact(
            run_id=run_id,
            path=path,
            artifact_type="processed_dataset",
            label=path.stem,
            source="runtime",
        )
        metadata_path = path.with_suffix(".metadata.json")
        if metadata_path.exists():
            diagnostics_count += 1
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            observability.register_artifact(
                run_id=run_id,
                path=metadata_path,
                artifact_type="dataset_diagnostics",
                label=f"{path.stem}_diagnostics",
                metadata=metadata_payload,
                source="runtime",
            )

    try:
        if dataset_name == "euroc":
            for sequence_name in requested_names:
                path = preprocess_euroc(sequence_name, data_dir=config.data.data_dir)
                print(path)
                _register_processed_artifacts(path)
            observability.finish_run(
                run_id=run_id,
                status="completed",
                summary={
                    "processed_sequences": processed_count,
                    "diagnostic_artifacts": diagnostics_count,
                },
                source="runtime",
            )
            return 0

        if dataset_name == "blackbird":
            for sequence_name in requested_names:
                if "_" in sequence_name:
                    trajectory, speed = sequence_name.rsplit("_", 1)
                    path = preprocess_blackbird(trajectory, speed, data_dir=config.data.data_dir)
                    print(path)
                    _register_processed_artifacts(path)
                    continue

                for speed in SPEEDS:
                    path = preprocess_blackbird(sequence_name, speed, data_dir=config.data.data_dir)
                    print(path)
                    _register_processed_artifacts(path)
            observability.finish_run(
                run_id=run_id,
                status="completed",
                summary={
                    "processed_sequences": processed_count,
                    "diagnostic_artifacts": diagnostics_count,
                },
                source="runtime",
            )
            return 0

        raise SystemExit(f"Unsupported dataset for preprocessing: {dataset_name}")
    except Exception:
        observability.finish_run(run_id=run_id, status="failed", source="runtime")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
