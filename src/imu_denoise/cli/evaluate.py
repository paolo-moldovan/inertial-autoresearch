"""Evaluation CLI preflight."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from imu_denoise.cli.common import add_common_config_arguments, build_model, resolve_config
from imu_denoise.data.datamodule import create_dataloaders
from imu_denoise.device import DeviceContext
from imu_denoise.evaluation.evaluator import Evaluator
from imu_denoise.observability import MissionControlQueries, ObservabilityWriter
from imu_denoise.observability.lineage import data_regime_fingerprint
from imu_denoise.training.reproducibility import seed_everything
from imu_denoise.utils.io import load_checkpoint, save_metrics
from imu_denoise.utils.paths import build_run_paths, write_run_manifest


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


def _resolve_checkpoint_path(config: Any, args_checkpoint: str) -> Path:
    if args_checkpoint:
        return Path(args_checkpoint)

    queries = MissionControlQueries(
        db_path=Path(config.observability.db_path),
        blob_dir=Path(config.observability.blob_dir),
    )
    for run in queries.list_runs(limit=500):
        if run["phase"] != "training" or run["status"] != "completed":
            continue
        if run["name"] != config.name:
            continue
        for artifact in queries.list_artifacts(run_id=str(run["id"])):
            if artifact["artifact_type"] != "checkpoint" or artifact.get("label") != "best":
                continue
            path = Path(str(artifact["path"]))
            if path.exists():
                return path

    legacy_path = Path(config.checkpoint_dir) / "best.pt"
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        "No checkpoint was provided and no matching completed training run checkpoint was found."
    )


def run_command(args: Any) -> int:
    """Evaluate a checkpoint and save metrics/figures."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    from imu_denoise.evaluation.visualization import plot_denoising_comparison, plot_psd

    config = resolve_config(args.config, args.overrides)
    seed_everything(config.training.seed)
    device_ctx = DeviceContext.from_config(config.device)
    model = build_model(config)
    observability = ObservabilityWriter.from_experiment_config(config)
    run_name = f"{config.name}-evaluation"
    run_id = observability.make_run_id(name=run_name, phase="evaluation")
    run_paths = build_run_paths(config.output_dir, run_name=run_name, run_id=run_id)
    run_id = observability.start_run(
        name=run_name,
        phase="evaluation",
        dataset=config.data.dataset,
        model=config.model.name,
        device=device_ctx.device.type,
        config=config,
        source="runtime",
        run_id=run_id,
    )
    write_run_manifest(
        run_paths,
        {
            "run_id": run_id,
            "name": run_name,
            "phase": "evaluation",
            "regime_fingerprint": data_regime_fingerprint(config),
        },
    )
    checkpoint_path = _resolve_checkpoint_path(config, args.checkpoint)

    _, _, test_loader = create_dataloaders(config.data, config.training, device_ctx)
    evaluator = Evaluator(model, device_ctx)
    load_checkpoint(checkpoint_path, evaluator.model, device=device_ctx.device)
    sampling_rate = 100.0 if config.data.dataset == "blackbird" else 200.0
    metrics = evaluator.evaluate(test_loader, fs=sampling_rate)

    run_paths.root.mkdir(parents=True, exist_ok=True)
    run_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_paths.metrics_path
    save_metrics(metrics_path, metrics)
    observability.register_artifact(
        run_id=run_id,
        path=checkpoint_path,
        artifact_type="checkpoint",
        label="evaluation_checkpoint",
        source="runtime",
    )
    observability.register_artifact(
        run_id=run_id,
        path=metrics_path,
        artifact_type="evaluation_metrics",
        label="evaluation_metrics",
        metadata=metrics,
        source="runtime",
    )

    sample_batch = next(iter(test_loader))
    noisy = sample_batch["noisy"].to(device_ctx.device)
    with torch.no_grad():
        pred = evaluator.model(noisy).cpu().float().numpy()

    clean = sample_batch["clean"].cpu().float().numpy()
    timestamps = sample_batch["timestamps"].cpu().float().numpy()

    plot_denoising_comparison(
        noisy=noisy.cpu().float().numpy()[0],
        denoised=pred[0],
        clean=clean[0],
        timestamps=timestamps[0],
        save_path=run_paths.figures_dir / "denoising_comparison.png",
    )
    observability.register_artifact(
        run_id=run_id,
        path=run_paths.figures_dir / "denoising_comparison.png",
        artifact_type="figure",
        label="denoising_comparison",
        source="runtime",
    )
    plot_psd(
        signals={
            "noisy": noisy.cpu().float().numpy()[0],
            "denoised": pred[0],
            "clean": clean[0],
        },
        fs=sampling_rate,
        save_path=run_paths.figures_dir / "psd.png",
    )
    observability.register_artifact(
        run_id=run_id,
        path=run_paths.figures_dir / "psd.png",
        artifact_type="figure",
        label="psd",
        source="runtime",
    )
    observability.finish_run(
        run_id=run_id,
        status="completed",
        summary={"rmse": metrics["rmse"], "mae": metrics["mae"]},
        source="runtime",
    )

    print("Evaluation complete:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  metrics_path: {metrics_path}")
    print(f"  rmse: {metrics['rmse']:.6f}")
    print(f"  mae: {metrics['mae']:.6f}")
    return 0


def main() -> int:
    """CLI entrypoint."""
    return run_command(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
