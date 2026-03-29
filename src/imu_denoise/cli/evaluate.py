"""Evaluation CLI preflight."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from imu_denoise.cli.common import add_common_config_arguments, build_model, resolve_config
from imu_denoise.data.datamodule import create_dataloaders
from imu_denoise.device import DeviceContext
from imu_denoise.evaluation.evaluator import Evaluator
from imu_denoise.training.reproducibility import seed_everything
from imu_denoise.utils.io import load_checkpoint, save_metrics


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
    """Evaluate a checkpoint and save metrics/figures."""
    args = build_parser().parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    from imu_denoise.evaluation.visualization import plot_denoising_comparison, plot_psd

    config = resolve_config(args.config, args.overrides)
    seed_everything(config.training.seed)
    device_ctx = DeviceContext.from_config(config.device)
    model = build_model(config)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else Path(config.checkpoint_dir) / "best.pt"
    )

    _, _, test_loader = create_dataloaders(config.data, config.training, device_ctx)
    evaluator = Evaluator(model, device_ctx)
    load_checkpoint(checkpoint_path, evaluator.model, device=device_ctx.device)
    sampling_rate = 100.0 if config.data.dataset == "blackbird" else 200.0
    metrics = evaluator.evaluate(test_loader, fs=sampling_rate)

    output_dir = Path(config.output_dir) / config.name / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics_path, metrics)

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
        save_path=output_dir / "denoising_comparison.png",
    )
    plot_psd(
        signals={
            "noisy": noisy.cpu().float().numpy()[0],
            "denoised": pred[0],
            "clean": clean[0],
        },
        fs=sampling_rate,
        save_path=output_dir / "psd.png",
    )

    print("Evaluation complete:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  metrics_path: {metrics_path}")
    print(f"  rmse: {metrics['rmse']:.6f}")
    print(f"  mae: {metrics['mae']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
