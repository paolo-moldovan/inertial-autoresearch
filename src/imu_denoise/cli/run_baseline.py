"""Run classical baselines against the current dataset config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from imu_denoise.classical import ComplementaryFilterBaseline, KalmanFilterBaseline
from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.data.datamodule import create_dataloaders
from imu_denoise.device import DeviceContext
from imu_denoise.evaluation.metrics import compute_all_metrics
from imu_denoise.training.reproducibility import seed_everything
from imu_denoise.utils.io import save_metrics


class BaselineProtocol(Protocol):
    def denoise(self, windows: np.ndarray) -> np.ndarray:
        """Denoise a batch of IMU windows."""


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for baseline evaluation."""
    parser = argparse.ArgumentParser(description="Run a classical IMU denoising baseline.")
    add_common_config_arguments(parser)
    parser.add_argument(
        "--baseline",
        choices=["kalman", "complementary"],
        default="kalman",
        help="Classical baseline to evaluate.",
    )
    return parser


def _build_baseline(name: str) -> BaselineProtocol:
    if name == "kalman":
        return KalmanFilterBaseline()
    if name == "complementary":
        return ComplementaryFilterBaseline()
    raise ValueError(f"Unsupported baseline: {name}")


def main() -> int:
    """Run the selected baseline and save metrics plus sample plots."""
    args = build_parser().parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    from imu_denoise.evaluation.visualization import plot_denoising_comparison, plot_psd

    config = resolve_config(args.config, args.overrides)
    seed_everything(config.training.seed)
    device_ctx = DeviceContext.from_config(config.device)

    _, _, test_loader = create_dataloaders(config.data, config.training, device_ctx)
    baseline = _build_baseline(args.baseline)

    all_noisy: list[np.ndarray] = []
    all_clean: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    sample_timestamps: np.ndarray | None = None

    for batch in test_loader:
        noisy = batch["noisy"].cpu().numpy().astype(np.float32)
        clean = batch["clean"].cpu().numpy().astype(np.float32)
        pred = baseline.denoise(noisy)
        all_noisy.append(noisy)
        all_clean.append(clean)
        all_pred.append(pred)
        if sample_timestamps is None:
            sample_timestamps = batch["timestamps"].cpu().numpy()[0]

    noisy_array = np.concatenate(all_noisy, axis=0)
    clean_array = np.concatenate(all_clean, axis=0)
    pred_array = np.concatenate(all_pred, axis=0)
    metrics = compute_all_metrics(
        pred_array.reshape(-1, pred_array.shape[-1]),
        clean_array.reshape(-1, clean_array.shape[-1]),
        fs=100.0 if config.data.dataset == "blackbird" else 200.0,
    )

    output_dir = Path(config.output_dir) / config.name / "baselines" / args.baseline
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics_path, metrics)

    assert sample_timestamps is not None
    plot_denoising_comparison(
        noisy=noisy_array[0],
        denoised=pred_array[0],
        clean=clean_array[0],
        timestamps=sample_timestamps,
        title=f"{args.baseline} baseline",
        save_path=output_dir / "comparison.png",
    )
    plot_psd(
        signals={"noisy": noisy_array[0], "denoised": pred_array[0], "clean": clean_array[0]},
        fs=100.0 if config.data.dataset == "blackbird" else 200.0,
        title=f"{args.baseline} baseline PSD",
        save_path=output_dir / "psd.png",
    )

    print("Baseline evaluation complete:")
    print(f"  baseline: {args.baseline}")
    print(f"  metrics_path: {metrics_path}")
    print(f"  rmse: {metrics['rmse']:.6f}")
    print(f"  mae: {metrics['mae']:.6f}")
    return 0
