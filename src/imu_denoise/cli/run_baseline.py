"""Run classical baselines against the current dataset config."""

from __future__ import annotations

import argparse
import os
from typing import Any, Protocol

import numpy as np

from imu_denoise.classical import ComplementaryFilterBaseline, KalmanFilterBaseline
from imu_denoise.cli.common import add_common_config_arguments, resolve_config
from imu_denoise.data.datamodule import create_dataloaders
from imu_denoise.device import DeviceContext
from imu_denoise.evaluation.metrics import compute_all_metrics
from imu_denoise.observability import ObservabilityWriter
from imu_denoise.observability.lineage import data_regime_fingerprint
from imu_denoise.training.reproducibility import seed_everything
from imu_denoise.utils.io import save_metrics
from imu_denoise.utils.paths import build_run_paths, update_run_manifest, write_run_manifest


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


def run_command(args: Any) -> int:
    """Run the selected baseline and save metrics plus sample plots."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    from imu_denoise.evaluation.visualization import plot_denoising_comparison, plot_psd

    config = resolve_config(args.config, args.overrides)
    seed_everything(config.training.seed)
    device_ctx = DeviceContext.from_config(config.device)
    observability = ObservabilityWriter.from_experiment_config(config)
    run_name = f"{config.name}-{args.baseline}"
    run_id = observability.make_run_id(name=run_name, phase="baseline")
    run_paths = build_run_paths(config.output_dir, run_name=run_name, run_id=run_id)
    run_id = observability.start_run(
        name=run_name,
        phase="baseline",
        dataset=config.data.dataset,
        model=args.baseline,
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
            "phase": "baseline",
            "baseline": args.baseline,
            "regime_fingerprint": data_regime_fingerprint(config),
        },
    )

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

    run_paths.root.mkdir(parents=True, exist_ok=True)
    run_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_paths.metrics_path
    save_metrics(metrics_path, metrics)
    observability.register_artifact(
        run_id=run_id,
        path=metrics_path,
        artifact_type="baseline_metrics",
        label=args.baseline,
        metadata=metrics,
        source="runtime",
    )

    assert sample_timestamps is not None
    plot_denoising_comparison(
        noisy=noisy_array[0],
        denoised=pred_array[0],
        clean=clean_array[0],
        timestamps=sample_timestamps,
        title=f"{args.baseline} baseline",
        save_path=run_paths.figures_dir / "comparison.png",
    )
    observability.register_artifact(
        run_id=run_id,
        path=run_paths.figures_dir / "comparison.png",
        artifact_type="figure",
        label=f"{args.baseline}_comparison",
        source="runtime",
    )
    plot_psd(
        signals={"noisy": noisy_array[0], "denoised": pred_array[0], "clean": clean_array[0]},
        fs=100.0 if config.data.dataset == "blackbird" else 200.0,
        title=f"{args.baseline} baseline PSD",
        save_path=run_paths.figures_dir / "psd.png",
    )
    observability.register_artifact(
        run_id=run_id,
        path=run_paths.figures_dir / "psd.png",
        artifact_type="figure",
        label=f"{args.baseline}_psd",
        source="runtime",
    )
    observability.finish_run(
        run_id=run_id,
        status="completed",
        summary={"rmse": metrics["rmse"], "mae": metrics["mae"]},
        source="runtime",
    )
    selection_event = observability.record_selection_event(
        run_id=run_id,
        loop_run_id=None,
        iteration=None,
        proposal_source="manual",
        description=f"{args.baseline} baseline",
        incumbent_run_id=None,
        candidate_count=1,
        rationale="launched manually from the CLI",
        policy_state={"mode": "manual", "baseline": args.baseline},
        source="runtime",
    )
    change_set = observability.record_change_set(
        run_id=run_id,
        loop_run_id=None,
        parent_run_id=None,
        incumbent_run_id=None,
        reference_kind="manual",
        proposal_source="manual",
        description=f"{args.baseline} baseline",
        overrides=list(args.overrides),
        current_config=config,
        reference_config=None,
        source="runtime",
    )
    observability.record_decision(
        run_id=run_id,
        iteration=None,
        proposal_source="manual",
        description=f"{args.baseline} baseline",
        status="completed",
        metric_key="rmse",
        metric_value=float(metrics["rmse"]),
        overrides=list(args.overrides),
        source="runtime",
    )
    update_run_manifest(
        run_paths,
        {
            "regime_fingerprint": data_regime_fingerprint(config),
            "resolved_config": observability.config_payload(config),
            "selection_event": selection_event,
            "change_set": change_set,
        },
    )

    print("Baseline evaluation complete:")
    print(f"  baseline: {args.baseline}")
    print(f"  metrics_path: {metrics_path}")
    print(f"  rmse: {metrics['rmse']:.6f}")
    print(f"  mae: {metrics['mae']:.6f}")
    return 0


def main() -> int:
    """CLI entrypoint."""
    return run_command(build_parser().parse_args())
