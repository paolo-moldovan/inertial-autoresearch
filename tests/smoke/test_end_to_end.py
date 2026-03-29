"""Smoke tests: full pipeline on synthetic data — no real datasets required."""

from __future__ import annotations

import pytest

from imu_denoise.config import load_config
from imu_denoise.data.synthetic.generator import generate_synthetic_imu
from imu_denoise.data.transforms import sliding_window
from imu_denoise.device import DeviceContext
from imu_denoise.evaluation.metrics import compute_all_metrics
from imu_denoise.models import get_model


def _make_batch(
    n_windows: int = 4, window_size: int = 64, channels: int = 6
) -> tuple:
    """Generate a small noisy/clean window batch."""
    import numpy as np
    import torch

    rng = np.random.default_rng(0)
    clean = rng.standard_normal((n_windows, window_size, channels)).astype("float32")
    noisy = clean + 0.05 * rng.standard_normal(clean.shape).astype("float32")
    return torch.from_numpy(noisy), torch.from_numpy(clean)


@pytest.fixture(scope="module")
def device_ctx() -> DeviceContext:
    from imu_denoise.config.schema import DeviceConfig

    return DeviceContext.from_config(DeviceConfig(preferred="cpu", dtype="float32"))


@pytest.mark.parametrize("model_name", ["lstm", "conv1d", "transformer"])
def test_model_forward_shape(model_name: str, device_ctx: DeviceContext) -> None:
    """Each registered model must produce (batch, seq, 6) output."""
    noisy, _ = _make_batch()
    model = get_model(model_name)
    model = device_ctx.to_device(model)
    model.eval()

    import torch

    with torch.no_grad():
        out = model(noisy.to(device_ctx.device))

    assert out.shape == noisy.shape, (
        f"{model_name}: expected {noisy.shape}, got {out.shape}"
    )


def test_synthetic_data_generation() -> None:
    """Synthetic generator returns a valid IMUSequence."""
    seq = generate_synthetic_imu(duration_sec=2.0, rate_hz=50.0)
    assert seq.num_samples == 100
    assert seq.ground_truth_accel is not None
    assert seq.ground_truth_gyro is not None
    assert seq.accel.shape == (100, 3)
    assert seq.gyro.shape == (100, 3)


def test_sliding_window() -> None:
    """Sliding window produces the correct number of windows from an IMUSequence."""
    seq = generate_synthetic_imu(duration_sec=10.0, rate_hz=50.0)  # 500 samples
    noisy_w, clean_w, _ = sliding_window(seq, window_size=100, stride=50)
    # (500 - 100) // 50 + 1 = 9
    assert noisy_w.shape == (9, 100, 6)
    assert clean_w.shape == (9, 100, 6)


def test_metrics_on_perfect_prediction() -> None:
    """RMSE and MAE should be ~0 for a perfect prediction."""
    import numpy as np

    gt = np.random.default_rng(1).standard_normal((200, 6)).astype("float32")
    metrics = compute_all_metrics(gt, gt, fs=50.0)
    assert metrics["rmse"] < 1e-6
    assert metrics["mae"] < 1e-6


def test_config_load_and_merge() -> None:
    """Config loader merges YAML files and respects CLI overrides."""
    cfg = load_config(
        "configs/base.yaml",
        "configs/training/quick.yaml",
        overrides=["training.lr=0.0005", "model.name=conv1d"],
    )
    assert cfg.training.lr == 0.0005
    assert cfg.model.name == "conv1d"
    assert cfg.data.dataset == "synthetic"


def test_full_quick_train_run() -> None:
    """2-epoch training run on synthetic data must complete without error."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/training/quick.yaml",
            "--set",
            "training.seed=0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"train.py exited with {result.returncode}:\n{result.stderr}"
    )
    assert "Training complete" in result.stdout
