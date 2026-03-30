"""Tests for evaluation cadence, reconstruction, and temporal metrics."""

from __future__ import annotations

import numpy as np
import pytest

from imu_denoise.cli.common import build_model
from imu_denoise.config import (
    DataConfig,
    DeviceConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from imu_denoise.data.datamodule import create_dataloaders
from imu_denoise.device import DeviceContext
from imu_denoise.evaluation.evaluator import Evaluator, evaluate_window_predictions
from imu_denoise.evaluation.reconstruction import reconstruct_window_predictions


def test_sequence_reconstruction_preserves_length_and_overlap() -> None:
    """Overlap-add reconstruction should recover the full sequence support."""
    pred_windows = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[2.0], [3.0], [4.0], [5.0]],
        ],
        dtype=np.float32,
    )
    reconstructed = reconstruct_window_predictions(
        pred_windows=pred_windows,
        target_windows=pred_windows.copy(),
        timestamps=np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        ),
        sequence_ids=["seq-a", "seq-a"],
        mode="hann",
    )

    assert list(reconstructed.keys()) == ["seq-a"]
    seq = reconstructed["seq-a"]
    assert seq["timestamps"].shape == (6,)
    assert seq["pred"].shape == (6, 1)
    assert np.allclose(seq["timestamps"], np.arange(6, dtype=np.float32))
    assert np.allclose(seq["pred"].squeeze(-1), np.arange(6, dtype=np.float32), atol=1e-5)


def test_window_and_sequence_metrics_can_coexist() -> None:
    """Evaluation should support both window-level and reconstructed sequence metrics."""
    pred_windows = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[2.0], [3.0], [4.0], [5.0]],
        ],
        dtype=np.float32,
    )
    metrics = evaluate_window_predictions(
        pred_windows=pred_windows,
        target_windows=pred_windows.copy(),
        timestamps=np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        ),
        sequence_ids=["seq-a", "seq-a"],
        fs=100.0,
        evaluation=EvaluationConfig(
            metrics=["rmse", "sequence_rmse", "smoothness", "drift_error"],
            reconstruction="hann",
        ),
    )

    assert metrics["rmse"] == 0.0
    assert metrics["sequence_rmse"] == 0.0
    assert metrics["smoothness"] == 0.0
    assert metrics["drift_error"] == 0.0


def test_realtime_mode_warns_for_non_causal_models() -> None:
    """Realtime evaluation should warn when the model can see future context."""
    config = ExperimentConfig(
        data=DataConfig(
            dataset="synthetic",
            window_size=12,
            stride=6,
            normalize=False,
            augment=False,
            dataset_kwargs={
                "duration_sec": 1.0,
                "rate_hz": 20.0,
                "num_sequences": 3,
                "seed": 5,
            },
        ),
        model=ExperimentConfig().model,
        training=TrainingConfig(batch_size=2, num_workers=0, seed=7),
        evaluation=EvaluationConfig(metrics=["rmse"], realtime_mode=True),
        device=DeviceConfig(preferred="cpu"),
    )
    device_ctx = DeviceContext.from_config(config.device)
    bundle = create_dataloaders(config.data, config.training, device_ctx)
    model = build_model(config)
    evaluator = Evaluator(model, device_ctx).with_config(config.evaluation)

    with pytest.warns(RuntimeWarning, match="non-causal"):
        metrics = evaluator.evaluate(bundle.val_loader, fs=bundle.sampling_rate_hz)

    assert "rmse" in metrics
