"""Tests for the shared IMU data pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from imu_denoise.config import DataConfig, DataSubsetConfig, DeviceConfig, TrainingConfig
from imu_denoise.data.datamodule import create_dataloaders
from imu_denoise.data.io import load_processed_sequence, save_processed_sequence
from imu_denoise.data.registry import get_dataset
from imu_denoise.data.splits import resolve_splits, split_by_sequence
from imu_denoise.data.synthetic.generator import generate_synthetic_imu
from imu_denoise.data.transforms import sliding_window
from imu_denoise.device import DeviceContext


def test_synthetic_generator_produces_ground_truth() -> None:
    """Synthetic generation should produce noisy data and clean targets together."""
    sequence = generate_synthetic_imu(duration_sec=2.0, rate_hz=10.0, seed=7, name="synthetic_000")

    assert sequence.name == "synthetic_000"
    assert sequence.has_ground_truth is True
    assert sequence.accel.shape == (20, 3)
    assert sequence.gyro.shape == (20, 3)
    assert sequence.clean_imu.shape == (20, 6)


def test_processed_sequence_roundtrip(tmp_path: Path) -> None:
    """Processed-sequence save/load should preserve arrays and metadata."""
    sequence = generate_synthetic_imu(duration_sec=1.0, rate_hz=8.0, seed=3, name="roundtrip")
    destination = tmp_path / "processed" / "synthetic" / "roundtrip.npz"

    save_processed_sequence(sequence, destination)
    loaded = load_processed_sequence(destination)

    assert loaded.name == "roundtrip"
    assert np.allclose(loaded.timestamps, sequence.timestamps)
    assert np.allclose(loaded.accel, sequence.accel)
    assert np.allclose(loaded.gyro, sequence.gyro)
    assert loaded.ground_truth_accel is not None
    assert sequence.ground_truth_accel is not None
    assert np.allclose(loaded.ground_truth_accel, sequence.ground_truth_accel)


def test_sliding_window_returns_noisy_clean_and_timestamps() -> None:
    """Windowing should preserve alignment between IMU data and timestamps."""
    sequence = generate_synthetic_imu(duration_sec=2.0, rate_hz=5.0, seed=1, name="windowed")

    noisy, clean, timestamps = sliding_window(sequence, window_size=4, stride=2)

    assert noisy.shape == (4, 4, 6)
    assert clean.shape == (4, 4, 6)
    assert timestamps.shape == (4, 4)
    assert np.allclose(timestamps[0], sequence.timestamps[:4])


def test_split_by_sequence_supports_prefix_matching() -> None:
    """Trajectory-only split names should match `trajectory_speed` sequence ids."""
    sequences = [
        generate_synthetic_imu(duration_sec=1.0, rate_hz=4.0, seed=0, name="sphinx_1.0"),
        generate_synthetic_imu(duration_sec=1.0, rate_hz=4.0, seed=1, name="sphinx_2.0"),
        generate_synthetic_imu(duration_sec=1.0, rate_hz=4.0, seed=2, name="winter_1.0"),
    ]

    train, val, test = split_by_sequence(
        sequences,
        train_names=["sphinx"],
        val_names=["winter"],
        test_names=[],
    )

    assert [sequence.name for sequence in train] == ["sphinx_1.0", "sphinx_2.0"]
    assert [sequence.name for sequence in val] == ["winter_1.0"]
    assert test == []


def test_resolve_splits_falls_back_to_ratios() -> None:
    """Synthetic configs without explicit split names should still yield non-empty splits."""
    sequences = get_dataset("synthetic", duration_sec=1.0, rate_hz=4.0, num_sequences=5, seed=10)

    train, val, test = resolve_splits(sequences, train_names=[], val_names=[], test_names=[])

    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1


def test_registry_prefers_processed_sequences(tmp_path: Path) -> None:
    """Real-dataset registry loading should consume cached processed `.npz` files when present."""
    sequence = generate_synthetic_imu(duration_sec=1.0, rate_hz=8.0, seed=2, name="MH_01_easy")
    destination = tmp_path / "processed" / "euroc" / "MH_01_easy.npz"
    save_processed_sequence(sequence, destination)

    loaded = get_dataset("euroc", data_dir=tmp_path, sequences=["MH_01_easy"])

    assert len(loaded) == 1
    assert loaded[0].name == "MH_01_easy"
    assert loaded[0].has_ground_truth is True


def test_create_dataloaders_builds_metadata_rich_batches() -> None:
    """Datamodule should emit non-empty batches with tensors plus sequence metadata."""
    data_config = DataConfig(
        dataset="synthetic",
        window_size=8,
        stride=4,
        normalize=True,
        augment=False,
        data_dir="data",
    )
    training_config = TrainingConfig(batch_size=2, num_workers=0, seed=42)
    device_ctx = DeviceContext.from_config(DeviceConfig(preferred="cpu"))

    train_loader, val_loader, test_loader = create_dataloaders(
        data_config,
        training_config,
        device_ctx,
    )

    train_batch = next(iter(train_loader))

    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))
    assert isinstance(train_batch["noisy"], torch.Tensor)
    assert train_batch["noisy"].shape[-1] == 6
    assert train_batch["clean"].shape == train_batch["noisy"].shape
    assert train_batch["timestamps"].shape[-1] == 8
    assert isinstance(train_batch["sequence_id"], list)
    assert isinstance(train_batch["sequence_id"][0], str)
    assert train_batch["dt"].dtype == torch.float32
    assert isinstance(val_batch["noisy"], torch.Tensor)
    assert isinstance(test_batch["noisy"], torch.Tensor)


def test_subset_controls_are_reproducible_and_split_safe() -> None:
    """Subsetting should be deterministic and preserve train/val/test isolation."""
    data_config = DataConfig(
        dataset="synthetic",
        window_size=8,
        stride=4,
        normalize=True,
        augment=False,
        dataset_kwargs={"duration_sec": 2.0, "rate_hz": 8.0, "num_sequences": 6, "seed": 10},
        subset=DataSubsetConfig(
            enabled=True,
            seed=123,
            train_max_sequences=2,
            val_max_sequences=1,
            test_max_sequences=1,
            train_max_windows=3,
            val_max_windows=2,
            test_max_windows=1,
        ),
    )
    training_config = TrainingConfig(batch_size=8, num_workers=0, seed=42)
    device_ctx = DeviceContext.from_config(DeviceConfig(preferred="cpu"))

    first_train, first_val, first_test = create_dataloaders(
        data_config,
        training_config,
        device_ctx,
    )
    second_train, second_val, second_test = create_dataloaders(
        data_config,
        training_config,
        device_ctx,
    )

    first_train_ids = set(first_train.dataset._sequence_ids)  # type: ignore[attr-defined]
    first_val_ids = set(first_val.dataset._sequence_ids)  # type: ignore[attr-defined]
    first_test_ids = set(first_test.dataset._sequence_ids)  # type: ignore[attr-defined]
    second_train_ids = set(second_train.dataset._sequence_ids)  # type: ignore[attr-defined]
    second_val_ids = set(second_val.dataset._sequence_ids)  # type: ignore[attr-defined]
    second_test_ids = set(second_test.dataset._sequence_ids)  # type: ignore[attr-defined]

    assert first_train_ids == second_train_ids
    assert first_val_ids == second_val_ids
    assert first_test_ids == second_test_ids
    assert len(first_train.dataset) == 3  # type: ignore[arg-type]
    assert len(first_val.dataset) == 2  # type: ignore[arg-type]
    assert len(first_test.dataset) == 1  # type: ignore[arg-type]
    assert first_train_ids.isdisjoint(first_val_ids)
    assert first_train_ids.isdisjoint(first_test_ids)
    assert first_val_ids.isdisjoint(first_test_ids)


def test_subset_controls_do_not_change_sampling_interval() -> None:
    """Window subsetting should not alter per-window timestamp spacing."""
    base_config = DataConfig(
        dataset="synthetic",
        window_size=8,
        stride=4,
        normalize=False,
        augment=False,
        dataset_kwargs={"duration_sec": 2.0, "rate_hz": 16.0, "num_sequences": 5, "seed": 5},
    )
    subset_config = DataConfig(
        dataset=base_config.dataset,
        window_size=base_config.window_size,
        stride=base_config.stride,
        normalize=base_config.normalize,
        augment=base_config.augment,
        dataset_kwargs=dict(base_config.dataset_kwargs),
        subset=DataSubsetConfig(enabled=True, seed=99, train_max_windows=2, val_max_windows=1),
    )
    training_config = TrainingConfig(batch_size=4, num_workers=0, seed=42)
    device_ctx = DeviceContext.from_config(DeviceConfig(preferred="cpu"))

    base_train, _, _ = create_dataloaders(base_config, training_config, device_ctx)
    subset_train, _, _ = create_dataloaders(subset_config, training_config, device_ctx)

    base_dt = float(base_train.dataset[0]["dt"])
    subset_dt = float(subset_train.dataset[0]["dt"])
    assert base_dt == subset_dt
