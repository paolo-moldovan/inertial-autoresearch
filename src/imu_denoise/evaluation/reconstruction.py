"""Sequence reconstruction helpers for overlapping denoising windows."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def build_window_weights(window_size: int, mode: str) -> np.ndarray:
    """Return per-sample overlap-add weights for a window."""
    if mode == "hann":
        # Avoid exact zeros at the boundaries so full-sequence endpoints remain defined.
        weights = np.hanning(window_size + 2)[1:-1]
        return np.maximum(weights.astype(np.float64), 1e-6)
    if mode == "none":
        raise ValueError("Reconstruction mode 'none' does not define overlap-add weights.")
    raise ValueError(f"Unsupported reconstruction mode: {mode}")


def reconstruct_window_predictions(
    *,
    pred_windows: np.ndarray,
    target_windows: np.ndarray,
    timestamps: np.ndarray,
    sequence_ids: list[str],
    mode: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Reconstruct full sequences from overlapping windows grouped by sequence id."""
    if mode == "none":
        raise ValueError("Sequence reconstruction requested with reconstruction='none'.")
    if len(pred_windows) == 0:
        return {}

    grouped: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    for index, sequence_id in enumerate(sequence_ids):
        grouped[sequence_id].append(
            {
                "pred": pred_windows[index],
                "target": target_windows[index],
                "timestamps": timestamps[index],
            }
        )

    reconstructed: dict[str, dict[str, np.ndarray]] = {}
    window_weights = build_window_weights(pred_windows.shape[1], mode)

    for sequence_id, windows in grouped.items():
        all_timestamps = np.unique(
            np.concatenate([window["timestamps"] for window in windows]).astype(np.float64)
        )
        all_timestamps.sort()
        pred_sum = np.zeros((len(all_timestamps), pred_windows.shape[2]), dtype=np.float64)
        target_sum = np.zeros_like(pred_sum)
        weight_sum = np.zeros((len(all_timestamps), 1), dtype=np.float64)

        for window in windows:
            indices = np.searchsorted(all_timestamps, window["timestamps"])
            weights = window_weights.reshape(-1, 1)
            pred_sum[indices] += window["pred"] * weights
            target_sum[indices] += window["target"] * weights
            weight_sum[indices] += weights

        safe_weights = np.maximum(weight_sum, 1e-8)
        reconstructed[sequence_id] = {
            "timestamps": all_timestamps.astype(np.float32),
            "pred": (pred_sum / safe_weights).astype(np.float32),
            "target": (target_sum / safe_weights).astype(np.float32),
        }

    return reconstructed
