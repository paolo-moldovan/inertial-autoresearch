"""Plotting utilities for IMU denoising evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Axis labels for 6-channel IMU data (accel x/y/z, gyro x/y/z)
_IMU_AXIS_LABELS = ["Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]


def plot_denoising_comparison(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray,
    timestamps: np.ndarray,
    title: str = "Denoising Comparison",
    save_path: Path | str | None = None,
) -> Figure:
    """Plot a 6-subplot figure comparing noisy, denoised, and clean signals per IMU axis.

    Args:
        noisy: Noisy input of shape ``(N, 6)``.
        denoised: Model output of shape ``(N, 6)``.
        clean: Ground truth of shape ``(N, 6)``.
        timestamps: Time axis of shape ``(N,)``.
        title: Figure super-title.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    n_axes = noisy.shape[1]
    fig: Figure
    raw_axes: Any
    fig, raw_axes = plt.subplots(n_axes, 1, figsize=(14, 2.5 * n_axes), sharex=True)
    axes_list: list[Axes] = [raw_axes] if n_axes == 1 else list(raw_axes)

    for i, ax in enumerate(axes_list):
        label = _IMU_AXIS_LABELS[i] if i < len(_IMU_AXIS_LABELS) else f"Axis {i}"
        ax.plot(timestamps, noisy[:, i], alpha=0.4, linewidth=0.7, label="Noisy", color="gray")
        ax.plot(timestamps, clean[:, i], linewidth=1.0, label="Clean", color="tab:blue")
        ax.plot(
            timestamps,
            denoised[:, i],
            linewidth=1.0,
            linestyle="--",
            label="Denoised",
            color="tab:orange",
        )
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes_list[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_psd(
    signals: dict[str, np.ndarray],
    fs: float,
    title: str = "Power Spectral Density",
    save_path: Path | str | None = None,
) -> Figure:
    """Plot power spectral density comparison for multiple signals.

    Args:
        signals: Mapping of label to 1-D signal array (or first column is used
            for multi-channel arrays).
        fs: Sampling frequency in Hz.
        title: Figure title.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    fig: Figure
    ax: Any
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, sig in signals.items():
        if sig.ndim > 1:
            sig = sig[:, 0]
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = np.abs(np.fft.rfft(sig)) ** 2 / n
        ax.semilogy(freqs, psd, label=label, alpha=0.8, linewidth=0.9)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_error_distribution(
    errors: np.ndarray,
    title: str = "Error Distribution",
    save_path: Path | str | None = None,
) -> Figure:
    """Plot histograms of error distributions per axis.

    Args:
        errors: Error array of shape ``(N, C)`` (e.g. pred - target).
        title: Figure super-title.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    n_axes = errors.shape[1]
    cols = min(n_axes, 3)
    rows = (n_axes + cols - 1) // cols
    fig: Figure
    raw_all: Any
    fig, raw_all = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat: list[Axes] = [cast(Axes, ax) for ax in np.asarray(raw_all, dtype=object).ravel()]

    for i in range(n_axes):
        ax = axes_flat[i]
        label = _IMU_AXIS_LABELS[i] if i < len(_IMU_AXIS_LABELS) else f"Axis {i}"
        ax.hist(errors[:, i], bins=80, alpha=0.75, edgecolor="black", linewidth=0.3)
        ax.set_title(label)
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_axes, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    log_path: Path | str,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot training and validation loss curves from a JSON-lines log file.

    Expects each line to be a JSON object with at least ``"epoch"`` and
    ``"train_loss"`` keys. ``"val_loss"`` is optional but plotted when present.

    Args:
        log_path: Path to the JSON-lines training log.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    log_path = Path(log_path)
    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record: dict[str, Any] = json.loads(line)
            if "epoch" not in record or "train_loss" not in record:
                continue
            epochs.append(record["epoch"])
            train_losses.append(record["train_loss"])
            if "val_loss" in record:
                val_losses.append(record["val_loss"])

    fig: Figure
    ax2: Any
    fig, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(epochs, train_losses, label="Train Loss", linewidth=1.2)
    if val_losses and len(val_losses) == len(epochs):
        ax2.plot(epochs, val_losses, label="Val Loss", linewidth=1.2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
