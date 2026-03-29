"""Checkpoint and metrics I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        path: Destination file path (e.g. ``checkpoints/best.pt``).
        model: The model whose ``state_dict`` will be saved.
        optimizer: The optimizer whose ``state_dict`` will be saved.
        epoch: Current epoch number.
        best_metric: Best validation metric achieved so far.
        extra: Optional dictionary of additional data to store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint and restore model/optimizer state.

    Args:
        path: Path to the saved checkpoint.
        model: Model to load weights into.
        optimizer: If provided, optimizer state is also restored.
        device: Device to map tensors to (defaults to CPU).

    Returns:
        Dictionary with ``epoch``, ``best_metric``, and optional ``extra``.
    """
    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "best_metric": checkpoint["best_metric"],
        "extra": checkpoint.get("extra", {}),
    }


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Save a metrics dictionary as a JSON file.

    Args:
        path: Destination JSON file path.
        metrics: Dictionary of metric names to values.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_metrics(path: Path) -> dict[str, Any]:
    """Load a metrics dictionary from a JSON file.

    Args:
        path: Path to the JSON metrics file.

    Returns:
        Dictionary of metric names to values.
    """
    with open(path, encoding="utf-8") as f:
        return dict(json.load(f))
