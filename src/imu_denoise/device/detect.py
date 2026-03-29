"""Auto-detect available compute device."""

from __future__ import annotations

import torch


def detect_device(preferred: str = "auto") -> torch.device:
    """Detect the best available device.

    Priority: CUDA > MPS > CPU, unless overridden by `preferred`.

    Args:
        preferred: One of "auto", "cuda", "mps", "cpu".

    Returns:
        A torch.device for the selected backend.
    """
    if preferred != "auto":
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
