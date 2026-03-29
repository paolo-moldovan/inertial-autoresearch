"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def synthetic_batch() -> torch.Tensor:
    """Small synthetic IMU batch with the canonical `(B, T, 6)` shape."""
    torch.manual_seed(0)
    return torch.randn(2, 16, 6)
