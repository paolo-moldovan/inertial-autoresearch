"""Device context manager for consistent device/dtype/AMP handling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from imu_denoise.config.schema import DeviceConfig
from imu_denoise.device.detect import detect_device

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class DeviceContext:
    """Manages device placement, dtype selection, and AMP contexts.

    Handles MPS-specific limitations:
    - No bfloat16 support (falls back to float32)
    - No torch.compile support
    - pin_memory disabled for MPS
    """

    device: torch.device
    dtype: torch.dtype
    amp_enabled: bool

    @classmethod
    def from_config(cls, config: DeviceConfig) -> DeviceContext:
        """Create a DeviceContext from a DeviceConfig."""
        device = detect_device(config.preferred)
        dtype = _DTYPE_MAP.get(config.dtype, torch.float32)

        # MPS limitations
        if device.type == "mps":
            if dtype == torch.bfloat16:
                dtype = torch.float32
            amp_enabled = False
        elif device.type == "cuda":
            amp_enabled = dtype in (torch.float16, torch.bfloat16)
        else:
            amp_enabled = False

        return cls(device=device, dtype=dtype, amp_enabled=amp_enabled)

    @property
    def pin_memory(self) -> bool:
        """Whether DataLoader should pin memory."""
        return self.device.type == "cuda"

    @property
    def supports_compile(self) -> bool:
        """Whether torch.compile is supported on this device."""
        return self.device.type == "cuda"

    @contextmanager
    def autocast(self) -> Iterator[None]:
        """Context manager for automatic mixed precision."""
        if self.amp_enabled:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                yield
        else:
            with nullcontext():
                yield

    def to_device(self, x: Tensor | nn.Module) -> Any:
        """Move a tensor or module to this device."""
        return x.to(self.device)

    def create_scaler(self) -> Any | None:
        """Create a GradScaler if AMP is enabled with float16 on CUDA."""
        if self.amp_enabled and self.device.type == "cuda" and self.dtype == torch.float16:
            return cast(Any, torch.amp.GradScaler("cuda"))  # type: ignore[attr-defined]
        return None
