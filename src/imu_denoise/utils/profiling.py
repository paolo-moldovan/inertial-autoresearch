"""Performance monitoring utilities."""

from __future__ import annotations

import functools
import logging
import platform
import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar

import torch

F = TypeVar("F", bound=Callable[..., Any])

_logger = logging.getLogger(__name__)


def timed(fn: F) -> F:
    """Decorator that logs the execution time of a function.

    Uses the module-level logger at DEBUG level.

    Args:
        fn: The function to wrap.

    Returns:
        Wrapped function that logs its execution time.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _logger.debug("%s completed in %.4f s", fn.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]


def get_gpu_memory_mb() -> float | None:
    """Return current GPU memory usage in MB, or ``None`` if CUDA is unavailable.

    Returns:
        Allocated GPU memory in megabytes, or ``None`` on non-CUDA devices.
    """
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024 * 1024)


def log_system_info(logger: logging.Logger) -> None:
    """Log system and hardware information useful for reproducibility.

    Args:
        logger: Logger instance to write to.
    """
    logger.info("Python %s", sys.version.split()[0])
    logger.info("Platform: %s", platform.platform())
    logger.info("PyTorch %s", torch.__version__)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        total_memory = getattr(properties, "total_memory", 0)
        vram_mb = total_memory / (1024 * 1024)
        logger.info("CUDA device: %s (%.0f MB VRAM)", device_name, vram_mb)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Device: Apple MPS")
    else:
        logger.info("Device: CPU only")

    logger.info("Default dtype: %s", torch.get_default_dtype())
