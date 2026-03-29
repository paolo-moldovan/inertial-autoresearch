"""Model registry for discovering and instantiating denoiser models."""

from __future__ import annotations

from collections.abc import Callable

from imu_denoise.models.base import BaseDenoiser

_MODEL_REGISTRY: dict[str, type[BaseDenoiser]] = {}


def register_model(name: str) -> Callable[[type[BaseDenoiser]], type[BaseDenoiser]]:
    """Decorator that registers a model class under the given name.

    Args:
        name: Unique string identifier for the model.

    Returns:
        Decorator that registers and returns the model class unchanged.

    Raises:
        ValueError: If a model with the given name is already registered.
    """

    def decorator(cls: type[BaseDenoiser]) -> type[BaseDenoiser]:
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered by {_MODEL_REGISTRY[name].__name__}"
            )
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs: object) -> BaseDenoiser:
    """Instantiate a registered model by name.

    Args:
        name: Registered model name.
        **kwargs: Keyword arguments forwarded to the model constructor.

    Returns:
        An instance of the requested model.

    Raises:
        KeyError: If no model is registered under the given name.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(_MODEL_REGISTRY.keys())
