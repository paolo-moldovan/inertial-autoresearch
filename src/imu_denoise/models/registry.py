"""Model registry for discovering and instantiating denoiser models."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

from imu_denoise.models.base import BaseDenoiser

_MODEL_REGISTRY: dict[str, type[BaseDenoiser]] = {}
_DISCOVERED_MODULES: set[str] = set()


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
    autodiscover_models()
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    autodiscover_models()
    return sorted(_MODEL_REGISTRY.keys())


def autodiscover_models() -> list[str]:
    """Import model modules from the package directory on demand."""
    import imu_denoise.models as models_package

    discovered: list[str] = []
    for module_info in pkgutil.iter_modules(models_package.__path__):
        module_name = module_info.name
        if module_info.ispkg:
            continue
        if module_name.startswith("_") or module_name in {"base", "registry"}:
            continue
        full_name = f"{models_package.__name__}.{module_name}"
        if full_name in _DISCOVERED_MODULES:
            continue
        importlib.import_module(full_name)
        _DISCOVERED_MODULES.add(full_name)
        discovered.append(module_name)
    return discovered
