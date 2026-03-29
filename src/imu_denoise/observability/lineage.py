"""Helpers for structured experiment lineage and config diffs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

_IGNORED_ROOTS = {"observability"}
_IGNORED_PATHS = {
    "name",
    "output_dir",
    "log_dir",
}


def normalize_config_payload(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalize a config object or mapping into a plain dictionary payload."""
    if is_dataclass(config) and not isinstance(config, type):
        payload = asdict(config)
        return payload if isinstance(payload, dict) else {}
    if isinstance(config, Mapping):
        return dict(config)
    return {}


def build_change_items(
    *,
    current_config: Mapping[str, Any] | Any,
    reference_config: Mapping[str, Any] | Any | None,
    overrides: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a stable list of config changes for lineage and traceability."""
    current_payload = normalize_config_payload(current_config)
    reference_payload = (
        normalize_config_payload(reference_config) if reference_config is not None else None
    )
    if reference_payload is None:
        return _change_items_from_overrides(current_payload, overrides or [])

    items: list[dict[str, Any]] = []
    _diff_mapping(current_payload, reference_payload, path="", items=items)
    items.sort(key=lambda item: str(item["path"]))
    return items


def summarize_change_items(change_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact summary from a list of structured change items."""
    return {
        "change_count": len(change_items),
        "paths": [str(item["path"]) for item in change_items],
        "categories": sorted({str(item["category"]) for item in change_items}),
    }


def _diff_mapping(
    current: Any,
    reference: Any,
    *,
    path: str,
    items: list[dict[str, Any]],
) -> None:
    if isinstance(current, Mapping) and isinstance(reference, Mapping):
        keys = sorted(set(current.keys()) | set(reference.keys()))
        for key in keys:
            next_path = f"{path}.{key}" if path else str(key)
            if _should_ignore_path(next_path):
                continue
            _diff_mapping(current.get(key), reference.get(key), path=next_path, items=items)
        return
    if current == reference:
        return
    if _should_ignore_path(path):
        return
    items.append(
        {
            "path": path,
            "category": path.split(".", 1)[0] if path else "config",
            "before": reference,
            "after": current,
        }
    )


def _change_items_from_overrides(
    current_payload: Mapping[str, Any],
    overrides: list[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for override in overrides:
        if "=" not in override:
            continue
        key, _value = override.split("=", 1)
        if _should_ignore_path(key) or key in seen_paths:
            continue
        seen_paths.add(key)
        items.append(
            {
                "path": key,
                "category": key.split(".", 1)[0] if key else "config",
                "before": None,
                "after": _lookup_path(current_payload, key),
            }
        )
    items.sort(key=lambda item: str(item["path"]))
    return items


def _lookup_path(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _should_ignore_path(path: str) -> bool:
    if not path:
        return False
    root = path.split(".", 1)[0]
    if root in _IGNORED_ROOTS:
        return True
    return path in _IGNORED_PATHS
