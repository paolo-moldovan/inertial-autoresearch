"""Generic Mission Control service bundle composition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoresearch_core.observability.facade import MissionControlFacade


@dataclass(frozen=True)
class MissionControlServices:
    """Resolved query/control stack used by UI and CLI composition layers."""

    facade: MissionControlFacade
    queries: Any
    controller: Any
    writer: Any


def compose_mission_control_services(
    *,
    queries: Any,
    controller: Any,
    writer: Any,
) -> MissionControlServices:
    """Create a Mission Control service bundle from injected store/query/control services."""
    facade = MissionControlFacade(queries=queries, controller=controller, writer=writer)
    return MissionControlServices(
        facade=facade,
        queries=queries,
        controller=controller,
        writer=writer,
    )
