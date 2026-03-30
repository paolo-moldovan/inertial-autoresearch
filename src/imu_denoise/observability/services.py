"""Composition helpers for Mission Control UI and operator surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autoresearch_core.observability import MissionControlFacade
from imu_denoise.config.schema import ObservabilityConfig
from imu_denoise.observability.control import LoopController
from imu_denoise.observability.queries import MissionControlQueries
from imu_denoise.observability.store import ObservabilityStore
from imu_denoise.observability.writer import ObservabilityWriter


@dataclass(frozen=True)
class MissionControlServices:
    """Resolved query/control stack used by the UI and CLI composition layers."""

    facade: MissionControlFacade
    queries: MissionControlQueries
    controller: LoopController
    writer: ObservabilityWriter


def build_mission_control_services(
    *,
    db_path: Path,
    blob_dir: Path,
) -> MissionControlServices:
    queries = MissionControlQueries(db_path=db_path, blob_dir=blob_dir)
    store = ObservabilityStore(db_path=db_path, blob_dir=blob_dir)
    writer = ObservabilityWriter(
        config=ObservabilityConfig(
            enabled=True,
            db_path=str(db_path),
            blob_dir=str(blob_dir),
        ),
        store=store,
    )
    controller = LoopController(store=store, writer=writer)
    facade = MissionControlFacade(queries=queries, controller=controller, writer=writer)
    return MissionControlServices(
        facade=facade,
        queries=queries,
        controller=controller,
        writer=writer,
    )
