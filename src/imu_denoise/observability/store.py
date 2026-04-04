"""IMU-domain wrapper over the reusable Mission Control SQLite store."""

from __future__ import annotations

from typing import Any

from autoresearch_core.observability.store import (
    ACTIVE_LOOP_HEARTBEAT_TIMEOUT_SEC,
    SCHEMA_SQL,
    BlobStore,
)
from autoresearch_core.observability.store import (
    ObservabilityStore as CoreObservabilityStore,
)
from imu_denoise.observability.lineage import data_regime_fingerprint


class ObservabilityStore(CoreObservabilityStore):
    """IMU-specific store wrapper that supplies regime-fingerprint semantics."""

    def _compute_regime_fingerprint(self, payload: dict[str, Any]) -> str | None:
        return data_regime_fingerprint(payload)


__all__ = [
    "ACTIVE_LOOP_HEARTBEAT_TIMEOUT_SEC",
    "BlobStore",
    "ObservabilityStore",
    "SCHEMA_SQL",
]
