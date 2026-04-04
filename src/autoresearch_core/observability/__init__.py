"""Reusable observability façades and analytics helpers."""

from autoresearch_core.observability.analytics import (
    compute_loop_analytics,
    compute_multi_loop_analytics,
)
from autoresearch_core.observability.backfill import (
    build_backfill_run_id,
    ensure_backfill_run,
    load_nearest_run_manifest,
    parse_iso_timestamp,
    read_json_lines,
    resolve_manifest_run_reference,
)
from autoresearch_core.observability.control import (
    LOOP_PAUSED,
    LOOP_RESUMED,
    LOOP_STOP_REQUESTED,
    LOOP_STOPPED,
    LOOP_TERMINATE_REQUESTED,
    LOOP_TERMINATED,
    QUEUE_APPLIED,
    QUEUE_CLAIMED,
    QUEUE_ENQUEUED,
    LoopAlreadyRunningError,
    LoopController,
)
from autoresearch_core.observability.facade import MissionControlFacade
from autoresearch_core.observability.hermes_import import (
    HERMES_SESSION_IMPORTED,
    HERMES_TRANSCRIPT_IMPORTED,
    import_hermes_state,
)
from autoresearch_core.observability.logging import MissionControlLogHandler
from autoresearch_core.observability.queries import CoreMissionControlQueries
from autoresearch_core.observability.read_models import (
    build_current_candidate_pool,
    build_current_run_summary,
    build_hermes_runtime_summary,
    build_mission_control_summary_payload,
    build_run_policy_context,
)
from autoresearch_core.observability.redaction import REDACTED, redact_payload, redact_text
from autoresearch_core.observability.services import (
    MissionControlServices,
    compose_mission_control_services,
)
from autoresearch_core.observability.store import (
    ACTIVE_LOOP_HEARTBEAT_TIMEOUT_SEC,
    SCHEMA_SQL,
    BlobStore,
    ObservabilityStore,
)
from autoresearch_core.observability.writer import CoreObservabilityWriter

__all__ = [
    "MissionControlFacade",
    "MissionControlServices",
    "LOOP_PAUSED",
    "LOOP_RESUMED",
    "LOOP_STOPPED",
    "LOOP_STOP_REQUESTED",
    "LOOP_TERMINATED",
    "LOOP_TERMINATE_REQUESTED",
    "QUEUE_APPLIED",
    "QUEUE_CLAIMED",
    "QUEUE_ENQUEUED",
    "ACTIVE_LOOP_HEARTBEAT_TIMEOUT_SEC",
    "BlobStore",
    "build_backfill_run_id",
    "ensure_backfill_run",
    "CoreMissionControlQueries",
    "CoreObservabilityWriter",
    "HERMES_SESSION_IMPORTED",
    "HERMES_TRANSCRIPT_IMPORTED",
    "LoopAlreadyRunningError",
    "LoopController",
    "MissionControlLogHandler",
    "ObservabilityStore",
    "REDACTED",
    "SCHEMA_SQL",
    "build_current_candidate_pool",
    "build_current_run_summary",
    "build_hermes_runtime_summary",
    "build_mission_control_summary_payload",
    "build_run_policy_context",
    "compose_mission_control_services",
    "compute_loop_analytics",
    "compute_multi_loop_analytics",
    "import_hermes_state",
    "load_nearest_run_manifest",
    "parse_iso_timestamp",
    "read_json_lines",
    "redact_payload",
    "redact_text",
    "resolve_manifest_run_reference",
]
