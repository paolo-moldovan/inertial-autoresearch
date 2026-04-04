"""Experiment and lineage tracking helpers for IMU observability."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from imu_denoise.observability.lineage import (
    build_change_items,
    data_regime_fingerprint,
    normalize_config_payload,
    summarize_change_items,
)

if TYPE_CHECKING:
    from imu_denoise.config import ExperimentConfig
    from imu_denoise.observability.writer import ObservabilityWriter


def _now_ts() -> float:
    return time.time()


def ensure_experiment(
    writer: ObservabilityWriter,
    *,
    config: ExperimentConfig,
    overrides: list[str] | None = None,
    objective_metric: str | None = None,
    objective_direction: str | None = None,
    source: str = "runtime",
    summary: dict[str, Any] | None = None,
) -> str | None:
    if writer.store is None:
        return None
    config_payload = _config_payload(writer, config)
    overrides_payload = list(overrides or [])
    experiment_id = sha256(
        json.dumps(
            {
                "config": config_payload,
                "overrides": overrides_payload,
                "objective_metric": objective_metric,
                "objective_direction": objective_direction,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    writer._safe(
        writer.store.upsert_experiment,
        experiment_id=experiment_id,
        name=config.name,
        config_json=config_payload,
        regime_fingerprint=data_regime_fingerprint(config_payload),
        overrides=overrides_payload,
        objective_metric=objective_metric,
        objective_direction=objective_direction,
        summary=summary,
        source=source,
    )
    return experiment_id


def _config_payload(writer: ObservabilityWriter, config: ExperimentConfig) -> dict[str, Any]:
    raw = asdict(config)
    sanitized = writer._sanitize(raw)
    if isinstance(sanitized, dict):
        return sanitized
    return raw


def config_payload(writer: ObservabilityWriter, config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return a sanitized plain-dict payload for a config-like object."""
    normalized = normalize_config_payload(config)
    sanitized = writer._sanitize(normalized)
    if isinstance(sanitized, dict):
        return sanitized
    return normalized


def record_change_set(
    writer: ObservabilityWriter,
    *,
    run_id: str,
    loop_run_id: str | None,
    parent_run_id: str | None,
    incumbent_run_id: str | None,
    reference_kind: str,
    proposal_source: str,
    description: str,
    overrides: list[str],
    current_config: Mapping[str, Any] | Any,
    reference_config: Mapping[str, Any] | Any | None = None,
    source: str = "runtime",
    change_set_id: str | None = None,
) -> dict[str, Any]:
    change_items = build_change_items(
        current_config=current_config,
        reference_config=reference_config,
        overrides=overrides,
    )
    summary = summarize_change_items(change_items)
    created_change_set_id = change_set_id or f"changeset_{uuid4().hex}"
    if writer.store is not None:
        writer._safe(
            writer.store.upsert_change_set,
            change_set_id=created_change_set_id,
            run_id=run_id,
            loop_run_id=loop_run_id,
            parent_run_id=parent_run_id,
            incumbent_run_id=incumbent_run_id,
            reference_kind=reference_kind,
            proposal_source=proposal_source,
            description=description,
            overrides=overrides,
            change_items=change_items,
            summary=summary,
            created_at=_now_ts(),
            source=source,
        )
    writer.append_event(
        run_id=run_id,
        event_type="change_set_recorded",
        level="INFO",
        title=description,
        payload={
            "change_set_id": created_change_set_id,
            "reference_kind": reference_kind,
            "proposal_source": proposal_source,
            "summary": summary,
        },
        source=source,
        created_at=_now_ts(),
        fingerprint=writer._fingerprint(run_id, "change_set_recorded", created_change_set_id),
    )
    return {
        "id": created_change_set_id,
        "run_id": run_id,
        "loop_run_id": loop_run_id,
        "parent_run_id": parent_run_id,
        "incumbent_run_id": incumbent_run_id,
        "reference_kind": reference_kind,
        "proposal_source": proposal_source,
        "description": description,
        "overrides": list(overrides),
        "change_items": change_items,
        "summary": summary,
    }


def record_selection_event(
    writer: ObservabilityWriter,
    *,
    run_id: str,
    loop_run_id: str | None,
    iteration: int | None,
    proposal_source: str,
    description: str,
    incumbent_run_id: str | None,
    candidate_count: int | None,
    rationale: str | None,
    policy_state: dict[str, Any] | None = None,
    source: str = "runtime",
    fingerprint: str | None = None,
) -> dict[str, Any]:
    created_at = _now_ts()
    sanitized_policy = writer._sanitize(policy_state)
    if writer.store is not None:
        writer._safe(
            writer.store.upsert_selection_event,
            fingerprint=fingerprint,
            run_id=run_id,
            loop_run_id=loop_run_id,
            iteration=iteration,
            proposal_source=proposal_source,
            description=description,
            incumbent_run_id=incumbent_run_id,
            candidate_count=candidate_count,
            rationale=rationale,
            policy_state=sanitized_policy,
            created_at=created_at,
            source=source,
        )
    writer.append_event(
        run_id=run_id,
        event_type="selection_event",
        level="INFO",
        title=description,
        payload={
            "iteration": iteration,
            "proposal_source": proposal_source,
            "incumbent_run_id": incumbent_run_id,
            "candidate_count": candidate_count,
            "rationale": rationale,
        },
        source=source,
        created_at=created_at,
        fingerprint=(
            fingerprint or writer._fingerprint(run_id, "selection_event", iteration, description)
        ),
    )
    return {
        "run_id": run_id,
        "loop_run_id": loop_run_id,
        "iteration": iteration,
        "proposal_source": proposal_source,
        "description": description,
        "incumbent_run_id": incumbent_run_id,
        "candidate_count": candidate_count,
        "rationale": rationale,
        "policy_state": sanitized_policy,
    }
