"""IMU-specific regime and incumbent query helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from autoresearch_core.observability.queries import _loads
from imu_denoise.observability.lineage import (
    data_regime_fingerprint,
    model_is_causal,
    normalize_config_payload,
)

if TYPE_CHECKING:
    from imu_denoise.observability.queries import MissionControlQueries


def list_leaderboard(
    queries: MissionControlQueries,
    *,
    limit: int = 10,
    metric_key: str = "val_rmse",
    regime_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    metric_keys = [metric_key]
    if metric_key == "val_rmse":
        metric_keys.append("rmse")
    where_clause = ""
    params: list[Any] = [metric_keys[0], metric_keys[-1]]
    if regime_fingerprint is not None:
        where_clause = "AND e.regime_fingerprint = ?"
        params.append(regime_fingerprint)
    rows = queries.store.fetch_all(
        """
        WITH ranked_decisions AS (
            SELECT
                d.*,
                ROW_NUMBER() OVER (
                    PARTITION BY d.run_id
                    ORDER BY d.created_at DESC, d.id DESC
                ) AS row_num
            FROM decisions d
            WHERE d.metric_key IN (?, ?)
        )
        SELECT
            r.id AS run_id,
            r.name AS run_name,
            r.model,
            r.phase,
            r.status AS run_status,
            r.experiment_id,
            e.regime_fingerprint,
            rd.proposal_source,
            rd.status AS decision_status,
            rd.metric_key,
            COALESCE(rd.metric_value, s.best_metric) AS metric_value,
            rd.iteration,
            rd.description,
            r.started_at
        FROM runs r
        LEFT JOIN experiments e ON e.id = r.experiment_id
        LEFT JOIN ranked_decisions rd ON rd.run_id = r.id AND rd.row_num = 1
        LEFT JOIN status_snapshots s ON s.run_id = r.id
        WHERE r.phase IN ('training', 'baseline')
          AND r.status = 'completed'
          AND COALESCE(rd.metric_value, s.best_metric) IS NOT NULL
        """
        + where_clause
        + """
        ORDER BY COALESCE(rd.metric_value, s.best_metric) ASC, r.started_at DESC
        LIMIT ?
        """,
        (*params, limit),
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def find_best_global_incumbent(
    queries: MissionControlQueries,
    *,
    metric_key: str,
    dataset: str,
    direction: str = "minimize",
    reference_config: Mapping[str, Any] | Any | None = None,
) -> dict[str, Any] | None:
    metric_keys = [metric_key]
    if metric_key == "val_rmse":
        metric_keys.append("rmse")
    order_direction = "DESC" if direction == "maximize" else "ASC"
    rows = queries.store.fetch_all(
        f"""
        WITH ranked_decisions AS (
            SELECT
                d.*,
                ROW_NUMBER() OVER (
                    PARTITION BY d.run_id
                    ORDER BY d.created_at DESC, d.id DESC
                ) AS row_num
            FROM decisions d
            WHERE d.metric_key IN (?, ?)
        )
        SELECT
            r.id AS run_id,
            r.name AS run_name,
            r.dataset,
            r.model,
            e.config_json,
            e.regime_fingerprint,
            rd.proposal_source,
            rd.status AS decision_status,
            rd.metric_key,
            COALESCE(rd.metric_value, s.best_metric) AS metric_value,
            r.started_at
        FROM runs r
        LEFT JOIN experiments e ON e.id = r.experiment_id
        LEFT JOIN ranked_decisions rd ON rd.run_id = r.id AND rd.row_num = 1
        LEFT JOIN status_snapshots s ON s.run_id = r.id
        WHERE r.phase IN ('training', 'baseline')
          AND r.status = 'completed'
          AND r.dataset = ?
          AND COALESCE(rd.metric_value, s.best_metric) IS NOT NULL
          AND (rd.status IS NULL OR rd.status IN ('baseline', 'keep', 'completed'))
        ORDER BY COALESCE(rd.metric_value, s.best_metric) {order_direction}, r.started_at DESC
        """,
        (metric_keys[0], metric_keys[-1], dataset),
    )
    reference_payload: Mapping[str, Any] | None = None
    if reference_config is not None:
        candidate_payload = normalize_config_payload(reference_config)
        if isinstance(candidate_payload, Mapping):
            reference_payload = candidate_payload
    reference_fingerprint = (
        data_regime_fingerprint(reference_payload)
        if reference_payload is not None
        else None
    )
    if reference_fingerprint is not None:
        filtered = [
            row
            for row in rows
            if (
                row.get("regime_fingerprint") == reference_fingerprint
                or (
                    row.get("regime_fingerprint") in {None, ""}
                    and isinstance(_loads(row.get("config_json")), Mapping)
                    and data_regime_fingerprint(_loads(row.get("config_json")))
                    == reference_fingerprint
                )
            )
        ]
        if filtered:
            selected = filtered[0]
            selected.pop("config_json", None)
            return selected
    for row in rows:
        config_payload = _loads(row.pop("config_json", None))
        if reference_fingerprint is None:
            return row
        if not isinstance(config_payload, Mapping):
            continue
        if data_regime_fingerprint(config_payload) == reference_fingerprint:
            return row
    return None


def get_run_identity(queries: MissionControlQueries, run_id: str) -> dict[str, Any] | None:
    row = super(type(queries), queries).get_run_identity(run_id)
    if row is None:
        return None
    payload = queries.get_run_config_payload(run_id)
    row["causal"] = None if payload is None else model_is_causal(payload)
    return row


def get_run_regime_fingerprint(queries: MissionControlQueries, run_id: str) -> str | None:
    payload = queries.get_run_config_payload(run_id)
    if payload is None:
        return None
    return data_regime_fingerprint(payload)


def get_run_reference(
    queries: MissionControlQueries,
    run_id: str | None,
) -> dict[str, Any] | None:
    row = super(type(queries), queries).get_run_reference(run_id)
    if row is None:
        return None
    payload = queries.get_run_config_payload(str(row["run_id"]))
    row["causal"] = None if payload is None else model_is_causal(payload)
    return row
