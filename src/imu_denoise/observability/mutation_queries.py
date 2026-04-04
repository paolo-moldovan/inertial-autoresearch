"""IMU-specific mutation-memory query helpers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from math import log1p
from typing import TYPE_CHECKING, Any

from imu_denoise.data.diagnostics import temporal_decay_weight

if TYPE_CHECKING:
    from imu_denoise.observability.queries import MissionControlQueries


def get_related_mutation_lessons(
    queries: MissionControlQueries,
    run_id: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    attempts = list_mutation_attempts(queries, run_id=run_id, limit=100)
    if not attempts:
        return []
    signatures = [str(item["signature"]) for item in attempts if item.get("signature")]
    regime_fingerprint = queries.get_run_regime_fingerprint(run_id)
    if not signatures or regime_fingerprint is None:
        return []
    placeholders = ", ".join("?" for _ in signatures)
    rows = queries.store.fetch_all(
        f"""
        SELECT
            l.*,
            sig.display_name,
            r.name AS run_name
        FROM mutation_lessons l
        JOIN mutation_signatures sig ON sig.signature = l.signature
        LEFT JOIN runs r ON r.id = l.run_id
        WHERE l.regime_fingerprint = ?
          AND l.signature IN ({placeholders})
        ORDER BY l.created_at DESC
        LIMIT ?
        """,
        (regime_fingerprint, *signatures, limit),
    )
    return rows


def list_mutation_leaderboard(
    queries: MissionControlQueries,
    *,
    limit: int = 10,
    regime_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    where_clause = ""
    params: list[Any] = []
    if regime_fingerprint is not None:
        where_clause = "WHERE s.regime_fingerprint = ?"
        params.append(regime_fingerprint)
    rows = queries.store.fetch_all(
        f"""
        SELECT
            s.signature,
            sig.display_name,
            s.regime_fingerprint,
            s.category,
            s.path,
            s.tries,
            s.keep_count,
            s.discard_count,
            s.crash_count,
            s.avg_metric_delta,
            s.last_metric_delta,
            s.last_status,
            s.last_run_id,
            r.name AS last_run_name,
            s.confidence,
            s.updated_at
        FROM mutation_stats s
        JOIN mutation_signatures sig ON sig.signature = s.signature
        LEFT JOIN runs r ON r.id = s.last_run_id
        {where_clause}
        ORDER BY s.confidence DESC, s.keep_count DESC,
                 COALESCE(s.avg_metric_delta, -1e9) DESC, s.updated_at DESC
        LIMIT ?
        """,
        (*params, limit),
    )
    return rows


def list_recent_mutation_lessons(
    queries: MissionControlQueries,
    *,
    limit: int = 20,
    regime_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    where_clause = ""
    params: list[Any] = []
    if regime_fingerprint is not None:
        where_clause = "WHERE l.regime_fingerprint = ?"
        params.append(regime_fingerprint)
    rows = queries.store.fetch_all(
        f"""
        SELECT
            l.*,
            sig.display_name,
            r.name AS run_name
        FROM mutation_lessons l
        JOIN mutation_signatures sig ON sig.signature = l.signature
        LEFT JOIN runs r ON r.id = l.run_id
        {where_clause}
        ORDER BY l.created_at DESC
        LIMIT ?
        """,
        (*params, limit),
    )
    return rows


def list_mutation_attempts(
    queries: MissionControlQueries,
    *,
    run_id: str | None = None,
    regime_fingerprint: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    params: list[Any] = []
    if run_id is not None:
        conditions.append("a.run_id = ?")
        params.append(run_id)
    if regime_fingerprint is not None:
        conditions.append("a.regime_fingerprint = ?")
        params.append(regime_fingerprint)
    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    rows = queries.store.fetch_all(
        f"""
        SELECT
            a.*,
            sig.display_name,
            sig.category,
            sig.path,
            r.name AS run_name
        FROM mutation_attempts a
        JOIN mutation_signatures sig ON sig.signature = a.signature
        LEFT JOIN runs r ON r.id = a.run_id
        {where_clause}
        ORDER BY a.created_at DESC
        LIMIT ?
        """,
        (*params, limit),
    )
    return rows


def get_mutation_stats_for_signatures(
    queries: MissionControlQueries,
    *,
    signatures: list[str],
    regime_fingerprint: str,
) -> dict[str, dict[str, Any]]:
    if not signatures:
        return {}
    placeholders = ", ".join("?" for _ in signatures)
    rows = queries.store.fetch_all(
        f"""
        SELECT
            s.*,
            sig.display_name
        FROM mutation_stats s
        JOIN mutation_signatures sig ON sig.signature = s.signature
        WHERE s.regime_fingerprint = ?
          AND s.signature IN ({placeholders})
        """,
        (regime_fingerprint, *signatures),
    )
    stats = {str(row["signature"]): dict(row) for row in rows}
    priors = _cross_regime_priors_for_signatures(
        queries,
        signatures=signatures,
        exclude_regime_fingerprint=regime_fingerprint,
    )
    for signature in signatures:
        prior = priors.get(signature)
        stat = stats.get(signature)
        local_tries = 0 if stat is None else int(stat.get("tries", 0))
        if prior is None or local_tries >= 3:
            if stat is not None:
                stat.setdefault("prior_strength", 0.0)
            continue
        prior_strength = min(float(prior["prior_strength"]), 0.30 / max(1, local_tries + 1))
        if stat is None:
            stats[signature] = {
                "signature": signature,
                "display_name": prior.get("display_name", signature),
                "tries": 0,
                "keep_count": 0,
                "discard_count": 0,
                "crash_count": 0,
                "avg_metric_delta": 0.0,
                "confidence": 0.0,
            }
            stat = stats[signature]
        stat["prior_strength"] = prior_strength
        stat["prior_confidence"] = prior["prior_confidence"]
        stat["prior_avg_metric_delta"] = prior["prior_avg_metric_delta"]
        stat["prior_discard_rate"] = prior["prior_discard_rate"]
        stat["prior_crash_rate"] = prior["prior_crash_rate"]
        stat["prior_tries"] = prior["prior_tries"]
        stat["evidence_scope"] = "blended" if local_tries > 0 else "cross_regime_prior"
    return stats


def _cross_regime_priors_for_signatures(
    queries: MissionControlQueries,
    *,
    signatures: Sequence[str],
    exclude_regime_fingerprint: str,
) -> dict[str, dict[str, Any]]:
    if not signatures:
        return {}
    placeholders = ", ".join("?" for _ in signatures)
    rows = queries.store.fetch_all(
        f"""
        SELECT
            a.signature,
            a.status,
            a.metric_delta,
            a.created_at,
            sig.display_name
        FROM mutation_attempts a
        JOIN mutation_signatures sig ON sig.signature = a.signature
        WHERE a.signature IN ({placeholders})
          AND a.regime_fingerprint != ?
        ORDER BY a.created_at DESC
        """,
        (*signatures, exclude_regime_fingerprint),
    )
    now = datetime.now(tz=UTC)
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        signature = str(row["signature"])
        bucket = aggregates.setdefault(
            signature,
            {
                "display_name": row.get("display_name") or signature,
                "weighted_tries": 0.0,
                "weighted_keep": 0.0,
                "weighted_discard": 0.0,
                "weighted_crash": 0.0,
                "weighted_delta_sum": 0.0,
                "weighted_delta_count": 0.0,
            },
        )
        created_at = row.get("created_at")
        age_days = 0.0
        if isinstance(created_at, str):
            try:
                created = datetime.fromisoformat(created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                age_days = max(0.0, (now - created).total_seconds() / 86_400.0)
            except ValueError:
                age_days = 0.0
        weight = temporal_decay_weight(age_days)
        bucket["weighted_tries"] += weight
        status = str(row.get("status") or "")
        if status == "keep":
            bucket["weighted_keep"] += weight
        elif status == "discard":
            bucket["weighted_discard"] += weight
        elif status == "crash":
            bucket["weighted_crash"] += weight
        metric_delta = row.get("metric_delta")
        if isinstance(metric_delta, (int, float)):
            bucket["weighted_delta_sum"] += weight * float(metric_delta)
            bucket["weighted_delta_count"] += weight

    priors: dict[str, dict[str, Any]] = {}
    for signature, bucket in aggregates.items():
        weighted_tries = float(bucket["weighted_tries"])
        if weighted_tries <= 0.0:
            continue
        weighted_delta_count = float(bucket["weighted_delta_count"])
        prior_avg_metric_delta = (
            float(bucket["weighted_delta_sum"]) / weighted_delta_count
            if weighted_delta_count > 0.0
            else 0.0
        )
        prior_confidence = min(0.30, 0.08 + 0.08 * log1p(weighted_tries))
        priors[signature] = {
            "display_name": bucket["display_name"],
            "prior_tries": weighted_tries,
            "prior_avg_metric_delta": prior_avg_metric_delta,
            "prior_confidence": prior_confidence,
            "prior_discard_rate": float(bucket["weighted_discard"]) / weighted_tries,
            "prior_crash_rate": float(bucket["weighted_crash"]) / weighted_tries,
            "prior_strength": min(0.30, weighted_tries / 6.0),
        }
    return priors
