"""Mutation-memory tracking helpers for IMU observability."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from imu_denoise.observability.lineage import build_mutation_signatures

if TYPE_CHECKING:
    from imu_denoise.observability.writer import ObservabilityWriter


def _now_ts() -> float:
    return time.time()


def record_mutation_outcome(
    writer: ObservabilityWriter,
    *,
    run_id: str,
    loop_run_id: str | None,
    regime_fingerprint: str,
    proposal_source: str,
    description: str,
    change_items: list[dict[str, Any]],
    status: str,
    metric_key: str,
    metric_value: float | None,
    incumbent_metric: float | None,
    direction: str,
    source: str = "runtime",
) -> list[dict[str, Any]]:
    if writer.store is None:
        return []
    signatures = build_mutation_signatures(change_items)
    if not signatures:
        return []

    metric_delta = _metric_delta(
        metric_value=metric_value,
        incumbent_metric=incumbent_metric,
        direction=direction,
    )
    created_at = _now_ts()
    attempts: list[dict[str, Any]] = []
    for signature_payload in signatures:
        signature = str(signature_payload["signature"])
        category = str(signature_payload["category"])
        path = signature_payload.get("path")
        writer._critical(
            writer.store.upsert_mutation_signature,
            signature=signature,
            display_name=str(signature_payload["display_name"]),
            category=category,
            path=None if path is None else str(path),
            before=signature_payload.get("before"),
            after=signature_payload.get("after"),
            created_at=created_at,
            source=source,
        )
        writer._critical(
            writer.store.insert_mutation_attempt,
            run_id=run_id,
            loop_run_id=loop_run_id,
            signature=signature,
            regime_fingerprint=regime_fingerprint,
            proposal_source=proposal_source,
            status=status,
            metric_key=metric_key,
            metric_value=metric_value,
            incumbent_metric=incumbent_metric,
            metric_delta=metric_delta,
            created_at=created_at,
            source=source,
        )
        aggregate = _aggregate_mutation_stats(
            writer,
            signature=signature,
            regime_fingerprint=regime_fingerprint,
        )
        confidence = _mutation_confidence(
            tries=int(aggregate["tries"]),
            keep_count=int(aggregate["keep_count"]),
            discard_count=int(aggregate["discard_count"]),
            crash_count=int(aggregate["crash_count"]),
            avg_metric_delta=aggregate["avg_metric_delta"],
        )
        writer._critical(
            writer.store.upsert_mutation_stat,
            signature=signature,
            regime_fingerprint=regime_fingerprint,
            category=category,
            path=None if path is None else str(path),
            tries=int(aggregate["tries"]),
            keep_count=int(aggregate["keep_count"]),
            discard_count=int(aggregate["discard_count"]),
            crash_count=int(aggregate["crash_count"]),
            avg_metric_delta=aggregate["avg_metric_delta"],
            last_metric_delta=metric_delta,
            last_status=status,
            last_run_id=run_id,
            confidence=confidence,
            updated_at=created_at,
        )
        lesson = _mutation_lesson(
            display_name=str(signature_payload["display_name"]),
            description=description,
            status=status,
            metric_key=metric_key,
            metric_delta=metric_delta,
        )
        if lesson is not None:
            writer._critical(
                writer.store.insert_mutation_lesson,
                run_id=run_id,
                loop_run_id=loop_run_id,
                signature=signature,
                regime_fingerprint=regime_fingerprint,
                severity=lesson["severity"],
                lesson_type=lesson["lesson_type"],
                summary=lesson["summary"],
                metric_delta=metric_delta,
                created_at=created_at,
                source=source,
                fingerprint=writer._fingerprint(
                    run_id,
                    signature,
                    regime_fingerprint,
                    status,
                    metric_delta,
                    lesson["lesson_type"],
                ),
            )
        attempts.append(
            {
                "signature": signature,
                "display_name": signature_payload["display_name"],
                "category": category,
                "path": path,
                "status": status,
                "metric_key": metric_key,
                "metric_value": metric_value,
                "incumbent_metric": incumbent_metric,
                "metric_delta": metric_delta,
                "confidence": confidence,
            }
        )

    writer.append_event(
        run_id=run_id,
        event_type="mutation_outcome",
        level="INFO",
        title=description,
        payload={
            "proposal_source": proposal_source,
            "status": status,
            "metric_key": metric_key,
            "metric_value": metric_value,
            "incumbent_metric": incumbent_metric,
            "metric_delta": metric_delta,
            "regime_fingerprint": regime_fingerprint,
            "signatures": [item["signature"] for item in attempts],
        },
        source=source,
        created_at=created_at,
        fingerprint=writer._fingerprint(
            run_id,
            "mutation_outcome",
            regime_fingerprint,
            status,
            metric_delta,
        ),
    )
    return attempts


def _aggregate_mutation_stats(
    writer: ObservabilityWriter,
    *,
    signature: str,
    regime_fingerprint: str,
) -> dict[str, Any]:
    if writer.store is None:
        return {
            "tries": 0,
            "keep_count": 0,
            "discard_count": 0,
            "crash_count": 0,
            "avg_metric_delta": None,
        }
    row = writer.store.fetch_one(
        """
        SELECT
            COUNT(*) AS tries,
            COALESCE(SUM(CASE WHEN status = 'keep' THEN 1 ELSE 0 END), 0) AS keep_count,
            COALESCE(SUM(CASE WHEN status = 'discard' THEN 1 ELSE 0 END), 0) AS discard_count,
            COALESCE(SUM(CASE WHEN status = 'crash' THEN 1 ELSE 0 END), 0) AS crash_count,
            AVG(metric_delta) AS avg_metric_delta
        FROM mutation_attempts
        WHERE signature = ? AND regime_fingerprint = ?
        """,
        (signature, regime_fingerprint),
    )
    return row or {
        "tries": 0,
        "keep_count": 0,
        "discard_count": 0,
        "crash_count": 0,
        "avg_metric_delta": None,
    }


def _metric_delta(
    *,
    metric_value: float | None,
    incumbent_metric: float | None,
    direction: str,
) -> float | None:
    if metric_value is None or incumbent_metric is None:
        return None
    if direction == "maximize":
        return metric_value - incumbent_metric
    return incumbent_metric - metric_value


def _mutation_confidence(
    *,
    tries: int,
    keep_count: int,
    discard_count: int,
    crash_count: int,
    avg_metric_delta: float | None,
) -> float:
    evidence = min(1.0, tries / 4.0)
    base = 0.5
    base += min(0.25, 0.08 * keep_count)
    base -= min(0.18, 0.04 * discard_count)
    base -= min(0.3, 0.08 * crash_count)
    if avg_metric_delta is not None:
        base += max(-0.2, min(0.2, avg_metric_delta * 4.0))
    return max(0.0, min(1.0, base * (0.5 + 0.5 * evidence)))


def _mutation_lesson(
    *,
    display_name: str,
    description: str,
    status: str,
    metric_key: str,
    metric_delta: float | None,
) -> dict[str, str] | None:
    if status == "keep":
        if metric_delta is not None:
            summary = (
                f"{display_name} helped: {description} improved {metric_key} by "
                f"{metric_delta:.6f}."
            )
        else:
            summary = f"{display_name} helped: {description} was accepted."
        return {
            "severity": "info",
            "lesson_type": "beneficial_mutation",
            "summary": summary,
        }
    if status == "discard":
        if metric_delta is not None:
            summary = (
                f"{display_name} did not beat the incumbent: {description} changed "
                f"{metric_key} by {metric_delta:.6f}."
            )
        else:
            summary = f"{display_name} was discarded after {description}."
        return {
            "severity": "warning",
            "lesson_type": "non_improving_mutation",
            "summary": summary,
        }
    if status == "crash":
        return {
            "severity": "error",
            "lesson_type": "unstable_mutation",
            "summary": f"{display_name} caused a crash during {description}.",
        }
    return None
