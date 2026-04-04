"""Historical artifact backfill into the mission-control store."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from autoresearch_core.observability.backfill import (
    build_backfill_run_id,
    ensure_backfill_run,
    parse_iso_timestamp,
    read_json_lines,
    resolve_manifest_run_reference,
)
from imu_denoise.config import ExperimentConfig
from imu_denoise.observability.events import LOG_EVENT
from imu_denoise.observability.writer import ObservabilityWriter


def _resolve_run_reference(
    path: Path,
    *,
    default_name: str,
    default_phase: str,
) -> tuple[str, str, str]:
    return resolve_manifest_run_reference(
        path,
        default_name=default_name,
        default_phase=default_phase,
    )


def _legacy_run_name_from_metrics_path(metrics_path: Path) -> str:
    run_name = metrics_path.parent.name
    if metrics_path.parent.parent.name == "baselines":
        return metrics_path.parent.parent.parent.name
    if metrics_path.parent.name == "evaluation":
        return metrics_path.parent.parent.name
    return run_name


def _legacy_run_name_from_artifact_path(artifact_path: Path) -> str:
    run_name = artifact_path.parent.name
    if "checkpoints" in artifact_path.parts:
        return artifact_path.parent.name
    if artifact_path.parent.name == "evaluation":
        return artifact_path.parent.parent.name
    if artifact_path.parent.parent.name == "baselines":
        return artifact_path.parent.parent.parent.name
    return run_name


def _infer_artifact_type(path: Path, run_name: str) -> tuple[str, str | None]:
    if path.suffix == ".pt":
        return "checkpoint", path.stem
    if path.suffix == ".png":
        return "figure", path.stem
    if path.name == "metrics.json":
        if "evaluation" in path.parts:
            return "evaluation_metrics", "evaluation"
        if "baselines" in path.parts:
            return "baseline_metrics", path.parent.name
        return "training_metrics", run_name
    if path.suffix == ".jsonl":
        return "log", path.stem
    return "artifact", path.name


def backfill_observability(
    *,
    config: ExperimentConfig,
    writer: ObservabilityWriter | None = None,
) -> dict[str, int]:
    """Import historical artifacts and logs into the observability store."""
    obs_writer = writer or ObservabilityWriter.from_experiment_config(config)
    counts = {"runs": 0, "events": 0, "artifacts": 0, "decisions": 0}

    artifacts_dir = Path(config.output_dir)
    logs_dir = Path(config.log_dir)
    known_run_ids: set[str] = set()

    history_paths = {
        *logs_dir.glob("*.history.jsonl"),
        *((artifacts_dir / "runs").glob("*/logs/history.jsonl")),
    }
    for history_path in sorted(history_paths):
        default_name = (
            history_path.name.removesuffix(".history.jsonl")
            if history_path.name != "history.jsonl"
            else history_path.parents[1].name
        )
        run_id, run_name, phase = _resolve_run_reference(
            history_path,
            default_name=default_name,
            default_phase="training",
        )
        ensure_backfill_run(obs_writer, run_id=run_id, run_name=run_name, phase=phase)
        if run_id not in known_run_ids:
            counts["runs"] += 1
            known_run_ids.add(run_id)
        best_metric: float | None = None
        for record in read_json_lines(history_path):
            epoch = int(record["epoch"])
            val_rmse = float(record["val_rmse"])
            best_metric = val_rmse if best_metric is None else min(best_metric, val_rmse)
            obs_writer.record_epoch(
                run_id=run_id,
                epoch=epoch,
                train_loss=float(record["train_loss"]),
                val_loss=float(record["val_loss"]),
                val_rmse=val_rmse,
                lr=float(record["lr"]),
                best_metric=best_metric,
                source="backfill",
            )
            counts["events"] += 1
        obs_writer.register_artifact(
            run_id=run_id,
            path=history_path,
            artifact_type="history",
            label=history_path.name,
            source="backfill",
        )
        counts["artifacts"] += 1
        obs_writer.finish_run(run_id=run_id, status="completed", source="backfill")

    log_paths = {
        *logs_dir.glob("*.jsonl"),
        *((artifacts_dir / "runs").glob("*/logs/runtime.jsonl")),
    }
    for log_path in sorted(log_paths):
        default_name = (
            log_path.name.removesuffix(".jsonl")
            if log_path.name != "runtime.jsonl"
            else log_path.parents[1].name
        )
        run_id, run_name, phase = _resolve_run_reference(
            log_path,
            default_name=default_name,
            default_phase="training",
        )
        ensure_backfill_run(obs_writer, run_id=run_id, run_name=run_name, phase=phase)
        if run_id not in known_run_ids:
            counts["runs"] += 1
            known_run_ids.add(run_id)
        obs_writer.register_artifact(
            run_id=run_id,
            path=log_path,
            artifact_type="log",
            label=log_path.name,
            source="backfill",
        )
        counts["artifacts"] += 1
        for record in read_json_lines(log_path):
            timestamp = record.get("timestamp")
            created_at = parse_iso_timestamp(str(timestamp)) if timestamp is not None else None
            obs_writer.append_event(
                run_id=run_id,
                event_type=LOG_EVENT,
                level=str(record.get("level") or "INFO"),
                title=str(record.get("message") or log_path.name),
                payload=record,
                source="backfill",
                created_at=created_at,
                fingerprint=ObservabilityWriter._fingerprint(run_id, "log", created_at, record),
            )
            counts["events"] += 1

    for metrics_path in sorted(artifacts_dir.rglob("metrics.json")):
        run_id, run_name, phase = _resolve_run_reference(
            metrics_path,
            default_name=_legacy_run_name_from_metrics_path(metrics_path),
            default_phase="training",
        )
        ensure_backfill_run(obs_writer, run_id=run_id, run_name=run_name, phase=phase)
        if run_id not in known_run_ids:
            counts["runs"] += 1
            known_run_ids.add(run_id)
        artifact_type, label = _infer_artifact_type(metrics_path, run_name)
        obs_writer.register_artifact(
            run_id=run_id,
            path=metrics_path,
            artifact_type=artifact_type,
            label=label,
            metadata=(
                dict(json.loads(metrics_path.read_text(encoding="utf-8")))
                if metrics_path.exists()
                else None
            ),
            source="backfill",
        )
        counts["artifacts"] += 1

    for artifact_path in sorted(artifacts_dir.rglob("*")):
        if not artifact_path.is_file():
            continue
        if artifact_path.name == "metrics.json":
            continue
        if artifact_path.suffix not in {".png", ".pt"}:
            continue
        run_id, run_name, phase = _resolve_run_reference(
            artifact_path,
            default_name=_legacy_run_name_from_artifact_path(artifact_path),
            default_phase="training",
        )
        ensure_backfill_run(obs_writer, run_id=run_id, run_name=run_name, phase=phase)
        if run_id not in known_run_ids:
            counts["runs"] += 1
            known_run_ids.add(run_id)
        artifact_type, label = _infer_artifact_type(artifact_path, run_name)
        obs_writer.register_artifact(
            run_id=run_id,
            path=artifact_path,
            artifact_type=artifact_type,
            label=label,
            source="backfill",
        )
        counts["artifacts"] += 1

    results_path = Path(config.autoresearch.results_file)
    if results_path.exists():
        with open(results_path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                run_name = row["run_name"]
                run_id = build_backfill_run_id(run_name, "training")
                ensure_backfill_run(
                    obs_writer,
                    run_id=run_id,
                    run_name=run_name,
                    phase="training",
                )
                if run_id not in known_run_ids:
                    counts["runs"] += 1
                    known_run_ids.add(run_id)
                overrides = json.loads(row["overrides"]) if row["overrides"] else []
                metric_value = float(row["metric_value"]) if row["metric_value"] else None
                obs_writer.record_decision(
                    run_id=run_id,
                    iteration=int(row["iteration"]),
                    proposal_source=row["proposal_source"],
                    description=row["description"],
                    status=row["status"],
                    metric_key=row["metric_key"],
                    metric_value=metric_value,
                    overrides=overrides,
                    reason=None,
                    source="backfill",
                    fingerprint=ObservabilityWriter._fingerprint(
                        run_id,
                        "decision",
                        row["iteration"],
                        row["description"],
                        row["status"],
                    ),
                )
                counts["decisions"] += 1
                metrics_path_value = row.get("metrics_path")
                if metrics_path_value:
                    obs_writer.register_artifact(
                        run_id=run_id,
                        path=str(metrics_path_value),
                        artifact_type="training_metrics",
                        label="autoresearch_metrics",
                        source="backfill",
                    )
                    counts["artifacts"] += 1

    return counts
