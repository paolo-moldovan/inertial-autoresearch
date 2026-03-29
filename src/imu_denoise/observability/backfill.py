"""Historical artifact backfill into the mission-control store."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from imu_denoise.config import ExperimentConfig
from imu_denoise.observability.events import LOG_EVENT
from imu_denoise.observability.writer import ObservabilityWriter


def _run_id(name: str, phase: str) -> str:
    return f"backfill:{phase}:{name}"


def _parse_timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(dict(json.loads(stripped)))
    return rows


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

    for history_path in sorted(logs_dir.glob("*.history.jsonl")):
        run_name = history_path.name.removesuffix(".history.jsonl")
        run_id = _run_id(run_name, "training")
        obs_writer.start_run(
            name=run_name,
            phase="training",
            dataset=None,
            model=None,
            device=None,
            config=config,
            source="backfill",
            run_id=run_id,
        )
        counts["runs"] += 1
        best_metric: float | None = None
        for record in _read_json_lines(history_path):
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

    for log_path in sorted(logs_dir.glob("*.jsonl")):
        run_name = log_path.name.removesuffix(".jsonl")
        run_id = _run_id(run_name, "training")
        obs_writer.start_run(
            name=run_name,
            phase="training",
            config=config,
            source="backfill",
            run_id=run_id,
        )
        obs_writer.register_artifact(
            run_id=run_id,
            path=log_path,
            artifact_type="log",
            label=log_path.name,
            source="backfill",
        )
        counts["artifacts"] += 1
        for record in _read_json_lines(log_path):
            timestamp = record.get("timestamp")
            created_at = _parse_timestamp(str(timestamp)) if timestamp is not None else None
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
        run_name = metrics_path.parent.name
        if metrics_path.parent.parent.name == "baselines":
            run_name = metrics_path.parent.parent.parent.name
        elif metrics_path.parent.name == "evaluation":
            run_name = metrics_path.parent.parent.name
        run_id = _run_id(run_name, "training")
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
        run_name = artifact_path.parent.name
        if "checkpoints" in artifact_path.parts:
            run_name = artifact_path.parent.name
        elif artifact_path.parent.name == "evaluation":
            run_name = artifact_path.parent.parent.name
        elif artifact_path.parent.parent.name == "baselines":
            run_name = artifact_path.parent.parent.parent.name
        run_id = _run_id(run_name, "training")
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
                run_id = _run_id(run_name, "training")
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
