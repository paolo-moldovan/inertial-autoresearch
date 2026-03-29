"""Sync mission-control observability runs into MLflow tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imu_denoise.config.schema import ObservabilityConfig
from imu_denoise.observability.events import TRAINING_EPOCH
from imu_denoise.observability.queries import MissionControlQueries


def _import_mlflow() -> Any:
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MLflow is not installed. Install `imu-denoise[monitor-adapters]` to sync runs."
        ) from exc
    return mlflow


class MlflowExporter:
    """Replay mission-control runs into MLflow while keeping SQLite canonical."""

    def __init__(
        self,
        *,
        config: ObservabilityConfig,
        queries: MissionControlQueries,
        mlflow_module: Any | None = None,
    ) -> None:
        self.config = config
        self.queries = queries
        self._mlflow = mlflow_module

    def sync(self, *, run_id: str | None = None, limit: int = 100) -> dict[str, int]:
        mlflow = self._mlflow or _import_mlflow()
        experiment_id = self._ensure_experiment(mlflow)
        run_rows = self._select_runs(run_id=run_id, limit=limit)
        counts = {
            "runs_seen": len(run_rows),
            "runs_synced": 0,
            "artifacts_logged": 0,
            "metrics_logged": 0,
        }
        for row in run_rows:
            detail = self.queries.get_run_detail(str(row["id"]))
            if detail is None:
                continue
            artifact_count, metric_count = self._sync_run(
                mlflow=mlflow,
                experiment_id=experiment_id,
                detail=detail,
            )
            counts["runs_synced"] += 1
            counts["artifacts_logged"] += artifact_count
            counts["metrics_logged"] += metric_count
        return counts

    def _ensure_experiment(self, mlflow: Any) -> str:
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.config.mlflow_experiment_name)
        if experiment is not None:
            return str(experiment.experiment_id)
        created = mlflow.create_experiment(self.config.mlflow_experiment_name)
        return str(created)

    def _select_runs(self, *, run_id: str | None, limit: int) -> list[dict[str, Any]]:
        if run_id is not None:
            detail = self.queries.get_run_detail(run_id)
            return [] if detail is None else [detail["run"]]
        return self.queries.list_runs(limit=limit)

    def _sync_run(
        self,
        *,
        mlflow: Any,
        experiment_id: str,
        detail: dict[str, Any],
    ) -> tuple[int, int]:
        run = detail["run"]
        existing_run_id = self._find_existing_run(
            mlflow=mlflow,
            experiment_id=experiment_id,
            mission_control_run_id=str(run["id"]),
        )
        if existing_run_id is not None:
            active_run = mlflow.start_run(run_id=existing_run_id)
        else:
            active_run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=str(run["name"]),
            )

        artifact_count = 0
        metric_count = 0
        with active_run:
            for key, value in self._tag_payload(run).items():
                mlflow.set_tag(key, value)

            experiment = detail.get("experiment")
            if isinstance(experiment, dict):
                objective_metric = experiment.get("objective_metric")
                if isinstance(objective_metric, str):
                    mlflow.set_tag("mission_control.objective_metric", objective_metric)
                objective_direction = experiment.get("objective_direction")
                if isinstance(objective_direction, str):
                    mlflow.set_tag("mission_control.objective_direction", objective_direction)
                for index, override in enumerate(experiment.get("overrides") or []):
                    mlflow.log_param(f"override.{index}", str(override))

            for key in ("dataset", "model", "device", "phase", "status", "source"):
                value = run.get(key)
                if value is not None:
                    mlflow.log_param(str(key), str(value))
            if run.get("iteration") is not None:
                mlflow.log_param("iteration", int(run["iteration"]))

            best_metric = run.get("best_metric")
            if isinstance(best_metric, (int, float)):
                mlflow.log_metric("best_metric", float(best_metric))
                metric_count += 1
            last_metric = run.get("last_metric")
            if isinstance(last_metric, (int, float)):
                mlflow.log_metric("last_metric", float(last_metric))
                metric_count += 1

            for event in detail.get("timeline", []):
                if event.get("event_type") != TRAINING_EPOCH:
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                step = payload.get("epoch")
                step_value = int(step) if isinstance(step, int) else None
                for metric_name in ("train_loss", "val_loss", "val_rmse", "lr"):
                    metric_value = payload.get(metric_name)
                    if isinstance(metric_value, (int, float)):
                        if step_value is None:
                            mlflow.log_metric(metric_name, float(metric_value))
                        else:
                            mlflow.log_metric(metric_name, float(metric_value), step=step_value)
                        metric_count += 1

            if self.config.mlflow_log_artifacts:
                artifact_count += self._log_json_payload(
                    mlflow,
                    detail["run"],
                    artifact_file="mission_control/run.json",
                )
                experiment = detail.get("experiment")
                if experiment is not None:
                    artifact_count += self._log_json_payload(
                        mlflow,
                        experiment,
                        artifact_file="mission_control/experiment.json",
                    )
                for name in ("timeline", "decisions", "llm_calls", "tool_calls", "logs"):
                    artifact_count += self._log_json_payload(
                        mlflow,
                        detail.get(name, []),
                        artifact_file=f"mission_control/{name}.json",
                    )
                for artifact in detail.get("artifacts", []):
                    artifact_path = Path(str(artifact["path"]))
                    if not artifact_path.exists():
                        continue
                    mlflow.log_artifact(
                        str(artifact_path),
                        artifact_path=f"linked_artifacts/{artifact['artifact_type']}",
                    )
                    artifact_count += 1
        return artifact_count, metric_count

    def _find_existing_run(
        self,
        *,
        mlflow: Any,
        experiment_id: str,
        mission_control_run_id: str,
    ) -> str | None:
        safe_run_id = mission_control_run_id.replace("'", "\\'")
        matches = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mission_control_run_id = '{safe_run_id}'",
            output_format="list",
            max_results=1,
        )
        if not matches:
            return None
        return str(matches[0].info.run_id)

    def _tag_payload(self, run: dict[str, Any]) -> dict[str, str]:
        tags: dict[str, str] = {
            "mission_control_run_id": str(run["id"]),
            "mission_control.phase": str(run["phase"]),
            "mission_control.status": str(run["status"]),
        }
        if run.get("parent_run_id") is not None:
            tags["mission_control.parent_run_id"] = str(run["parent_run_id"])
        if run.get("experiment_id") is not None:
            tags["mission_control.experiment_id"] = str(run["experiment_id"])
        return tags

    def _log_json_payload(self, mlflow: Any, payload: Any, *, artifact_file: str) -> int:
        if hasattr(mlflow, "log_dict"):
            mlflow.log_dict(payload, artifact_file)
            return 1
        if hasattr(mlflow, "log_text"):
            mlflow.log_text(
                json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str),
                artifact_file,
            )
            return 1
        return 0
