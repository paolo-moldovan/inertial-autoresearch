"""Tests for the unified CLI, control plane, and Mission Control queries."""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from autoresearch_loop.loop import run_autoresearch
from imu_denoise.cli import imu
from imu_denoise.cli.common import resolve_config
from imu_denoise.models import get_model, list_models
from imu_denoise.models import registry as model_registry
from imu_denoise.observability import (
    LoopAlreadyRunningError,
    LoopController,
    MissionControlQueries,
    ObservabilityWriter,
)


def _run_imu_command(argv: list[str]) -> int:
    args = imu.build_parser().parse_args(argv)
    pre_handler = getattr(args, "pre_handler", None)
    if callable(pre_handler):
        pre_handler(args)
    return int(args.handler(args))


def _small_runtime_overrides(tmp_path: Path) -> list[str]:
    return [
        f"output_dir={tmp_path / 'artifacts'}",
        f"log_dir={tmp_path / 'artifacts' / 'logs'}",
        f"observability.db_path={tmp_path / 'artifacts' / 'observability' / 'mission_control.db'}",
        f"observability.blob_dir={tmp_path / 'artifacts' / 'observability' / 'blobs'}",
        "training.epochs=1",
        "training.batch_size=4",
        "training.num_workers=0",
        "data.dataset_kwargs.duration_sec=2.0",
        "data.dataset_kwargs.rate_hz=20.0",
        "data.dataset_kwargs.num_sequences=3",
        "data.window_size=20",
        "data.stride=10",
    ]


def test_imu_run_and_baseline_record_manual_decisions(tmp_path: Path) -> None:
    """Manual CLI runs should surface alongside loop runs in Mission Control."""
    common = ["--config", "configs/training/quick.yaml"]
    for override in _small_runtime_overrides(tmp_path):
        common.extend(["--set", override])

    assert _run_imu_command(["run", *common, "--name", "manual-lstm"]) == 0
    assert _run_imu_command(["eval", *common, "--name", "manual-lstm"]) == 0
    assert _run_imu_command(
        ["baseline", *common, "--baseline", "kalman", "--name", "manual-kf"]
    ) == 0

    queries = MissionControlQueries(
        db_path=tmp_path / "artifacts" / "observability" / "mission_control.db",
        blob_dir=tmp_path / "artifacts" / "observability" / "blobs",
    )

    manual_runs = queries.list_runs_by_source("manual", limit=20)
    assert any(row["phase"] == "training" for row in manual_runs)
    assert any(row["phase"] == "baseline" for row in manual_runs)

    runs = queries.list_runs(limit=50)
    assert any(
        row["phase"] == "evaluation" and row["name"] == "manual-lstm-evaluation"
        for row in runs
    )

    leaderboard = queries.list_leaderboard(limit=10)
    assert any(
        row["proposal_source"] == "manual" and row["phase"] == "training"
        for row in leaderboard
    )
    assert any(
        row["proposal_source"] == "manual" and row["phase"] == "baseline"
        for row in leaderboard
    )

    training_row = next(row for row in leaderboard if row["phase"] == "training")
    run_detail = queries.get_run_detail(str(training_row["run_id"]))
    assert run_detail is not None
    assert run_detail["curves"]
    assert run_detail["identity"]["experiment_id"] is not None
    assert run_detail["selection_event"] is not None
    assert run_detail["selection_event"]["proposal_source"] == "manual"
    assert run_detail["change_set"] is not None
    assert run_detail["change_set"]["reference_kind"] == "manual"
    assert run_detail["links"]["experiment_id"] == run_detail["identity"]["experiment_id"]
    assert any("/runs/" in str(artifact["path"]) for artifact in run_detail["artifacts"])
    assert (
        queries.resolve_id_fragment(str(training_row["run_id"])[:8])["entity_type"] == "run"  # type: ignore[index]
    )
    assert (
        queries.resolve_id_fragment(str(run_detail["identity"]["experiment_id"])[:8])["entity_type"]  # type: ignore[index]
        == "experiment"
    )


def test_model_autodiscovery_and_config_autoload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New model modules and configs should be discovered without editing __init__ files."""
    import imu_denoise.models as models_package

    module_name = "wavenet_autotest"
    full_name = f"imu_denoise.models.{module_name}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import torch",
                "from torch import Tensor",
                "from imu_denoise.models.base import BaseDenoiser",
                "from imu_denoise.models.registry import register_model",
                "",
                '@register_model("wavenet_autotest")',
                "class WaveNetAutoTest(BaseDenoiser):",
                "    def __init__(",
                "        self,",
                "        hidden_dim: int = 8,",
                "        num_layers: int = 1,",
                "        dropout: float = 0.0,",
                "        **_: object,",
                "    ) -> None:",
                "        super().__init__()",
                "        self.proj = torch.nn.Linear(6, 6)",
                "",
                "    def forward(",
                "        self, noisy_imu: Tensor, timestamps: Tensor | None = None",
                "    ) -> Tensor:",
                "        return self.proj(noisy_imu)",
            ]
        ),
        encoding="utf-8",
    )

    original_path = list(models_package.__path__)
    monkeypatch.setattr(models_package, "__path__", [*original_path, str(tmp_path)])
    importlib.invalidate_caches()
    model_registry._DISCOVERED_MODULES.discard(full_name)
    sys.modules.pop(full_name, None)

    discovered = model_registry.autodiscover_models()
    assert module_name in discovered
    assert "wavenet_autotest" in list_models()
    assert get_model("wavenet_autotest").__class__.__name__ == "WaveNetAutoTest"

    config_path = Path("configs/models/test_autoload_model.yaml")
    config_path.write_text(
        "model:\n"
        "  name: test_autoload_model\n"
        "  hidden_dim: 17\n"
        "  dropout: 0.25\n",
        encoding="utf-8",
    )
    try:
        config = resolve_config([], ["model.name=test_autoload_model"])
    finally:
        config_path.unlink(missing_ok=True)

    assert config.model.name == "test_autoload_model"
    assert config.model.hidden_dim == 17
    assert config.model.dropout == 0.25


def test_batch_pause_queue_and_resume_prioritize_human_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queued proposals should be consumed before Hermes once a paused loop resumes."""
    base_overrides = _small_runtime_overrides(tmp_path) + [
        f"autoresearch.results_file={tmp_path / 'artifacts' / 'autoresearch' / 'results.tsv'}",
        "autoresearch.orchestrator=hermes",
        "autoresearch.max_iterations=1",
    ]

    monkeypatch.setattr(
        "autoresearch_loop.hermes.hermes_backend_ready",
        lambda *args, **kwargs: True,
    )

    def _should_not_be_called(**_: Any) -> Any:
        raise AssertionError("Hermes should not be consulted when a queued proposal is pending.")

    monkeypatch.setattr(
        "autoresearch_loop.hermes.choose_mutation_proposal_with_trace",
        _should_not_be_called,
    )

    holder: dict[str, Any] = {}

    def _target() -> None:
        holder["results"] = run_autoresearch(
            config_paths=["configs/training/quick.yaml"],
            base_overrides=base_overrides,
            max_iterations=1,
            batch_size=1,
            pause_enabled=True,
        )

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    queries = MissionControlQueries(
        db_path=tmp_path / "artifacts" / "observability" / "mission_control.db",
        blob_dir=tmp_path / "artifacts" / "observability" / "blobs",
    )
    paused_state: dict[str, Any] | None = None
    for _ in range(200):
        paused_state = queries.get_active_loop_state()
        if paused_state is not None and paused_state["status"] == "paused":
            break
        time.sleep(0.1)

    assert paused_state is not None
    assert paused_state["status"] == "paused"
    assert paused_state["current_iteration"] == 1

    config = resolve_config(["configs/training/quick.yaml"], base_overrides)
    writer = ObservabilityWriter.from_experiment_config(config)
    controller = LoopController.from_experiment_config(config, writer=writer)
    proposal = controller.enqueue_proposal(
        description="try huber loss from human queue",
        overrides=["training.loss=huber"],
        requested_by="pytest",
    )
    assert proposal["status"] == "pending"

    queued = queries.list_queued_proposals(str(paused_state["loop_run_id"]))
    assert queued
    assert queued[0]["description"] == "try huber loss from human queue"

    summary = queries.get_mission_control_summary(limit=10)
    assert summary["current_run"] is not None
    assert summary["current_run"]["is_active"] is False
    assert summary["current_run"]["run_name"] == "autoresearch_000"
    assert summary["current_run"]["candidate_pool"]
    assert isinstance(summary["current_run"]["blocked_candidates"], dict)
    assert summary["current_candidate_pool"] is not None
    assert isinstance(summary["current_candidate_pool"]["blocked_candidates"], dict)
    assert summary["current_candidate_pool"]["run_name"] == "autoresearch_000"

    resumed_state = controller.resume_loop(loop_run_id=str(paused_state["loop_run_id"]))
    assert resumed_state is not None
    assert resumed_state["status"] == "running"

    thread.join(timeout=30.0)
    assert not thread.is_alive()

    results = holder["results"]
    assert len(results) == 2
    assert results[1].proposal_source == "human-queued"
    assert "training.loss=huber" in results[1].overrides

    final_status = queries.get_loop_status()
    assert final_status is not None
    assert final_status["status"] == "completed"

    final_queue_rows = queries.list_queued_proposals(str(paused_state["loop_run_id"]))
    assert final_queue_rows[0]["status"] == "applied"
    loop_events = queries.list_recent_loop_events(limit=20)
    event_types = {row["event_type"] for row in loop_events}
    assert "loop_paused" in event_types
    assert "loop_resumed" in event_types
    assert "queue_enqueued" in event_types
    assert "queue_claimed" in event_types
    assert "queue_applied" in event_types


def test_stop_request_stops_a_paused_loop(tmp_path: Path) -> None:
    """A stop request should end the paused loop cleanly without resuming more iterations."""
    base_overrides = _small_runtime_overrides(tmp_path) + [
        f"autoresearch.results_file={tmp_path / 'artifacts' / 'autoresearch' / 'results.tsv'}",
        "autoresearch.orchestrator=none",
        "autoresearch.max_iterations=2",
    ]
    holder: dict[str, Any] = {}

    def _target() -> None:
        holder["results"] = run_autoresearch(
            config_paths=["configs/training/quick.yaml"],
            base_overrides=base_overrides,
            max_iterations=2,
            batch_size=1,
            pause_enabled=True,
        )

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    queries = MissionControlQueries(
        db_path=tmp_path / "artifacts" / "observability" / "mission_control.db",
        blob_dir=tmp_path / "artifacts" / "observability" / "blobs",
    )
    paused_state: dict[str, Any] | None = None
    for _ in range(200):
        paused_state = queries.get_active_loop_state()
        if paused_state is not None and paused_state["status"] == "paused":
            break
        time.sleep(0.1)

    assert paused_state is not None
    config = resolve_config(["configs/training/quick.yaml"], base_overrides)
    writer = ObservabilityWriter.from_experiment_config(config)
    controller = LoopController.from_experiment_config(config, writer=writer)
    stopped = controller.request_stop(loop_run_id=str(paused_state["loop_run_id"]))
    assert stopped is not None
    assert stopped["stop_requested"] is True

    thread.join(timeout=30.0)
    assert not thread.is_alive()

    final_status = queries.get_loop_status()
    assert final_status is not None
    assert final_status["status"] == "stopped"


def test_terminate_request_interrupts_active_training_run(tmp_path: Path) -> None:
    """A terminate request should interrupt the current child run and mark the loop terminated."""
    base_overrides = _small_runtime_overrides(tmp_path) + [
        f"autoresearch.results_file={tmp_path / 'artifacts' / 'autoresearch' / 'results.tsv'}",
        "autoresearch.orchestrator=none",
        "autoresearch.max_iterations=0",
        "training.epochs=20",
        "training.batch_size=1",
        "data.dataset_kwargs.duration_sec=12.0",
        "data.dataset_kwargs.rate_hz=50.0",
        "data.dataset_kwargs.num_sequences=5",
    ]
    holder: dict[str, Any] = {}

    def _target() -> None:
        holder["results"] = run_autoresearch(
            config_paths=["configs/training/quick.yaml"],
            base_overrides=base_overrides,
            max_iterations=0,
        )

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    queries = MissionControlQueries(
        db_path=tmp_path / "artifacts" / "observability" / "mission_control.db",
        blob_dir=tmp_path / "artifacts" / "observability" / "blobs",
    )
    active_state: dict[str, Any] | None = None
    for _ in range(200):
        active_state = queries.get_active_loop_state()
        if active_state is not None and active_state.get("active_child_run_id"):
            break
        time.sleep(0.1)

    assert active_state is not None
    config = resolve_config(["configs/training/quick.yaml"], base_overrides)
    writer = ObservabilityWriter.from_experiment_config(config)
    controller = LoopController.from_experiment_config(config, writer=writer)
    terminated = controller.request_terminate(loop_run_id=str(active_state["loop_run_id"]))
    assert terminated is not None
    assert terminated["terminate_requested"] is True

    thread.join(timeout=30.0)
    assert not thread.is_alive()

    final_status = queries.get_loop_status()
    assert final_status is not None
    assert final_status["status"] == "terminated"


def test_singleton_loop_guard_rejects_second_live_loop(tmp_path: Path) -> None:
    """Only one live loop should be able to acquire the Mission Control control plane."""
    config = resolve_config(
        ["configs/training/quick.yaml"],
        _small_runtime_overrides(tmp_path),
    )
    writer = ObservabilityWriter.from_experiment_config(config)
    controller = LoopController.from_experiment_config(config, writer=writer)

    loop_one = writer.start_run(
        name="loop-one",
        phase="autoresearch_loop",
        dataset=config.data.dataset,
        model=config.model.name,
        device=config.device.preferred,
        config=config,
        overrides=[],
        objective_metric=config.autoresearch.metric_key,
        objective_direction=config.autoresearch.metric_direction,
        source="runtime",
    )
    controller.initialize_loop(
        loop_run_id=loop_one,
        max_iterations=4,
        batch_size=None,
        pause_enabled=False,
        current_iteration=0,
    )

    loop_two = writer.start_run(
        name="loop-two",
        phase="autoresearch_loop",
        dataset=config.data.dataset,
        model=config.model.name,
        device=config.device.preferred,
        config=config,
        overrides=[],
        objective_metric=config.autoresearch.metric_key,
        objective_direction=config.autoresearch.metric_direction,
        source="runtime",
    )

    with pytest.raises(LoopAlreadyRunningError):
        controller.initialize_loop(
            loop_run_id=loop_two,
            max_iterations=4,
            batch_size=None,
            pause_enabled=False,
            current_iteration=0,
        )
