"""Tests for the local autoresearch loop."""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

from _pytest.monkeypatch import MonkeyPatch

from autoresearch_loop.hermes import HermesQueryTrace
from autoresearch_loop.loop import build_parser, run_autoresearch
from autoresearch_loop.mutations import MutationProposal, build_mutation_schedule
from imu_denoise.cli.common import resolve_config
from imu_denoise.config import DataConfig, ExperimentConfig, ObservabilityConfig
from imu_denoise.observability import LoopController, MissionControlQueries
from imu_denoise.observability.writer import ObservabilityWriter


def test_mutation_schedule_is_baseline_first() -> None:
    """Mutation schedules should always start with the baseline configuration."""
    schedule = build_mutation_schedule(3, Random(7))

    assert schedule[0].description == "baseline"
    assert schedule[0].overrides == []
    assert len(schedule) == 4


def test_mutation_schedule_can_skip_baseline() -> None:
    """Schedules without a per-loop baseline should start directly with mutations."""
    schedule = build_mutation_schedule(3, Random(7), include_baseline=False)

    assert schedule[0].description != "baseline"
    assert len(schedule) == 3


def test_autoresearch_loop_writes_results(tmp_path: Path) -> None:
    """The local loop should write a TSV with a baseline and one mutation result."""
    results_path = tmp_path / "results.tsv"
    config_paths = [
        "configs/base.yaml",
        "configs/device.yaml",
        "configs/models/lstm.yaml",
        "configs/training/quick.yaml",
        "configs/autoresearch.yaml",
    ]
    base_overrides = [
        f"autoresearch.results_file={results_path}",
        "autoresearch.orchestrator=none",
        f"observability.db_path={tmp_path / 'observability' / 'mission_control.db'}",
        f"observability.blob_dir={tmp_path / 'observability' / 'blobs'}",
        "training.epochs=1",
        "training.batch_size=4",
        "data.dataset_kwargs.duration_sec=2.0",
        "data.dataset_kwargs.rate_hz=20.0",
        "data.dataset_kwargs.num_sequences=3",
        "data.window_size=20",
        "data.stride=10",
        "autoresearch.max_iterations=1",
    ]

    results = run_autoresearch(
        config_paths=config_paths,
        base_overrides=base_overrides,
        max_iterations=1,
    )

    assert len(results) == 2
    assert results[0].status == "baseline"
    assert results[0].proposal_source == "static"
    assert results_path.exists()
    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    queries = MissionControlQueries(
        db_path=tmp_path / "observability" / "mission_control.db",
        blob_dir=tmp_path / "observability" / "blobs",
    )
    mutation_run = next(
        row
        for row in queries.list_recent_decisions(limit=10)
        if row["run_name"] == "autoresearch_001"
    )
    detail = queries.get_run_detail(str(mutation_run["run_id"]))
    assert detail is not None
    assert detail["selection_event"] is not None
    assert detail["selection_event"]["loop_run_id"] is not None
    assert detail["change_set"] is not None
    assert detail["change_set"]["reference_kind"] in {"base", "incumbent"}
    assert detail["mutation_attempts"]
    summary = queries.get_mission_control_summary(limit=10)
    assert summary["mutation_leaderboard"]
    assert summary["recent_mutation_lessons"]


def test_autoresearch_loop_uses_hermes_proposals_when_enabled(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Hermes should be able to drive proposal selection without changing execution."""
    results_path = tmp_path / "results.tsv"
    config_paths = [
        "configs/base.yaml",
        "configs/device.yaml",
        "configs/models/lstm.yaml",
        "configs/training/quick.yaml",
        "configs/autoresearch.yaml",
    ]
    base_overrides = [
        f"autoresearch.results_file={results_path}",
        "autoresearch.orchestrator=hermes",
        f"observability.db_path={tmp_path / 'observability' / 'mission_control.db'}",
        f"observability.blob_dir={tmp_path / 'observability' / 'blobs'}",
        "observability.import_hermes_state=false",
        "training.epochs=1",
        "training.batch_size=4",
        "data.dataset_kwargs.duration_sec=2.0",
        "data.dataset_kwargs.rate_hz=20.0",
        "data.dataset_kwargs.num_sequences=3",
        "data.window_size=20",
        "data.stride=10",
        "autoresearch.max_iterations=1",
    ]

    def _backend_ready(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr("autoresearch_loop.hermes.hermes_backend_ready", _backend_ready)
    monkeypatch.setattr(
        "autoresearch_loop.hermes.choose_mutation_proposal_with_trace",
        lambda **kwargs: (
            MutationProposal(
                description="switch to huber loss",
                overrides=["training.loss=huber"],
            ),
            HermesQueryTrace(
                prompt="choose",
                command={"argv": ["hermes"]},
                status="ok",
                latency_ms=12.0,
                stdout='{"candidate_index": 0, "reason": "good"}',
                stderr="",
                parsed_payload={"candidate_index": 0, "reason": "good"},
                session_id="session-1",
                reason="good",
            ),
        ),
    )

    results = run_autoresearch(
        config_paths=config_paths,
        base_overrides=base_overrides,
        max_iterations=1,
    )

    assert len(results) == 2
    assert results[1].description == "switch to huber loss"
    assert results[1].proposal_source == "hermes"
    assert "training.loss=huber" in results[1].overrides
    queries = MissionControlQueries(
        db_path=tmp_path / "observability" / "mission_control.db",
        blob_dir=tmp_path / "observability" / "blobs",
    )
    assert len(queries.list_recent_decisions(limit=10)) >= 2
    assert any(
        call["session_id"] == "session-1"
        for call in queries.list_recent_llm_calls(limit=10)
    )


def test_loop_parser_does_not_inject_quick_config_when_explicit_config_is_passed() -> None:
    """Explicit loop configs should not be merged with the synthetic quick profile."""
    parser = build_parser()

    args = parser.parse_args(["--config", "configs/mission_control/hermes_euroc_subset.yaml"])

    assert args.config == ["configs/mission_control/hermes_euroc_subset.yaml"]


def test_global_incumbent_prefers_best_accepted_run_over_latest_baseline(tmp_path: Path) -> None:
    """Global mode should use the best accepted prior run, not merely the last baseline."""
    config = ExperimentConfig(
        observability=ObservabilityConfig(
            enabled=True,
            db_path=str(tmp_path / "observability" / "mission_control.db"),
            blob_dir=str(tmp_path / "observability" / "blobs"),
        ),
    )
    writer = ObservabilityWriter.from_experiment_config(config)
    queries = MissionControlQueries(
        db_path=tmp_path / "observability" / "mission_control.db",
        blob_dir=tmp_path / "observability" / "blobs",
    )

    baseline_run = writer.start_run(
        name="baseline-run",
        phase="training",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=config,
    )
    writer.finish_run(
        run_id=baseline_run,
        status="completed",
        summary={"best_val_rmse": 0.40},
    )
    writer.record_decision(
        run_id=baseline_run,
        iteration=0,
        proposal_source="static",
        description="baseline",
        status="baseline",
        metric_key="val_rmse",
        metric_value=0.40,
        overrides=[],
    )

    keep_run = writer.start_run(
        name="better-run",
        phase="training",
        dataset="synthetic",
        model="conv1d",
        device="cpu",
        config=config,
    )
    writer.finish_run(
        run_id=keep_run,
        status="completed",
        summary={"best_val_rmse": 0.20},
    )
    writer.record_decision(
        run_id=keep_run,
        iteration=1,
        proposal_source="hermes",
        description="better model",
        status="keep",
        metric_key="val_rmse",
        metric_value=0.20,
        overrides=["model.name=conv1d"],
    )

    newer_baseline_run = writer.start_run(
        name="newer-baseline",
        phase="training",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=config,
    )
    writer.finish_run(
        run_id=newer_baseline_run,
        status="completed",
        summary={"best_val_rmse": 0.35},
    )
    writer.record_decision(
        run_id=newer_baseline_run,
        iteration=0,
        proposal_source="static",
        description="baseline",
        status="baseline",
        metric_key="val_rmse",
        metric_value=0.35,
        overrides=[],
    )

    incumbent = queries.find_best_global_incumbent(
        metric_key="val_rmse",
        dataset="synthetic",
        direction="minimize",
        reference_config=config,
    )

    assert incumbent is not None
    assert incumbent["run_id"] == keep_run
    assert incumbent["decision_status"] == "keep"
    assert float(incumbent["metric_value"]) == 0.20


def test_global_incumbent_requires_matching_data_regime(tmp_path: Path) -> None:
    """Global mode should ignore runs from a different split/subset regime."""
    reference_config = ExperimentConfig(
        observability=ObservabilityConfig(
            enabled=True,
            db_path=str(tmp_path / "observability" / "mission_control.db"),
            blob_dir=str(tmp_path / "observability" / "blobs"),
        ),
    )
    writer = ObservabilityWriter.from_experiment_config(reference_config)
    queries = MissionControlQueries(
        db_path=tmp_path / "observability" / "mission_control.db",
        blob_dir=tmp_path / "observability" / "blobs",
    )

    matching_config = ExperimentConfig(
        data=reference_config.data,
        observability=reference_config.observability,
    )
    mismatched_config = ExperimentConfig(
        data=DataConfig(
            dataset="synthetic",
            window_size=32,
            stride=16,
            normalize=True,
            augment=False,
        ),
        observability=reference_config.observability,
    )

    matching_run = writer.start_run(
        name="matching-run",
        phase="training",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=matching_config,
    )
    writer.finish_run(
        run_id=matching_run,
        status="completed",
        summary={"best_val_rmse": 0.30},
    )
    writer.record_decision(
        run_id=matching_run,
        iteration=0,
        proposal_source="static",
        description="baseline",
        status="baseline",
        metric_key="val_rmse",
        metric_value=0.30,
        overrides=[],
    )

    better_but_mismatched = writer.start_run(
        name="mismatched-better",
        phase="training",
        dataset="synthetic",
        model="conv1d",
        device="cpu",
        config=mismatched_config,
    )
    writer.finish_run(
        run_id=better_but_mismatched,
        status="completed",
        summary={"best_val_rmse": 0.10},
    )
    writer.record_decision(
        run_id=better_but_mismatched,
        iteration=1,
        proposal_source="hermes",
        description="better but mismatched",
        status="keep",
        metric_key="val_rmse",
        metric_value=0.10,
        overrides=["data.window_size=32"],
    )

    incumbent = queries.find_best_global_incumbent(
        metric_key="val_rmse",
        dataset="synthetic",
        direction="minimize",
        reference_config=reference_config,
    )

    assert incumbent is not None
    assert incumbent["run_id"] == matching_run
    assert float(incumbent["metric_value"]) == 0.30
    matching_regime = queries.get_run_regime_fingerprint(matching_run)
    assert matching_regime is not None
    identity = queries.get_run_identity(matching_run)
    assert identity is not None
    assert identity["regime_fingerprint"] == matching_regime
    leaderboard = queries.list_leaderboard(limit=10, regime_fingerprint=matching_regime)
    leaderboard_run_ids = {str(row["run_id"]) for row in leaderboard}
    assert matching_run in leaderboard_run_ids
    assert better_but_mismatched not in leaderboard_run_ids


def test_mission_control_summary_filters_leaderboard_to_active_regime(tmp_path: Path) -> None:
    """Active-loop views should only compare runs from the compatible regime."""
    reference_config = ExperimentConfig(
        observability=ObservabilityConfig(
            enabled=True,
            db_path=str(tmp_path / "observability" / "mission_control.db"),
            blob_dir=str(tmp_path / "observability" / "blobs"),
        ),
    )
    writer = ObservabilityWriter.from_experiment_config(reference_config)
    queries = MissionControlQueries(
        db_path=tmp_path / "observability" / "mission_control.db",
        blob_dir=tmp_path / "observability" / "blobs",
    )
    controller = LoopController.from_experiment_config(reference_config, writer=writer)

    matching_run = writer.start_run(
        name="matching-run",
        phase="training",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=reference_config,
    )
    writer.finish_run(run_id=matching_run, status="completed", summary={"best_val_rmse": 0.3})
    writer.record_decision(
        run_id=matching_run,
        iteration=0,
        proposal_source="static",
        description="baseline",
        status="baseline",
        metric_key="val_rmse",
        metric_value=0.3,
        overrides=[],
    )

    mismatched_config = ExperimentConfig(
        data=DataConfig(
            dataset="synthetic",
            window_size=32,
            stride=16,
            normalize=True,
            augment=False,
        ),
        observability=reference_config.observability,
    )
    mismatched_run = writer.start_run(
        name="mismatched-run",
        phase="training",
        dataset="synthetic",
        model="conv1d",
        device="cpu",
        config=mismatched_config,
    )
    writer.finish_run(run_id=mismatched_run, status="completed", summary={"best_val_rmse": 0.1})
    writer.record_decision(
        run_id=mismatched_run,
        iteration=1,
        proposal_source="hermes",
        description="mismatched better",
        status="keep",
        metric_key="val_rmse",
        metric_value=0.1,
        overrides=["data.window_size=32"],
    )

    loop_run_id = writer.start_run(
        name="active-loop",
        phase="autoresearch_loop",
        dataset="synthetic",
        model="lstm",
        device="cpu",
        config=reference_config,
    )
    controller.initialize_loop(
        loop_run_id=loop_run_id,
        max_iterations=4,
        batch_size=None,
        pause_enabled=False,
        current_iteration=1,
        best_metric=0.3,
        best_run_id=matching_run,
    )

    summary = queries.get_mission_control_summary(limit=10)

    assert summary["regime_fingerprint"] == queries.get_run_regime_fingerprint(loop_run_id)
    leaderboard_run_ids = {str(row["run_id"]) for row in summary["leaderboard"]}
    assert matching_run in leaderboard_run_ids
    assert mismatched_run not in leaderboard_run_ids


def test_autoresearch_global_baseline_reuses_previous_best_incumbent(tmp_path: Path) -> None:
    """Global baseline mode should reuse the best compatible prior incumbent."""
    results_path = tmp_path / "results.tsv"
    common_overrides = [
        f"autoresearch.results_file={results_path}",
        "autoresearch.orchestrator=none",
        f"observability.db_path={tmp_path / 'observability' / 'mission_control.db'}",
        f"observability.blob_dir={tmp_path / 'observability' / 'blobs'}",
        "training.epochs=1",
        "training.batch_size=4",
        "data.dataset_kwargs.duration_sec=2.0",
        "data.dataset_kwargs.rate_hz=20.0",
        "data.dataset_kwargs.num_sequences=3",
        "data.window_size=20",
        "data.stride=10",
    ]
    config_paths = [
        "configs/base.yaml",
        "configs/device.yaml",
        "configs/models/lstm.yaml",
        "configs/training/quick.yaml",
        "configs/autoresearch.yaml",
    ]

    first_results = run_autoresearch(
        config_paths=config_paths,
        base_overrides=[*common_overrides, "autoresearch.max_iterations=1"],
        max_iterations=1,
    )
    assert len(first_results) == 2
    assert first_results[0].status == "baseline"

    second_results = run_autoresearch(
        config_paths=config_paths,
        base_overrides=[
            *common_overrides,
            "autoresearch.max_iterations=1",
            "autoresearch.baseline.mode=global",
        ],
        max_iterations=1,
    )
    assert len(second_results) == 1
    assert second_results[0].status in {"keep", "discard"}
    queries = MissionControlQueries(
        db_path=tmp_path / "observability" / "mission_control.db",
        blob_dir=tmp_path / "observability" / "blobs",
    )
    incumbent = queries.find_best_global_incumbent(
        metric_key="val_rmse",
        dataset="synthetic",
        direction="minimize",
        reference_config=resolve_config(config_paths, common_overrides),
    )
    assert incumbent is not None
    assert incumbent["decision_status"] in {"baseline", "keep", "completed"}


def test_autoresearch_manual_baseline_uses_selected_run(tmp_path: Path) -> None:
    """Manual baseline mode should compare against the explicitly selected baseline run."""
    results_path = tmp_path / "results.tsv"
    config_paths = [
        "configs/base.yaml",
        "configs/device.yaml",
        "configs/models/lstm.yaml",
        "configs/training/quick.yaml",
        "configs/autoresearch.yaml",
    ]
    common_overrides = [
        f"autoresearch.results_file={results_path}",
        "autoresearch.orchestrator=none",
        f"observability.db_path={tmp_path / 'observability' / 'mission_control.db'}",
        f"observability.blob_dir={tmp_path / 'observability' / 'blobs'}",
        "training.epochs=1",
        "training.batch_size=4",
        "data.dataset_kwargs.duration_sec=2.0",
        "data.dataset_kwargs.rate_hz=20.0",
        "data.dataset_kwargs.num_sequences=3",
        "data.window_size=20",
        "data.stride=10",
        "autoresearch.max_iterations=1",
    ]

    run_autoresearch(
        config_paths=config_paths,
        base_overrides=common_overrides,
        max_iterations=1,
    )

    queries = MissionControlQueries(
        db_path=tmp_path / "observability" / "mission_control.db",
        blob_dir=tmp_path / "observability" / "blobs",
    )
    baseline_run = next(
        row for row in queries.list_recent_decisions(limit=10) if row["status"] == "baseline"
    )

    manual_results = run_autoresearch(
        config_paths=config_paths,
        base_overrides=[
            *common_overrides,
            "autoresearch.baseline.mode=manual",
            f"autoresearch.baseline.run_id={baseline_run['run_id']}",
        ],
        max_iterations=1,
    )
    assert len(manual_results) == 1
    assert manual_results[0].status in {"keep", "discard"}
