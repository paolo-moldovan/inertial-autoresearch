"""Tests for the local autoresearch loop."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from random import Random
from types import SimpleNamespace
from typing import Any

import pytest
from _pytest.monkeypatch import MonkeyPatch

from autoresearch_loop.hermes import (
    HermesProposalError,
    HermesQueryTrace,
    _run_hermes_query,
)
from autoresearch_loop.hermes import (
    _build_prompt as build_hermes_prompt,
)
from autoresearch_loop.loop import (
    _resolve_iteration_config,
    _select_mutation_proposal,
    build_parser,
    run_autoresearch,
)
from autoresearch_loop.mutations import (
    MutationPolicyCandidate,
    MutationProposal,
    build_mutation_schedule,
    choose_policy_candidate,
    default_mutation_pool,
    filter_mutation_proposals,
)
from imu_denoise.cli.common import resolve_config
from imu_denoise.config import DataConfig, ExperimentConfig, HermesConfig, ObservabilityConfig
from imu_denoise.config.schema import AutoResearchSearchSpaceConfig, AutoResearchStrategyConfig
from imu_denoise.observability import LoopController, MissionControlQueries
from imu_denoise.observability.lineage import build_change_items
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


def test_policy_deprioritizes_bad_mutation_history() -> None:
    """Exploit mode should not keep selecting a mutation family with bad prior outcomes."""
    decision = choose_policy_candidate(
        candidates=[
            MutationPolicyCandidate(
                proposal=MutationProposal(
                    description="switch to huber loss",
                    overrides=["training.loss=huber"],
                ),
                signatures=["training.loss:mse->huber"],
                stats=[
                    {
                        "signature": "training.loss:mse->huber",
                        "tries": 3,
                        "keep_count": 0,
                        "discard_count": 2,
                        "crash_count": 1,
                        "avg_metric_delta": -0.05,
                        "confidence": 0.1,
                    }
                ],
                hermes_preferred=True,
            ),
            MutationPolicyCandidate(
                proposal=MutationProposal(
                    description="lower learning rate",
                    overrides=["training.lr=0.0003"],
                ),
                signatures=["training.lr:0.001->0.0003"],
                stats=[],
            ),
        ],
        strategy=AutoResearchStrategyConfig(mode="exploit", max_retries_per_signature=1),
        recent_results=[],
        rng=Random(0),
    )

    assert decision.mode == "exploit"
    assert decision.selected.description == "lower learning rate"


def test_policy_explores_when_recent_results_stagnate() -> None:
    """Adaptive mode should switch to exploration after repeated non-improving iterations."""
    decision = choose_policy_candidate(
        candidates=[
            MutationPolicyCandidate(
                proposal=MutationProposal(
                    description="repeat known good tweak",
                    overrides=["training.lr=0.0003"],
                ),
                signatures=["training.lr:0.001->0.0003"],
                stats=[
                    {
                        "signature": "training.lr:0.001->0.0003",
                        "tries": 2,
                        "keep_count": 1,
                        "discard_count": 1,
                        "crash_count": 0,
                        "avg_metric_delta": 0.02,
                        "confidence": 0.7,
                    }
                ],
            ),
            MutationPolicyCandidate(
                proposal=MutationProposal(
                    description="small transformer",
                    overrides=["model.name=transformer", "model.hidden_dim=64"],
                ),
                signatures=["model.name:lstm->transformer"],
                stats=[],
            ),
        ],
        strategy=AutoResearchStrategyConfig(
            mode="adaptive",
            explore_probability=0.0,
            stagnation_patience=3,
            stagnation_explore_boost=1.0,
            exploit_top_k=1,
        ),
        recent_results=[
            {"status": "discard"},
            {"status": "discard"},
            {"status": "discard"},
        ],
        rng=Random(0),
    )

    assert decision.stagnating is True
    assert decision.mode == "explore"
    assert decision.selected.description == "small transformer"


def test_iteration_config_inherits_from_incumbent_instead_of_resetting_to_base() -> None:
    """Model-agnostic mutations should keep the incumbent architecture unless overridden."""
    base_config = resolve_config(
        ["configs/base.yaml", "configs/device.yaml", "configs/models/lstm.yaml"],
        [],
    )
    incumbent_config = {
        **asdict(base_config),
        "model": {
            **asdict(base_config.model),
            "name": "conv1d",
            "hidden_dim": 64,
            "num_layers": 4,
        },
    }

    resolved = _resolve_iteration_config(
        base_config=base_config,
        base_overrides=[],
        proposal_overrides=["training.lr=0.0003"],
        incumbent_config=incumbent_config,
    )

    assert resolved.model.name == "conv1d"
    assert resolved.training.lr == 0.0003


def test_search_space_can_freeze_architecture_for_hermes_candidates() -> None:
    """Architecture-fixed search space should remove model-family mutations before Hermes."""
    allowed, blocked = filter_mutation_proposals(
        default_mutation_pool()[1:],
        AutoResearchSearchSpaceConfig(architecture_mode="fixed"),
    )

    allowed_descriptions = {proposal.description for proposal in allowed}
    assert "conv1d baseline" not in allowed_descriptions
    assert "small transformer" not in allowed_descriptions
    assert "deeper lstm" not in allowed_descriptions
    assert blocked["conv1d baseline"] == ["architecture_fixed"]
    assert "lower learning rate" in allowed_descriptions


def test_search_space_exploit_mode_keeps_incumbent_model_family() -> None:
    """Exploit mode should block proposals that switch away from the incumbent family."""
    allowed, blocked = filter_mutation_proposals(
        default_mutation_pool()[1:],
        AutoResearchSearchSpaceConfig(baseline_mode="exploit"),
        incumbent_model_name="conv1d",
    )

    allowed_descriptions = {proposal.description for proposal in allowed}
    assert "lower learning rate" in allowed_descriptions
    assert "small transformer" not in allowed_descriptions
    assert "deeper lstm" not in allowed_descriptions
    assert blocked["small transformer"] == ["exploit_incumbent_model=conv1d"]


def test_default_mutation_pool_is_broader_than_minimal_smoke_space() -> None:
    """The default search pool should cover more than a handful of local tweaks."""
    pool = default_mutation_pool()
    descriptions = {proposal.description for proposal in pool}

    assert len(pool) >= 18
    assert "wider current model" in descriptions
    assert "plateau scheduler" in descriptions
    assert "causal lstm" in descriptions
    assert "medium transformer" in descriptions


def test_realtime_mode_filters_noncausal_candidates_before_hermes() -> None:
    """Realtime mode should keep Hermes on causal candidates only."""
    config = resolve_config(
        ["configs/base.yaml", "configs/device.yaml", "configs/models/lstm.yaml"],
        ["evaluation.realtime_mode=true"],
    )

    candidates, blocked, _preferred_index, _source, _trace = _select_mutation_proposal(
        iteration=1,
        base_config=config,
        base_overrides=[],
        rng=Random(0),
        results=[],
        fallback_proposal=default_mutation_pool()[1],
        hermes_used_descriptions=set(),
        incumbent_summary=None,
        incumbent_config=None,
        mutation_lessons=[],
    )

    descriptions = {candidate.description for candidate in candidates}
    assert "small transformer" not in descriptions
    assert "deeper lstm" not in descriptions
    assert "conv1d baseline" in descriptions
    assert blocked["small transformer"] == ["realtime_requires_causal_model"]


def test_change_items_with_reference_only_tracks_explicit_override_paths() -> None:
    """Mutation memory should not absorb unrelated loop-level config differences."""
    reference = {
        "model": {"name": "conv1d", "hidden_dim": 64},
        "training": {"lr": 0.001, "time_budget_sec": 0},
        "autoresearch": {"hermes": {"pass_session_id": False}},
    }
    current = {
        "model": {"name": "conv1d", "hidden_dim": 64},
        "training": {"lr": 0.0003, "time_budget_sec": 600},
        "autoresearch": {"hermes": {"pass_session_id": True}},
    }

    change_items = build_change_items(
        current_config=current,
        reference_config=reference,
        overrides=["training.lr=0.0003"],
    )

    assert [item["path"] for item in change_items] == ["training.lr"]


def test_hermes_prompt_includes_incumbent_constraints_and_lessons() -> None:
    """Hermes should receive the local policy context instead of choosing semi-blindly."""
    prompt = build_hermes_prompt(
        iteration=3,
        metric_key="val_rmse",
        metric_direction="minimize",
        history=[{"iteration": 2, "status": "keep", "metric_value": 0.16}],
        candidates=[
            MutationProposal(
                description="lower learning rate",
                overrides=["training.lr=0.0003"],
                groups=("training_core", "optimizer"),
            )
        ],
        incumbent={
            "run_id": "run-123",
            "run_name": "autoresearch_004",
            "model": "conv1d",
            "metric_value": 0.167,
        },
        search_space={
            "architecture_mode": "fixed",
            "freeze": ["model.name"],
            "blocked_candidates": {"small transformer": ["architecture_fixed"]},
        },
        mutation_lessons=[
            {
                "signature": "training.lr:0.001->0.0003",
                "lesson_text": "Lower learning rate helped recent conv1d runs.",
            }
        ],
    )

    assert "Current incumbent:" in prompt
    assert '"model":"conv1d"' in prompt
    assert '"architecture_mode":"fixed"' in prompt
    assert "Recent mutation lessons" in prompt
    assert "groups=[\"training_core\",\"optimizer\"]" in prompt


def test_hermes_query_syncs_project_skill_and_passes_skill_flags(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Hermes proposal mode should preload the repo skill and pass native CLI flags."""
    python_bin = tmp_path / "python"
    cli_path = tmp_path / "cli.py"
    python_bin.write_text("", encoding="utf-8")
    cli_path.write_text("", encoding="utf-8")
    hermes_home = tmp_path / ".hermes"

    captured: dict[str, Any] = {}

    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        captured["command"] = list(args[0])
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = dict(kwargs.get("env", {}))
        return SimpleNamespace(returncode=0, stdout='{"candidate_index":0}', stderr="")

    monkeypatch.setattr("autoresearch_loop.hermes.subprocess.run", _fake_run)

    _run_hermes_query(
        prompt="pick one",
        config=HermesConfig(
            python_bin=str(python_bin.relative_to(tmp_path)),
            cli_path=str(cli_path.relative_to(tmp_path)),
            home_dir=str(hermes_home.relative_to(tmp_path)),
            skills=["imu-autoresearch-policy"],
            pass_session_id=True,
        ),
        root=tmp_path,
    )

    command = captured["command"]
    assert "--skills" in command
    assert "imu-autoresearch-policy" in command
    assert "--pass_session_id" in command
    assert "--api_key" in command
    assert "ollama" in command
    synced_skill = hermes_home / "skills" / "imu-autoresearch-policy" / "SKILL.md"
    assert synced_skill.exists()


def test_hermes_query_raises_with_trace_when_cli_prints_retry_failure(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Hermes CLI retry failures should not degrade into opaque parse errors."""
    python_bin = tmp_path / "python"
    cli_path = tmp_path / "cli.py"
    python_bin.write_text("", encoding="utf-8")
    cli_path.write_text("", encoding="utf-8")

    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "API call failed after 3 retries: "
                "HTTP 500: model failed to load, this may be due to resource limitations\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("autoresearch_loop.hermes.subprocess.run", _fake_run)

    with pytest.raises(HermesProposalError) as exc_info:
        _run_hermes_query(
            prompt="pick one",
            config=HermesConfig(
                python_bin=str(python_bin.relative_to(tmp_path)),
                cli_path=str(cli_path.relative_to(tmp_path)),
                home_dir=".hermes",
            ),
            root=tmp_path,
        )

    assert exc_info.value.trace is not None
    assert exc_info.value.trace.reason is not None
    assert "model failed to load" in exc_info.value.trace.reason


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
    assert detail["selection_event"]["policy_state"]["strategy"]["mode"] == "adaptive"
    assert detail["selection_event"]["policy_state"]["policy_mode"] in {"explore", "exploit"}
    assert detail["change_set"] is not None
    assert detail["change_set"]["reference_kind"] in {"base", "incumbent"}
    assert detail["lineage"]["incumbent"] is not None
    assert detail["policy_context"] is not None
    assert isinstance(detail["change_diff"], list)
    assert isinstance(detail["related_lessons"], list)
    assert detail["mutation_attempts"]
    summary = queries.get_mission_control_summary(limit=10)
    assert summary["mutation_leaderboard"]
    assert summary["recent_mutation_lessons"]
    assert summary["current_candidate_pool"] is not None
    assert summary["current_candidate_pool"]["candidates"]
    assert isinstance(summary["current_candidate_pool"]["blocked_candidates"], dict)
    assert summary["current_candidate_pool"]["run_name"] == summary["current_run"]["run_name"]


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
