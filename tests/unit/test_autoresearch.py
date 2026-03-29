"""Tests for the local autoresearch loop."""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

from _pytest.monkeypatch import MonkeyPatch

from autoresearch_loop.hermes import HermesQueryTrace
from autoresearch_loop.loop import build_parser, run_autoresearch
from autoresearch_loop.mutations import MutationProposal, build_mutation_schedule
from imu_denoise.observability import MissionControlQueries


def test_mutation_schedule_is_baseline_first() -> None:
    """Mutation schedules should always start with the baseline configuration."""
    schedule = build_mutation_schedule(3, Random(7))

    assert schedule[0].description == "baseline"
    assert schedule[0].overrides == []
    assert len(schedule) == 4


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
