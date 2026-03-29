"""Tests for the local autoresearch loop."""

from __future__ import annotations

from pathlib import Path
from random import Random

from autoresearch_loop.loop import run_autoresearch
from autoresearch_loop.mutations import build_mutation_schedule


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
    assert results_path.exists()
    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
