# CLAUDE.md — IMU Denoising Auto-Research

## Project Overview
Autonomous research pipeline for IMU (Inertial Measurement Unit) signal denoising.
Combines deep learning models with classical signal processing baselines, evaluated
on EuRoC Machine Hall and MIT Blackbird datasets.

## Repository Layout
- `src/imu_denoise/` — Main Python package (installable via `pip install -e .`)
- `configs/` — All YAML configuration files (data, model, training, device)
- `scripts/` — CLI entry points: train, evaluate, download data, visualize
- `autoresearch_loop/` — Auto-research integration (edit-run-eval loop + ResearchClaw)
- `tests/` — Unit, integration, and smoke tests
- `notebooks/` — Jupyter notebooks for exploration and analysis
- `vendor/` — External tools (AutoResearchClaw, hermes-agent, llama.cpp, datasets)
- `data/` — Downloaded/processed data (gitignored)
- `artifacts/` — Checkpoints, logs, figures, papers (gitignored)

## Setup
```bash
uv sync                              # Install dependencies
uv run scripts/download_data.py      # Download EuRoC + Blackbird
uv run scripts/preprocess_data.py    # Preprocess to numpy arrays
```

## Key Commands
```bash
uv run scripts/train.py --config configs/models/lstm.yaml       # Train a model
uv run scripts/evaluate.py --checkpoint artifacts/checkpoints/best.pt  # Evaluate
uv run -m pytest                                                  # Run tests
uv run -m pytest -m slow                                          # Include slow tests
```

## Device Support
Automatic detection: CUDA > MPS > CPU. Override via `configs/device.yaml` or `--set device.preferred=cpu`.
MPS (Apple Silicon) works but does not support bfloat16 or torch.compile.

## Architecture Conventions
- All models subclass `BaseDenoiser` and are registered via `@register_model("name")`
- Input/output shape: `(batch, seq_len, 6)` — 3 accelerometer + 3 gyroscope channels
- Configs are YAML, loaded via `imu_denoise.config.loader.load_config()`
- Dataset splits are by sequence (not random) for temporal integrity
- Seeds are set via `imu_denoise.training.reproducibility.seed_everything()`

## Auto-Research Mode
Two modes:
1. **Fast loop** (`autoresearch_loop/loop.py`): Agent modifies config/code, runs time-budgeted training, logs to TSV
2. **Full pipeline** (`researchclaw run --config autoresearch_loop/researchclaw_config.yaml`): 23-stage pipeline producing a paper

## Testing
- Unit tests: `tests/unit/` — fast, no data required, synthetic fixtures
- Integration tests: `tests/integration/` — requires `data/processed/`, marked `@pytest.mark.slow`
- Smoke tests: `tests/smoke/` — full pipeline on synthetic data

## Style
- Python 3.11+, type hints everywhere
- Ruff for linting/formatting (line length 100)
- mypy strict mode
- Dataclasses for config (no Pydantic)
- No wildcard imports
