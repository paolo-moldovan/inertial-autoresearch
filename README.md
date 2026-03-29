# IMU Denoise

Autonomous research infrastructure for IMU denoising experiments across CUDA,
Apple MPS, and CPU backends.

The repository is organized around a typed Python package in
`src/imu_denoise/`, YAML-based experiment configuration in `configs/`, thin
CLI entrypoints in `scripts/`, and vendorized research tooling in `vendor/`.

## Current Status

The repo now includes:

- typed config loading with hierarchical YAML merge
- multi-device data, training, evaluation, and auto-research loops
- Hermes-driven autoresearch orchestration via local Ollama (`qwen3.5`)
- model zoo with LSTM, Conv1D, and Transformer baselines
- classical baselines with Kalman and complementary smoothing
- mission-control observability with SQLite-backed traces, backfill, TUI, and dashboard
- package-backed CLI entrypoints, CI, and Docker assets
- synthetic quick-run path for fast verification

## Quick Start

```bash
uv sync
uv run -m pytest tests/unit
uv run scripts/train.py --config configs/training/quick.yaml
uv run scripts/evaluate.py --config configs/training/quick.yaml --checkpoint artifacts/checkpoints/default/best.pt
uv run scripts/run_baseline.py --config configs/training/quick.yaml --baseline kalman
uv run autoresearch_loop/loop.py --max-iterations 1
```

## Repository Layout

- `src/imu_denoise/`: installable Python package
- `configs/`: experiment configuration fragments
- `scripts/`: thin wrappers around package CLI modules
- `tests/`: unit, integration, and smoke tests
- `autoresearch_loop/`: local config-first and Hermes-assisted experiment loop
- `vendor/`: external tools and datasets kept isolated from package imports
- `docker/`: container image and compose stack
- `.github/workflows/`: CI automation

## Useful Commands

```bash
# Train a quick synthetic experiment
uv run scripts/train.py --config configs/training/quick.yaml

# Evaluate the resulting checkpoint
uv run scripts/evaluate.py \
  --config configs/training/quick.yaml \
  --checkpoint artifacts/checkpoints/default/best.pt

# Run a classical baseline
uv run scripts/run_baseline.py \
  --config configs/training/quick.yaml \
  --baseline kalman

# Run one baseline + one mutation in the local autoresearch loop
uv run autoresearch_loop/loop.py --max-iterations 1

# Backfill historical logs and import Hermes sessions into mission control
uv run scripts/observability_backfill.py

# Launch the live TUI monitor
uv run scripts/monitor.py

# Launch the Streamlit dashboard
uv run scripts/dashboard.py

# Start the whole stack with tmuxinator
uv run scripts/start_mission_control.py
```

## Hermes + Ollama

`configs/autoresearch.yaml` is set up to use the vendored Hermes CLI with a
local Ollama endpoint at `http://127.0.0.1:11434/v1` and the `qwen3.5:latest`
model. The current integration is config-first: Hermes chooses the next safe
mutation from the bounded candidate pool, while training and evaluation still
run through the local Python stack. If Hermes or Ollama is unavailable, the
loop falls back to the deterministic built-in mutation schedule.

## Mission Control

Mission control stores a repo-local observability database under
`artifacts/observability/` and indexes:

- live run status and training epochs
- autoresearch decisions and mutation history
- raw Hermes/Ollama prompt and response traces
- Hermes session imports from `.hermes/`
- registered artifacts such as checkpoints, metrics, and figures

The read-only surfaces are:

- `imu-monitor` / `scripts/monitor.py`: Textual TUI for live status
- `imu-dashboard` / `scripts/dashboard.py`: Streamlit browser dashboard
- `imu-observability-backfill` / `scripts/observability_backfill.py`: historical import
- `imu-mission-control` / `scripts/start_mission_control.py`: tmuxinator stack launcher

The default stack config for the local Hermes + Ollama smoke path lives at
`configs/mission_control/hermes_smoke.yaml`, and the tmuxinator project file
is at `.tmuxinator/mission-control.yml`.

## CI And Docker

- CI workflow: [ci.yml](./.github/workflows/ci.yml)
- Docker image: [Dockerfile](./docker/Dockerfile)
- Compose stack: [docker-compose.yml](./docker/docker-compose.yml)

## Design Principles

- One clean package boundary: application code lives under `src/imu_denoise/`
- Device portability first: CUDA > MPS > CPU with explicit overrides
- Reproducibility over magic: typed config, fixed outputs, deterministic tests
- Vendor isolation: local adapters and CLI wrappers instead of direct imports
