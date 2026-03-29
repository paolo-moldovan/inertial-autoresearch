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
- model zoo with LSTM, Conv1D, and Transformer baselines
- classical baselines with Kalman and complementary smoothing
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
- `autoresearch_loop/`: future autonomous experiment loop
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
```

## CI And Docker

- CI workflow: [ci.yml](/Users/paolo/development/inertial-autoresearch/.github/workflows/ci.yml)
- Docker image: [Dockerfile](/Users/paolo/development/inertial-autoresearch/docker/Dockerfile)
- Compose stack: [docker-compose.yml](/Users/paolo/development/inertial-autoresearch/docker/docker-compose.yml)

## Design Principles

- One clean package boundary: application code lives under `src/imu_denoise/`
- Device portability first: CUDA > MPS > CPU with explicit overrides
- Reproducibility over magic: typed config, fixed outputs, deterministic tests
- Vendor isolation: local adapters and CLI wrappers instead of direct imports
