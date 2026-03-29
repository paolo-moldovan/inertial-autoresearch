# IMU Denoise

Autonomous research infrastructure for IMU denoising experiments across CUDA,
Apple MPS, and CPU backends.

The repository is organized around a typed Python package in
`src/imu_denoise/`, YAML-based experiment configuration in `configs/`, thin
CLI entrypoints in `scripts/`, and vendorized research tooling in `vendor/`.

## Current Status

Phase 1 foundations are in place:

- typed config loading with hierarchical YAML merge
- device detection and runtime context for CUDA, MPS, and CPU
- initial denoiser model registry with LSTM, Conv1D, and Transformer baselines
- evaluation metrics and visualization helpers
- package-backed CLI entrypoints and fast unit tests

The data pipeline, trainer, and auto-research loop are planned next.

## Quick Start

```bash
uv sync
uv run -m pytest tests/unit
uv run scripts/train.py --dry-run
```

## Repository Layout

- `src/imu_denoise/`: installable Python package
- `configs/`: experiment configuration fragments
- `scripts/`: thin wrappers around package CLI modules
- `tests/`: unit, integration, and smoke tests
- `autoresearch_loop/`: future autonomous experiment loop
- `vendor/`: external tools and datasets kept isolated from package imports

## Design Principles

- One clean package boundary: application code lives under `src/imu_denoise/`
- Device portability first: CUDA > MPS > CPU with explicit overrides
- Reproducibility over magic: typed config, fixed outputs, deterministic tests
- Vendor isolation: local adapters and CLI wrappers instead of direct imports
