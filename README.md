# IMU Denoising Auto-Research

Autonomous research pipeline for IMU denoising with deep models, classical baselines, Hermes + Ollama orchestration, and Mission Control observability.

## Setup

Base dev environment:

```bash
uv sync --extra dev
```

With Mission Control UIs:

```bash
uv sync --extra dev --extra monitor
```

With Phase 2 observability adapters (MLflow + Phoenix):

```bash
uv sync --extra dev --extra monitor --extra monitor-adapters
```

## Core Commands

Train a model:

```bash
uv run scripts/train.py --config configs/models/lstm.yaml
```

Quick synthetic smoke run:

```bash
uv run scripts/train.py --config configs/models/lstm.yaml --config configs/training/quick.yaml
```

Evaluate a checkpoint:

```bash
uv run scripts/evaluate.py \
  --config configs/models/lstm.yaml \
  --config configs/training/quick.yaml \
  --checkpoint artifacts/checkpoints/default/best.pt
```

Run a classical baseline:

```bash
uv run scripts/run_baseline.py --config configs/training/quick.yaml --baseline kalman
```

Preprocess data:

```bash
uv run scripts/preprocess_data.py
```

## AutoResearch

Local Hermes + Ollama smoke run:

```bash
uv run autoresearch_loop/loop.py --config configs/mission_control/hermes_smoke.yaml
```

The validated local path today is the config-first loop above. The `researchclaw` config scaffold is present, but the actively exercised repo path is `autoresearch_loop/loop.py`.

## Mission Control

One command to run the dashboard:

```bash
uv run --extra monitor imu-dashboard
```

That starts the Streamlit dashboard on `http://localhost:8501` by default.

One command to run the live Textual monitor:

```bash
uv run --extra monitor imu-monitor
```

Backfill existing artifacts and Hermes state into Mission Control:

```bash
uv run --extra monitor imu-observability-backfill
```

Start the full tmuxinator Mission Control session:

```bash
uv run scripts/start_mission_control.py
```

That session starts:
- backfill
- monitor
- dashboard
- autoresearch

## Mission Control Phase 2

Enable the external adapter config:

```bash
uv run --extra monitor-adapters imu-observability-sync \
  --config configs/mission_control/adapters.yaml \
  --target all
```

Sync only MLflow:

```bash
uv run --extra monitor-adapters imu-observability-sync \
  --config configs/mission_control/adapters.yaml \
  --target mlflow
```

Sync only Phoenix:

```bash
uv run --extra monitor-adapters imu-observability-sync \
  --config configs/mission_control/adapters.yaml \
  --target phoenix
```

## Testing

Focused unit tests:

```bash
uv run --extra dev python -m pytest tests/unit
```

Lint:

```bash
uv run --extra dev ruff check src tests scripts
```

Type checking:

```bash
uv run --extra dev mypy src
```

## Notes

- Device auto-detection is `CUDA > MPS > CPU`.
- Override device manually with `--set device.preferred=cpu`.
- Default Mission Control database: `artifacts/observability/mission_control.db`
- Default Mission Control blob store: `artifacts/observability/blobs`
