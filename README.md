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

Preferred unified CLI:

```bash
uv run imu run --config configs/models/lstm.yaml
```

Quick synthetic smoke run:

```bash
uv run imu run --config configs/training/quick.yaml
```

Evaluate a checkpoint:

```bash
uv run imu eval \
  --config configs/training/quick.yaml
```

If you omit `--checkpoint`, `imu eval` now looks up the latest matching completed training run in Mission Control and uses its `best` checkpoint automatically.

Run a classical baseline:

```bash
uv run imu baseline --config configs/training/quick.yaml --baseline kalman
```

Preprocess data:

```bash
uv run imu-preprocess
```

Legacy compatibility entrypoints still work:
- `imu-train`
- `imu-eval`
- `imu-baseline`
- `imu-monitor`
- `imu-dashboard`

## AutoResearch

Local Hermes + Ollama smoke run:

```bash
uv run imu loop --config configs/mission_control/hermes_smoke.yaml
```

Batch/review mode:

```bash
uv run imu loop --config configs/mission_control/hermes_smoke.yaml --batch 5 --pause
```

Queue a human proposal for the active loop:

```bash
uv run imu queue "try transformer with huber loss" \
  --set model.name=transformer \
  --set training.loss=huber
```

Resume a paused loop and inspect status:

```bash
uv run imu loop --resume
uv run imu status
```

The validated local path today is the config-first loop above. The `researchclaw` config scaffold is present, but the actively exercised repo path is [autoresearch_loop/loop.py](/Users/paolo/development/inertial-autoresearch/autoresearch_loop/loop.py).

## Mission Control

One command to run the dashboard:

```bash
uv run --extra monitor imu dashboard
```

That starts the Streamlit dashboard on `http://localhost:8501` by default.

One command to run the live Textual monitor:

```bash
uv run --extra monitor imu monitor
```

Backfill existing artifacts and Hermes state into Mission Control:

```bash
uv run --extra monitor imu-observability-backfill
```

Start the full tmuxinator Mission Control session:

```bash
uv run scripts/start_mission_control.py
```

That opens one tmux window in a tiled grid with panes for:
- backfill
- monitor
- dashboard
- autoresearch

So you can watch everything at once instead of switching between tmux windows. If one command exits, that pane drops you into a shell instead of disappearing.

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
- Run outputs now use a per-run layout under `artifacts/runs/<run-name>--<token>/`.
- Typical run contents are `checkpoints/`, `logs/runtime.jsonl`, `logs/history.jsonl`, `figures/`, `metrics.json`, and `run.json`.
- The observability DB remains shared under `artifacts/observability/`; experiment grouping lives there, while filesystem layout is run-centric to avoid overwriting repeated runs.
- Default Mission Control database: `artifacts/observability/mission_control.db`
- Default Mission Control blob store: `artifacts/observability/blobs`
