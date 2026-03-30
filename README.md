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

Processed datasets now carry ground-truth diagnostics in their metadata, and preprocessing registers those diagnostics in Mission Control as dataset artifacts.

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

Request a graceful stop after the current iteration, or force-terminate the active run:

```bash
uv run imu stop
uv run imu stop --terminate
```

Queue a rerun of a previous run for the active loop:

```bash
uv run imu rerun --run-id <run-id-or-prefix>
```

Control how loop baselines are chosen:

```yaml
autoresearch:
  baseline:
    mode: per_loop   # per_loop | global | manual
    run_id: ""       # required when mode=manual
```

- `per_loop`: run a fresh baseline at iteration `0`
- `global`: reuse the best completed compatible incumbent for the same apples-to-apples data regime
- `manual`: pin the baseline to a specific prior run id or id prefix

## Evaluation and Validity

Evaluation behavior is configured explicitly under `evaluation:`:

```yaml
evaluation:
  frequency_epochs: 1
  metrics:
    - rmse
    - mae
    - spectral_divergence
  reconstruction: none   # none | hann
  realtime_mode: false
```

Notes:
- `frequency_epochs` controls how often the full evaluator runs during training
- `metrics` selects which metrics are computed
- `reconstruction: hann` enables overlap-add sequence reconstruction for:
  - `sequence_rmse`
  - `sequence_mae`
  - `sequence_spectral_divergence`
  - `smoothness`
  - `drift_error`
- `realtime_mode: true` warns on non-causal models and constrains autoresearch toward causal candidates

The default search objective remains `val_rmse`. Sequence-level and temporal metrics are opt-in until you explicitly choose them.

Loss weighting is configurable:

```yaml
training:
  loss: mse
  channel_loss_weights: []   # optional explicit length-6 weights
  accel_loss_weight: 1.0
  gyro_loss_weight: 1.0
```

If `channel_loss_weights` is set, it takes precedence over the accel/gyro convenience weights.

The validated local path today is the config-first loop above. The `researchclaw` config scaffold is present, but the actively exercised repo path is [autoresearch_loop/loop.py](/Users/paolo/development/inertial-autoresearch/autoresearch_loop/loop.py).

## Mission Control

One command to run the dashboard:

```bash
uv run --extra monitor imu dashboard
```

That starts the local Mission Control web dashboard on `http://127.0.0.1:8501` by default.
It is no longer Streamlit-backed, so the page updates in place from JSON polling instead of rerunning and losing UI state.

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
- control
- backfill
- monitor
- dashboard
- autoresearch

The `control` pane is the session supervisor. In that pane you can:
- type `1`, `2`, `3`, `4` to jump between panes
- type `list` to show pane mappings
- type `exit` or press `Ctrl-C` to kill the entire Mission Control session immediately

From any pane in the grid, you can also use tmux shortcuts:
- `Ctrl-b 0` to jump back to the supervisor
- `Ctrl-b 1` for backfill
- `Ctrl-b 2` for the monitor
- `Ctrl-b 3` for the dashboard
- `Ctrl-b 4` for the autoresearch loop
- `Ctrl-b X` to kill the full Mission Control session
- `Ctrl-b q` to show tmux's pane-number overlay

So you can watch everything at once instead of switching between tmux windows.

Start the EuRoC Mission Control profile:

```bash
uv run scripts/start_mission_control.py --profile euroc
```

Run a sequence-aware EuRoC autoresearch profile that optimizes reconstructed
`sequence_rmse` instead of plain window `val_rmse`:

```bash
uv run imu loop --config configs/mission_control/hermes_euroc_temporal.yaml
```

That uses [configs/mission_control/hermes_euroc_subset.yaml](/Users/paolo/development/inertial-autoresearch/configs/mission_control/hermes_euroc_subset.yaml) and is intended for a real-data subset run on the currently processed EuRoC Machine Hall sequences.

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
