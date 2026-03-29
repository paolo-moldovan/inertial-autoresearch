# IMU Denoising Auto-Research

This loop is config-first and local by default.

## Scope

- Mutate YAML-style experiment settings, not source code.
- Use the existing training stack in `src/imu_denoise/training/`.
- Optimize the metric configured in `configs/autoresearch.yaml`, currently `val_rmse`.

## Default workflow

1. Run a baseline experiment.
2. Apply one config mutation at a time.
3. Train and evaluate on the current pipeline.
4. Log results to `artifacts/autoresearch/results.tsv`.
5. Mark each trial as `baseline`, `keep`, `discard`, or `crash`.

## Safety

- Prefer synthetic quick runs until the real datasets are fully validated.
- Do not mutate code automatically in the local loop.
- Keep experiment names unique and write artifacts under `artifacts/`.

## Future extensions

- Add code-mutation mode.
- Add vendor orchestrator adapters for Hermes and ResearchClaw.
- Add longer-running real-dataset search schedules.
