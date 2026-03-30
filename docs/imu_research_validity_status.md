# IMU Research Validity Status

Status snapshot against the 10-point assessment that originally motivated the
evaluation and loop-reliability work.

## Implemented

1. Temporal evaluation metrics
- Added `smoothness` and `drift_error` in [metrics.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/evaluation/metrics.py).
- Added sequence-level reconstruction-aware evaluation in [evaluator.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/evaluation/evaluator.py) and [reconstruction.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/evaluation/reconstruction.py).

2. Sampling rate derived from data
- Implemented timestamp-derived sampling-rate inference in [datamodule.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/data/datamodule.py) and [diagnostics.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/data/diagnostics.py).

3. Channel-weighted loss
- Added `training.channel_loss_weights`, `training.accel_loss_weight`, and `training.gyro_loss_weight` in [schema.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/config/schema.py).
- Applied deterministic channel weighting in [losses.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/training/losses.py).

4. Causality discipline
- Added `BaseDenoiser.causal` in [base.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/models/base.py).
- Realtime evaluation warnings and autoresearch filtering use this metadata in [evaluator.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/evaluation/evaluator.py) and [tests/unit/test_autoresearch.py](/Users/paolo/development/inertial-autoresearch/tests/unit/test_autoresearch.py).

5. Window-boundary blending
- Implemented overlap-add sequence reconstruction with `hann` weighting in [reconstruction.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/evaluation/reconstruction.py).

6. Two-tier mutation memory
- Implemented regime-local evidence plus bounded cross-regime priors in [queries.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/observability/queries.py).

7. Configurable evaluation cadence
- Added `evaluation.frequency_epochs`, `evaluation.metrics`, `evaluation.reconstruction`, and `evaluation.realtime_mode` in [schema.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/config/schema.py).
- Trainer now honors cadence in [trainer.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/training/trainer.py).

9. Ground-truth quality diagnostics
- Added PSD-gap and Allan-variance diagnostics in [diagnostics.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/data/diagnostics.py).
- Preprocessing registers diagnostics artifacts in [preprocess_data.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/cli/preprocess_data.py).

10. Critical observability writes
- Added `_critical(...)` in [writer.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/observability/writer.py).
- Loop-control writes are critical in [control.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/observability/control.py).

## Partial

1. Temporal evaluation breadth
- `smoothness` and `drift_error` are in place.
- Allan variance is currently a preprocessing diagnostic, not yet a first-class evaluator metric for model ranking.

5. Deployment parity
- Reconstruction is implemented for evaluation.
- Full deployment/inference-time stitching is still not a first-class runtime path.

## Not Yet Done

8. Config-surface cleanup
- The original examples in the assessment were partly stale.
- A dedicated static audit/removal pass still has not been done.

## New Follow-Through Added After The Initial Plan

Sequence-aware autoresearch can now use non-RMSE evaluation metrics as the actual loop objective:
- training summaries now expose `best_metric_key`, `best_metric_value`, and `best_eval_metrics` in [trainer.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/training/trainer.py)
- loop ranking now reads those values in [loop.py](/Users/paolo/development/inertial-autoresearch/autoresearch_loop/loop.py)
- Mission Control leaderboard selection for the active loop now respects the loop objective metric in [queries.py](/Users/paolo/development/inertial-autoresearch/src/imu_denoise/observability/queries.py)
- a concrete sequence-aware EuRoC profile now exists at [hermes_euroc_temporal.yaml](/Users/paolo/development/inertial-autoresearch/configs/mission_control/hermes_euroc_temporal.yaml)

## Recommended Next Steps

1. Validate temporal objectives on real EuRoC and Blackbird runs.
- Compare `val_rmse` loops vs `sequence_rmse` loops.
- Check whether improvements survive across larger subsets and full sequences.

2. Add loop-level and multi-loop metastatistics to Mission Control.
- Objective trajectory by iteration
- incumbent changes and time-to-improvement
- keep/discard/crash rates by mutation group
- Hermes vs fallback contribution
- model-family win rates
- per-loop and cross-loop distribution plots of metric deltas

3. Add Allan variance as an optional evaluator metric.

4. Do a true config-surface audit and remove only provably unused fields.
