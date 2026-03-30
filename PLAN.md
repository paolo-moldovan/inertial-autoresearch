# IMU Research Validity and Loop-Reliability Plan

## Summary
Claude’s assessment is directionally good, but three of the “dead field” examples are stale in this repo: `data.sequences`, `DataSubsetConfig.*_sequence_fraction`, and `HermesConfig.api_key` are currently used. The strongest gaps are still real: sampling rate is hardcoded, evaluation is window-local and mostly pointwise, causality is implicit, eval cadence is fixed, and observability uses one best-effort path even for writes the loop depends on.

Chosen defaults:
- First wave is **correctness first**
- New temporal metrics are **opt-in**, not the new default objective
- Existing `val_rmse` remains the default search objective until new sequence-level metrics are validated on real runs

## Implementation Changes

### Phase 1: Correctness Foundation
- Add a new `EvaluationConfig` under `ExperimentConfig` and move evaluation-specific behavior there:
  - `frequency_epochs: int = 1`
  - `metrics: list[str]`
  - `reconstruction: str = "none"` initially
  - `realtime_mode: bool = False`
- Replace tuple-only dataloader returns with a small `DataBundle` read model that includes:
  - `train_loader`, `val_loader`, `test_loader`
  - `sampling_rate_hz`
  - split sequence names
  - any reconstruction metadata needed later
- Derive `sampling_rate_hz` from sequence timestamps using median `dt`, aggregated across the active split data, and remove trainer string-matching for EuRoC/Blackbird.
- Add `causal: bool` to `BaseDenoiser`; each model declares it explicitly.
  - `conv1d` sets `True`
  - `lstm` sets `False` when bidirectional, `True` otherwise
  - evaluation warns when `realtime_mode=True` and the selected model is non-causal
- Split observability writes into:
  - decision-critical writes: mutation outcome, incumbent update, loop-state update, queued-proposal claim/apply
  - best-effort writes: heartbeats, log mirrors, UI events
- Decision-critical write failures must bubble up to the loop with retry and explicit failure handling; best-effort writes keep the current warning-only behavior.

### Phase 2: Sequence-Level Evaluation
- Extend evaluation from flattened window metrics to full-sequence reconstruction.
- Implement overlap-add reconstruction for overlapping windows using a fixed weighted blending strategy:
  - default `hann`
  - normalize overlap weights so reconstructed output matches original sequence length exactly
- Keep both scopes available:
  - window-level metrics for backward compatibility
  - sequence-level reconstructed metrics for research validity
- Add first temporal metrics:
  - `smoothness` based on second-difference magnitude
  - `drift_error` from short-horizon forward integration against timestamps and ground-truth trajectory proxy
- Make `Evaluator` compute only the metrics listed in `EvaluationConfig.metrics`.
- Make training honor `EvaluationConfig.frequency_epochs`; skipped epochs should still log train/val loss, but not run the full evaluator.

### Phase 3: Training Signal and Deployment Discipline
- Extend loss building to support weighted reconstruction losses:
  - per-channel weights of length 6
  - convenience sensor-type weights expanded to `[acc, acc, acc, gyro, gyro, gyro]`
- Keep weighting config-driven and deterministic; do not introduce learned uncertainty weighting yet.
- Apply the same causality metadata in search-space and evaluation surfaces:
  - autoresearch can constrain to causal-only candidates
  - dashboard and run detail expose causal vs non-causal status
- Preserve current normalization behavior; loss weighting operates on the normalized training target unless normalization is disabled.

### Phase 4: Search Memory and Dataset Diagnostics
- Add a two-tier mutation-memory policy without changing the current regime-specific tables:
  - regime-specific evidence remains primary
  - cross-regime prior is computed from existing `mutation_attempts` across all regimes with time decay
- The cross-regime prior is advisory only:
  - it contributes a bounded prior score when regime-specific evidence is sparse
  - it must never override strong regime-specific evidence
- Add ground-truth quality diagnostics during preprocessing:
  - PSD gap between noisy and clean
  - Allan variance summary
  - simple flags for suspicious “clean” signals
- Persist GT diagnostics in processed-dataset metadata and register them as observability artifacts so they show up in Mission Control.

### Phase 5: Config and Documentation Audit
- Do a static config-surface audit after the above changes land.
- Remove only fields that are provably unused after the audit.
- Update docs/examples so the public config surface matches actual behavior, especially around:
  - `EvaluationConfig`
  - causal vs non-causal models
  - sequence-level vs window-level metrics
  - opt-in temporal objectives

## Public Interfaces / Types
- Add `EvaluationConfig` to `ExperimentConfig`
- Replace raw `(train_loader, val_loader, test_loader)` returns with a `DataBundle`
- Add `BaseDenoiser.causal: bool`
- Extend loss config to support channel/sensor weighting
- New valid metric names become available for evaluation and autoresearch configs, but `val_rmse` stays the repo default

## Test Plan
- Sampling rate is inferred correctly from timestamps for synthetic 100 Hz, 200 Hz, and non-round rates.
- `EvaluationConfig.frequency_epochs` skips expensive evaluator passes while preserving training history.
- Sequence reconstruction preserves length and blends overlaps without double-counting.
- Window-level and sequence-level metrics can coexist in one run without ambiguity.
- `realtime_mode=True` warns or blocks appropriately for non-causal models.
- Weighted losses produce deterministic expected values for mixed accel/gyro errors.
- Critical observability write failures are surfaced to the loop; non-critical failures remain warning-only.
- Cross-regime mutation prior helps only when regime-local evidence is weak and never overrides stronger local evidence.
- GT diagnostics are generated during preprocess and surfaced as artifacts/metadata.

## Assumptions and Defaults
- No objective switch is made automatically; temporal metrics are added first and selected explicitly via config when desired.
- Overlap-add is evaluation-only in the first pass; training remains window-based.
- Drift metrics use the existing timestamps and available sequence ground truth proxy; no new external trajectory pipeline is introduced in this phase.
- “Dead field” cleanup is deferred until after the new evaluation/config surface is in place, because the current assessment was partly outdated.
