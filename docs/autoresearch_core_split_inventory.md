# Autoresearch Core Split Inventory

This document is the working inventory for the remaining modularization passes.
It answers two questions:

1. which modules still carry mixed responsibilities
2. which responsibilities should end up in `autoresearch_core` versus the IMU domain

The goal state is:

- `src/autoresearch_core/`: reusable orchestration, providers, policy, Mission Control store/control/query/writer/read-model/analytics logic
- `src/imu_denoise/`: domain runtime only
- `autoresearch_loop/`: compatibility wrappers only

## Current Hotspots

### IMU autoresearch runtime

1. `src/imu_denoise/autoresearch/coordinator.py`
- now slimmer and delegates the generic schedule skeleton to `autoresearch_core.engine`
- mixed concerns:
  - adapter-backed execution callbacks
  - adapter-backed training execution
  - IMU-specific outcome handling
- target:
  - keep moving generic iteration orchestration into `autoresearch_core.engine`
  - IMU layer keeps only adapter-backed execution and domain-specific result semantics

2. `src/imu_denoise/autoresearch/iteration.py`
- completed as a compatibility facade
- the previous mixed responsibilities are now split into:
  - `src/imu_denoise/autoresearch/selection_state.py`
  - `src/imu_denoise/autoresearch/run_preparation.py`
- remaining work:
  - continue moving generic provider/policy-state assembly into `autoresearch_core`
  - keep IMU-specific config resolution and mutation-catalog application local

3. `src/imu_denoise/autoresearch/outcomes.py`
- completed as a compatibility facade
- the previous mixed responsibilities are now split into:
  - `src/imu_denoise/autoresearch/result_recording.py`
  - `src/imu_denoise/autoresearch/result_persistence.py`
- remaining work:
  - generalize only truly reusable decision/result-recording abstractions into core
  - leave mutation-memory and IMU manifest semantics in the domain layer

### Mission Control / observability

4. `src/imu_denoise/observability/queries.py`
- now much thinner and mostly a domain extension/delegation layer
- the previous mixed responsibilities are now split into:
  - `src/imu_denoise/observability/regime_queries.py`
  - `src/imu_denoise/observability/mutation_queries.py`
  - `src/imu_denoise/observability/summary_queries.py`
- remaining work:
  - keep shrinking helper logic that is not actually IMU-specific
  - treat `queries.py` as the IMU read-model extension layer, not the default home for new Mission Control queries

5. `src/imu_denoise/observability/writer.py`
- now much thinner and mostly a domain-facing façade
- the previous mixed responsibilities are now split into:
  - `src/imu_denoise/observability/experiment_tracking.py`
  - `src/imu_denoise/observability/mutation_memory.py`
- remaining work:
  - determine whether mutation-outcome aggregation can be generalized enough for core
  - leave regime/mutation semantics in IMU if they remain domain-shaped

6. `src/imu_denoise/observability/hermes_import.py`
- completed
- generic Hermes-session import now lives in
  `src/autoresearch_core/observability/hermes_import.py`
- the IMU module is now only a thin compatibility wrapper

7. `src/imu_denoise/observability/backfill.py`
- partially extracted
- generic manifest/path/run-reference helpers now live in
  `src/autoresearch_core/observability/backfill.py`
- IMU-specific pieces still local:
  - artifact layout discovery under `artifacts/`
  - training-history row assumptions
  - TSV results replay

8. `src/imu_denoise/observability/web_dashboard.py`
9. `src/imu_denoise/observability/monitor_app.py`
- intentionally remain in the app/domain layer
- but they still need internal modularization:
  - routing / request handlers
  - rendering helpers
  - UI-specific formatting
- these are not “core extraction” problems, they are UI maintainability problems

### Training boundary

10. `src/imu_denoise/observability/training_hooks.py`
- already decouples trainer from store/controller internals
- remaining split:
  - generic writer-backed hook forwarding can live in `autoresearch_core.training`
  - loop-aware termination semantics may move into core once the interrupt contract is abstracted

## Ownership Rules

### Belongs in `autoresearch_core`
- loop progression state
- pause/stop/terminate interpretation
- provider selection orchestration
- generic Mission Control store, controller, queries, writer mechanics
- read-model assembly that does not rely on IMU semantics
- loop and multi-loop analytics
- Hermes session import and other provider/session importers

### Belongs in `imu_denoise`
- config dataclasses and config normalization specific to this project
- data regime fingerprint semantics
- causality semantics for this model family
- mutation catalog and mutation signatures
- trainer/evaluator execution
- artifact naming/layout specific to IMU runs
- IMU-specific leaderboard/incumbent rules and mutation-memory policy extensions

## Next Extraction Order

1. deepen `autoresearch_core.engine`
- absorb more of the generic iteration coordinator
- leave adapter-backed config resolution and experiment execution in IMU

2. split UI modules internally
- `web_dashboard.py`: routing vs rendering
- `monitor_app.py`: layout/state vs formatting helpers

3. revisit mutation writer/query semantics
- only generalize pieces that genuinely transfer to non-IMU projects
- avoid forcing regime-specific behavior into core

## Non-goals

- splitting the repo into multiple packages right now
- changing the SQLite schema just to support the refactor
- moving genuinely domain-specific mutation semantics into the reusable core
