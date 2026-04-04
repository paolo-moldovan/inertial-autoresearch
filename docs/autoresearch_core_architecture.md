# Autoresearch Core Architecture Roadmap

This document tracks the modularization work that turns the IMU project into a
domain implementation on top of a reusable autoresearch core.

## Target Layering

1. `src/autoresearch_core/`
- reusable contracts
- policy selection
- provider integrations
- observability-facing analytics and façades
- Mission Control read-model assembly helpers
- no IMU imports
- no Torch, dataset, model, or project-specific runtime imports

2. `src/imu_denoise/autoresearch/`
- IMU adapter layer
- IMU mutation catalog
- IMU runtime execution and config resolution
- IMU proposal-selection helpers
- IMU loop lifecycle helpers
- IMU loop artifact/manifest helpers
- IMU iteration-planning helpers
- IMU outcome/result-recording helpers
- IMU loop session/context helpers
- IMU top-level loop coordinator helpers
- bridge between core orchestration and the IMU training/evaluation stack

3. `src/imu_denoise/training/`
- training runtime only
- interacts with mission control through hooks and control adapters
- no direct loop-controller or store access

4. `src/imu_denoise/observability/`
- project composition layer and backward-compatible imports
- UI/service bootstrap
- delegates reusable analytics/facade logic to `autoresearch_core`

5. `autoresearch_loop/`
- temporary compatibility wrappers only
- scheduled for eventual removal after parity is verified

## Current Extraction Status

Implemented:
- `autoresearch_core.contracts`
- `autoresearch_core.training`
- generic writer-backed training hooks and loop-aware training control now live in `autoresearch_core.training`
- `autoresearch_core.policy`
- `autoresearch_core.engine`
- generic loop progress and pause/stop/terminate resolution now live in `autoresearch_core.engine`
- generic loop schedule execution now lives in `autoresearch_core.engine`
- generic run-result history snapshots now live in `autoresearch_core.engine`
- generic run-result construction and candidate-payload helpers now live in `autoresearch_core.engine`
- `autoresearch_core.providers.hermes`
- `autoresearch_core.observability.analytics`
- `autoresearch_core.observability.backfill`
- `autoresearch_core.observability.control`
- `autoresearch_core.observability.facade`
- `autoresearch_core.observability.hermes_import`
- `autoresearch_core.observability.logging`
- `autoresearch_core.observability.queries`
- `autoresearch_core.observability.read_models`
- `autoresearch_core.observability.redaction`
- `autoresearch_core.observability.store`
- `autoresearch_core.observability.writer`
- `autoresearch_core.observability.services`
- generic current-run, policy-context, and Hermes runtime read-model assembly in `autoresearch_core.observability.read_models`
- reusable Mission Control control-plane logic and event constants in `autoresearch_core.observability.control`
- reusable Mission Control log mirroring now lives in `autoresearch_core.observability.logging`
- reusable Mission Control generic query methods now live in `autoresearch_core.observability.queries`
- reusable Hermes session import now lives in `autoresearch_core.observability.hermes_import`
- reusable manifest/path/run-reference backfill helpers now live in `autoresearch_core.observability.backfill`
- reusable run-detail, traceability, current-run, and Hermes-runtime query helpers now live in `autoresearch_core.observability.queries`
- reusable payload redaction now lives in `autoresearch_core.observability.redaction`
- reusable Mission Control SQLite store now lives in `autoresearch_core.observability.store`
- reusable Mission Control writer mechanics now live in `autoresearch_core.observability.writer`
- IMU-domain mutation catalog under `imu_denoise.autoresearch`
- IMU-domain execution/config helpers under `imu_denoise.autoresearch.execution`
- IMU-domain proposal selection under `imu_denoise.autoresearch.selection`
- IMU loop lifecycle helpers under `imu_denoise.autoresearch.lifecycle`
- IMU loop artifact helpers under `imu_denoise.autoresearch.artifacts`
- IMU iteration planning now split into `imu_denoise.autoresearch.selection_state` and `imu_denoise.autoresearch.run_preparation`
- `imu_denoise.autoresearch.iteration` is now a thin compatibility facade over those helpers
- IMU outcome handling now split into `imu_denoise.autoresearch.result_recording` and `imu_denoise.autoresearch.result_persistence`
- `imu_denoise.autoresearch.outcomes` is now a thin compatibility facade over those helpers
- IMU loop session setup under `imu_denoise.autoresearch.session`
- IMU top-level loop coordination under `imu_denoise.autoresearch.coordinator`
- IMU runtime implementation under `imu_denoise.autoresearch.runner`
- IMU loop runtime now uses the shared `RunResult` contract instead of a duplicate local result type
- IMU adapter-backed loop runtime now resolves base configs, iteration configs, mutation catalogs, and run execution through `IMUProjectAdapter`
- `imu_denoise.autoresearch.runtime` is now a compatibility shim that re-exports the public runtime surface
- compatibility wrappers for `autoresearch_loop/{loop,hermes,mutations}.py`
- trainer hook/control boundary
- Mission Control UI bootstrap via a service layer instead of direct store construction
- Mission Control web/TUI/Streamlit surfaces now read/control through a reusable facade layer
- loop-control CLI now composes through Mission Control services instead of direct controller/writer setup
- Mission Control summary payload assembly now delegates to reusable read-model helpers
- Mission Control service-bundle composition now lives in `autoresearch_core`
- Mission Control loop controller now lives in `autoresearch_core`, with `imu_denoise` keeping only a config-aware wrapper
- Mission Control store now lives in `autoresearch_core`, with `imu_denoise` keeping only the regime-fingerprint subclass
- IMU writer responsibilities now split into `imu_denoise.observability.experiment_tracking` and `imu_denoise.observability.mutation_memory`
- `imu_denoise.observability.writer` is now a much thinner domain-facing façade over those helpers
- IMU query responsibilities now split into `imu_denoise.observability.regime_queries`, `mutation_queries`, and `summary_queries`
- `imu_denoise.observability.queries` is now a thin domain extension/delegation layer

Still to deepen:
- move more of the generic loop coordinator into `autoresearch_core.engine`
- migrate Hermes import and any remaining generic observability helpers out of `imu_denoise.observability`
- revisit historical backfill next; path inference is still IMU-specific but the replay/import mechanics are now the main remaining generic ingestion seam
- move more of the implementation currently living in `imu_denoise.autoresearch.runner` into `autoresearch_core.engine`
- complete import-boundary enforcement for all CLI and UI modules
- split large UI modules (`web_dashboard.py`, `monitor_app.py`) internally for maintainability

See also:
- `docs/autoresearch_core_split_inventory.md` for the current hotspot inventory and extraction order

## Principles

- reusable core is policy/orchestration/platform logic
- IMU package owns domain configs, models, datasets, evaluation semantics, and mutation catalog
- wrappers remain until tests prove parity
- schema churn is avoided unless it removes an actual boundary violation
