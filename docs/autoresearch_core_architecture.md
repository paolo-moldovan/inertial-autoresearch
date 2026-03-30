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
- `autoresearch_core.policy`
- `autoresearch_core.engine`
- `autoresearch_core.providers.hermes`
- `autoresearch_core.observability.analytics`
- `autoresearch_core.observability.facade`
- `autoresearch_core.observability.read_models`
- IMU-domain mutation catalog under `imu_denoise.autoresearch`
- IMU-domain execution/config helpers under `imu_denoise.autoresearch.execution`
- IMU-domain proposal selection under `imu_denoise.autoresearch.selection`
- IMU adapter-backed loop runtime now resolves base configs, iteration configs, mutation catalogs, and run execution through `IMUProjectAdapter`
- compatibility wrappers for `autoresearch_loop/{loop,hermes,mutations}.py`
- trainer hook/control boundary
- Mission Control UI bootstrap via a service layer instead of direct store construction
- Mission Control web/TUI/Streamlit surfaces now read/control through a reusable facade layer
- loop-control CLI now composes through Mission Control services instead of direct controller/writer setup
- Mission Control summary payload assembly now delegates to reusable read-model helpers

Still to deepen:
- move more of the loop coordinator into `autoresearch_core.engine`
- migrate larger observability services out of `imu_denoise.observability`
- reduce the size of `imu_denoise.autoresearch.runtime` further
- complete import-boundary enforcement for all CLI and UI modules

## Principles

- reusable core is policy/orchestration/platform logic
- IMU package owns domain configs, models, datasets, evaluation semantics, and mutation catalog
- wrappers remain until tests prove parity
- schema churn is avoided unless it removes an actual boundary violation
