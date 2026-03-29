# Mission Control Research Policy Plan

This roadmap captures the next phases for making the autoresearch loop more
traceable, iterative, and scientifically disciplined.

## Phase 1: Structured Experiment Lineage

Status: completed

Goals:
- Record how each run was selected.
- Record what changed from the relevant prior run.
- Persist parent/incumbent references and per-run manifests.

Implementation:
- Add first-class `change_sets` and `selection_events` to Mission Control.
- Every run should record:
  - `run_id`
  - `loop_run_id`
  - `parent_run_id`
  - `incumbent_run_id`
  - `proposal_source`
  - `description`
  - `selection rationale`
  - `config diff vs parent`
  - resolved config snapshot via experiment config
- Store this in SQLite and in the run manifest under `artifacts/runs/<run>/run.json`.

Acceptance:
- Any run can answer “what changed?”
- Any run can answer “what incumbent was it compared against?”
- Any run can answer “why was it selected?”

## Phase 2: Mutation Memory

Status: completed

Goals:
- Learn from which mutation families help or hurt.
- Reuse historical evidence instead of re-trying blind changes.

Implementation:
- Add local memory tables:
  - `mutation_signatures`
  - `mutation_attempts`
  - `mutation_stats`
  - `lessons`
- Track:
  - times tried
  - keep/discard/crash counts
  - average delta on objective
  - last regime
  - confidence score

Acceptance:
- The loop can query which changes helped before in the current regime.
- Mission Control can display a mutation leaderboard and recent lessons.

## Phase 3: Apples-to-Apples Regime Matching

Status: completed

Goals:
- Ensure incumbent reuse, memory reuse, and comparisons only happen across
  compatible evaluation regimes.

Implementation:
- Use a stable regime fingerprint derived from:
  - dataset
  - explicit splits / sequences
  - subset config
  - window size
  - stride
  - normalization
  - dataset kwargs
- Use it for:
  - incumbent selection
  - mutation memory retrieval
  - filtered comparisons

Acceptance:
- No global incumbent or mutation memory is reused across incompatible data
  regimes.

## Phase 4: Adaptive Explore / Exploit Policy

Status: completed

Goals:
- Balance exploitation of known-good mutation families with exploration of novel
  directions.

Implementation:
- Add policy config such as:
  - `explore_probability`
  - `stagnation_patience`
  - `exploit_top_k`
  - `novelty_bonus`
  - `max_retries_per_signature`
- Hermes proposes candidates, local policy ranks them using mutation memory and
  current loop context.

Acceptance:
- Repeatedly bad mutation families get deprioritized.
- Good mutation families can be intentionally revisited.
- Exploration increases when improvement stalls.

## Phase 5: Mission Control UX for Research Policy

Goals:
- Make lineage and policy decisions visible in the dashboard and monitor.

Implementation:
- Add views for:
  - lineage graph
  - change diff
  - why-selected panel
  - mutation leaderboard
  - explore / exploit state
  - lesson history

Acceptance:
- A human can understand why the loop selected a run without reading raw logs.

## Phase 6: Reproducibility and Versioning

Goals:
- Make every kept run reproducible from its artifacts and metadata.

Implementation:
- Expand per-run manifests with:
  - resolved config
  - change set
  - selection context
  - incumbent snapshot
  - policy snapshot
  - git sha when code mutation is added

Acceptance:
- Any kept run is reproducible from its manifest plus artifacts.

## Reuse Strategy

What to borrow from `vendor/AutoResearchClaw`:
- experiment memory concepts
- confidence updates
- time-weighted retrieval
- lesson extraction and prompt overlay ideas

What not to directly reuse:
- the full ResearchClaw stage runner
- paper-pipeline abstractions
- the minimal `vendor/autoresearch` keep/discard loop as the long-term policy
  model

## Recommended Order

1. Structured experiment lineage
2. Mutation memory
3. Adaptive explore / exploit policy
4. Mission Control lineage / policy UX
