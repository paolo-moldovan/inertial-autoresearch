# Loop Analytics Backlog

Mission Control should expose loop-level and multi-loop analytics as first-class
debugging and research tools, not just as a leaderboard.

## High-Value Views

1. Objective trajectory by iteration
- raw objective per run
- running best/incumbent trajectory
- time to first improvement
- time to best result

2. Outcome breakdowns
- keep/discard/crash counts
- queue vs Hermes vs static-fallback contribution
- provider failure rates

3. Search-space diagnostics
- mutation group win rates
- blocked-candidate counts by veto reason
- exploration vs exploitation ratio over time
- search-space collapse indicators

4. Model-family analytics
- win rate by model family
- incumbent transitions between model families
- architecture-change vs tune-only outcome rates

5. Cross-loop comparisons
- best objective per compatible regime
- median improvement per loop
- average iterations to improvement
- model-family success by regime

## Intended Surfaces

- Web Mission Control analytics section
- Textual monitor summary panel
- SQLite-backed read models in the reusable core
- Exportable summaries for offline notebooks and papers
