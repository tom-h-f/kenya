# Embedding coverage lag

Measured 2026-10-10: the v2 daily's text trace sees 12% to 24% of post rows
(14% for the 1 d window). The corpus was 100% embedded on 2026-09-12 by a
one-off mike-pc backfill; the live path has not kept up since.

## Plan

1. Measure, per collection day for the last 30 days, text-eligible posts vs
   posts with a vector (R2 reads, windowed on `dt`, projected; eligibility is
   `coord2.text_rows`). Measure the tf1 embed pass: cycle time, posts per
   cycle, CPU, memory.
2. Pick the steady-state path by measurement and cost:
   - a. raise tf1's per-cycle limit
   - b. embed the day's new posts on Modal GPU inside the relevance job
   - c. embed on `cloud` CPU just before the daily v2 run
3. Backfill the gap on mike-pc with the 2026-09-12 scripts, validated on a
   slice first (cosine 1.000000 against the production pass).
4. Deploy the chosen path only after tests pass and the Modal budget holds.
5. Docs; re-measure the 30 d window peak RSS at the new coverage.

Filled in below as the work lands.
