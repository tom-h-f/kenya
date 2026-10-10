# v2 daily on the `cloud` host, off Modal

Decided 2026-10-10: the v2 daily detector runs as a CPU job on `cloud` (OVH,
4 vCPU, 15 GB RAM). Modal keeps only relevance scoring (`kma-relevance`); the
daily never calls it.

## Shape

One module, `kma.v2_daily_run` (`uv run python -m kma.v2_daily_run`), reusing
`v2_daily.orchestrate` so the stages and their timings are the Modal app's:

1. window manifest (`window.ensure`)
2. text trace if missing (`textsim_run.build`, `time_bucket="day"`,
   `coord2.gpu_cosine_pairs`, which falls back to CPU torch)
3. rank + `daily_*` tables (`v2_daily.rank`), optionally with the canary
4. `coord2/kind=scores` with the `community` column (`coord2_run.persist`)
5. freshness gauge, only after 1-4 all succeeded

No `modal` import on this path. Nothing here contacts X.

## Decisions

- **Scores come from the report's listed accounts**, not a second top-N: the
  collector's promotion ranks within community, and the listing is what a
  reader sees. Canary accounts are dropped from `kind=scores` (they are not
  real ids; deep-timelines would try to fetch them) and stay in `daily_*`.
- **Floors.** `windowed.DETECTION` (day, 3) and `windowed.SERIES` (week, 5)
  are the decided values (`2026-09-15-windowed-detection.md`, "Measured and
  decided"; commit 1302f78). They supersede decision 5 of
  `2026-09-22-detection-readiness.md` (5 / 10), which was explicitly pending
  that sweep. They are floors for per-day and per-week windows over the four
  bipartite traces; the sweep never ran the rolling 90-day rank. So:
  - the rolling rank keeps `coord2.MIN_ENTITIES_PER_USER` (2), exposed as
    `--min-entities` and recorded in every run row. Raising it is OPEN: the
    depth test (2026-09-14) asked for a global sweep at 5/10/20 that was never
    run.
  - `--windowed day|week` adds the windowed pass as an extra pass, using
    `windowed.DETECTION` / `windowed.SERIES`, written as
    `kind=daily_windowed_<width>`.
- **Canary** is `--canary`: `canary.load_active` posts go to `rank`. No
  schedule here.
- **Gauge** is `kma_v2_daily.prom` in `$KMA_TEXTFILE_DIR`, tmp file + rename.
- **Measurement mode** `--out-dir DIR`: manifest and text trace stay in
  memory, every table lands under DIR, nothing is written to R2. Tests and the
  window-size measurements use it.

## Deployment artifacts (not deployed)

`deploy/cloud/`: systemd service + timer (03:30 UTC) and a README: checkout,
uv, env file contract (names only), logs, `MemoryMax`.

## Measured

2026-10-10, window ending 2026-10-09, on a Mac (CPU torch, home link to R2,
`--out-dir`, nothing written to R2). The 4 vCPU `cloud` box is not measured.

| window | post rows | wall | peak RSS | vectors (coverage) | text edges | accounts in graph | listed |
|---|---|---|---|---|---|---|---|
| 1 d | 50,031 | 239 s | 4.0 GB | 2,812 (14%) | 84 | 588 | 355 |
| 3 d | 195,794 | 225 s | 4.1 GB | 7,331 (12%) | 302 | 2,766 | 500 |
| 7 d | 455,792 | 283 s | 6.6 GB | 14,396 (12%) | 506 | 5,031 | 500 |
| 30 d | 2,469,983 | 802 s | 6.3 GB | 139,995 (24%) | 10,451 | 14,938 | 357 |
| 90 d | 5,295,770 | 2,166 s | 13.8 GB | 678,506 (60%) | 137,656 | 34,538 | 263 |

Stage split at 90 d: manifest 223 s, text trace 1,089 s (pair search 555 s),
rank 850 s (four trace extractions 600 s). Trace parquet: 8.6 KB (1 d) to
2.8 MB (90 d). `--windowed day` on 7 d: +36 s, +3.3 GB.

- The 90 d window does not fit 15 GB with the iot stack beside it. Default is
  30 d, `MemoryMax=12G`.
- Peak RSS is a ~4 GB floor from reading the embeddings object set, plus about
  13 KB per embedded vector (30 d to 90 d slope). Coverage was 12-60%; a 30 d
  window at full coverage holds about 584,000 vectors and is projected past
  12 GB. Re-measure when the embedding backlog clears.
- A 4 vCPU x86 host will be slower than this Mac on the quadratic pair search
  by an unmeasured factor.

## Open

- Rolling-rank floor, above. Raising it needs the global 5/10/20 sweep the
  depth test asked for; the windowed sweep does not answer it.
- `--min-entities` does not reach the text trace, which keeps
  `apply_text_floor`'s default. Settle together with the floor.
- The window length is a product decision: 90 d is the validated
  configuration, 30 d is what fits.
