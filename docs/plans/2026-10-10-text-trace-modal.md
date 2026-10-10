# Text trace on Modal, rank on cloud

Written 2026-10-10. Tom's decision: the v2 daily's text trace (embed the
window's missing posts, pairwise cosine search) moves to a Modal A10G. Rank
stays on `cloud`. Reason: a 21 d window at full coverage holds about 410k
vectors and the quadratic search does not fit in `MemoryMax=12G` there (30 d at
24% coverage peaked at 11.85 GiB). Measured 2026-10-10: the 21 d window has
393,104 text-eligible posts, 351,292 without a vector.

## Shape

    02:30 UTC  Modal cron  kma-text-trace.build_trace      writes R2
    03:30 UTC  cloud timer kma.v2_daily_run --text-trace modal   reads R2

The only coupling is R2. Cloud holds no Modal credentials.

## Modal side

- New app `kma-text-trace`, file `analysis/modal_text_trace.py`, logic in
  `kma.text_trace` (Modal-free, tested offline). Not an extension of
  `kma-textsim`: that app reads a bench snapshot from a volume, has no
  sentence-transformers, and a schedule on it would bill under a name that
  hides what it does. Own app = own line in `kma.modal_spend`.
- Window: `end` = yesterday UTC, `days` = `v2_daily_run.DEFAULT_DAYS`, read at
  runtime, not copied.
- Steps, in the order `v2_daily.orchestrate` already uses: embed the window's
  missing posts (`embed_window.embed_missing`), pin the manifest
  (`window.ensure`), build the trace (`textsim_run.build`, same threshold,
  overlap floor and day bucket as the cloud runner) unless it exists
  (`v2_daily.text_trace_exists`, the same check the cloud makes). A covered
  window costs two reads.
- The trace is written under the key the cloud's `text_trace_exists` reads
  (`coord2/platform=x/kind=textsim/snapshot=<window id>/threshold=0.85/overlap=0.10/edges.parquet`),
  and the day-bucketed object beside it.
- If the manifest already existed and the embed stage then wrote vectors, the
  manifest is rebuilt: a manifest pinned before those vectors exist would hide
  them from the trace silently.
- Schedule: gated by `KMA_TEXT_TRACE_SCHEDULE=1` at deploy time. A plain
  `modal deploy` creates the app with no cron. Tom turns it on after reading
  the measured cost.
- `scratch=<prefix>`: the manifest is held in memory and the trace is written
  under `<prefix>/` in the bucket. Used for the equivalence test, touches no
  production key.

## Cloud side

`--text-trace {modal,local,none}`, default `modal`.

| mode | embed | trace | if absent |
|---|---|---|---|
| `modal` | no | reuse the persisted trace | rank without text; loud |
| `local` | per `--no-embed` | build on this host (today) | n/a |
| `none` | no | skipped on purpose | n/a |

When `modal` finds no trace it never builds one locally (that is the OOM). It:

- sets `text_trace = "missing"` on the `daily_runs` row, with
  `text_threshold` null on the row and on `kind=scores`;
- logs a WARNING with the key it looked for;
- writes `kma_v2_daily_text_trace_present 0` beside the unchanged freshness
  gauge, in the same atomic write.

`v2_daily_check postrun` reports the flag from the latest `daily_runs` row and
fails on `missing`. The freshness gauge name, file and semantics are unchanged.

## Verification

1. Offline tests: stage order, key equality, mode behaviour, gauge, check.
2. Equivalence: one small window, Modal path to a scratch prefix vs the local
   path (`textsim_run.build`, same pinned manifest, CPU torch). Edges and
   weights within float tolerance.
3. Measure on Modal: small window, then the real window. Wall, peak memory,
   cost. Budget for testing: under $3.

Results are filled in below as they land.
