# Embedding coverage lag

Measured 2026-10-10: the v2 daily's text trace saw 12% to 24% of the
text-eligible posts in its windows (24% for 30 d, 14% for 1 d). The corpus was
100% embedded on 2026-09-12 by a one-off mike-pc backfill; the live path has
not kept up since.

## Root cause

Text-eligible posts (`coord2.text_rows`) in the 30 d window ending 2026-10-09,
by collection day, with a vector as of 2026-10-10:

| collection days | eligible | with vector |
|---|---|---|
| 09-10 to 09-13 | 56,613 | 100% (the 09-12 backfill) |
| 09-14 | 34,473 | 78% (a 169,279-row run on 09-14) |
| 09-15 to 10-09 | 492,896 | 6 to 20%, median 11% |
| window | 583,982 | 140,007 (24.0%) |

- The enrich worker on tf1 writes 2,500 to 5,000 vectors a day (5 to 10 runs of
  500 per day since 09-15). The corpus adds about 43,000 unique posts with
  text a day (1,287,066 in the 30 d window), 19,500 of them text-eligible
  (583,982). The backlog grew by roughly 39,000 posts a day: the per-cycle cap
  of 500 is the cause, as hypothesised.
- A cycle is not cheap either. tf1 has 2 vCPU and 3.7 GiB; the embed pass
  spends about 12 minutes in `pending_posts` (an unwindowed anti-join over the
  whole corpus, once per pass) and 50 s encoding its 500 posts (10 posts/s);
  classify, incitement and hate follow, and incitement is being OOM-killed at
  the 2 GiB cap. One cycle runs about 1 h 53 min and embeds 500.
- `ENRICH_LIMIT` is global to all four passes, so raising it on tf1 by env
  would also raise classify, incitement and hate by the same factor on a host
  that already kills one of them. That path is not viable without a compose or
  code change.

## Decision: embed on `cloud`, as the first stage of the daily

| path | marginal cost | verdict |
|---|---|---|
| a. raise tf1's limit | 0 | One limit for four passes, 2 vCPU, incitement already OOM-killed. 43,000 posts a day is 72 min of encode a day plus a 12 min scan per cycle. No. |
| b. embed inside the Modal relevance job | about +$3.6 to $8 a month (estimate: it adds a second unwindowed `pending_posts` scan at A10G rates; the GPU encode itself is under $0.01 a day) | Fits the budget, but the relevance job fires at 14:45 UTC, so the part of each day collected after that is never in the window of the 03:30 run. Not measured by deploying. |
| c. embed on `cloud` before the daily | $0, 22 minutes of niced CPU a day | Chosen. |

Measured on `cloud` (AMD EPYC-Milan, 4 vCPU, CPU torch 2.13, scratch venv under
/tmp, 600 to 1,000 real posts from 2026-10-09):

- 4 threads: 16.2 posts/s on text-eligible-shaped posts, 14.0 posts/s on posts
  of all kinds; 3 threads: 13.2 posts/s. RSS 1.9 GB, model load 16 s.
- 19,500 eligible posts a day is 22 minutes; encoding every post with text
  (43,000) would be 48 minutes, so the stage embeds only what the trace reads.
- The daily already runs `Nice=10` inside `MemoryMax=12G`; the model is dropped
  (`semantic.release_models`) before the trace builds.

### What changed

- `kma.embed_window`: window posts read by `dt`, projected, deduped;
  `coord2.text_rows(missing_from=...)` drops posts that have a vector in SQL
  before any text is cleaned; only the missing posts are encoded, in chunks of
  5,000, one `embeddings/` run per chunk (same row format as `embed_new` and
  the Sept backfill). A covered window costs two reads and no model load.
- `v2_daily.orchestrate(embed=...)` runs it BEFORE the manifest, because the
  manifest pins the vectors that exist when it is built and a re-run reads the
  manifest back. `--no-embed` skips it; `--out-dir` never runs it (it writes
  nothing to R2).
- `window.keeps`: embeddings are no longer bounded at `end`. The daily embeds
  yesterday's posts today, so their vectors are dated after `end` and the old
  bound hid exactly those. The persisted manifest, not the bound, is what
  makes a window reproducible.
- The enrich worker is untouched on tf1. Its embed pass becomes redundant for
  the trace; it can be turned off when its compose file is next edited.

## Backfill

Gap since the last full coverage: 943,967 posts pending by the enrich worker's
own definition (`db.pending_posts`, all posts with text, including retweets).
Run with `analysis/investigations/2026-09-12-embedding-backfill` on mike-pc.

Results (mike-pc, RTX 3070, fp32, `awake on 3h` held, `awake off` after):

- 5,000-post smoke: 898 posts/s, 2.17 GiB peak; check on 2,000 already
  embedded posts, cosine min 1.000000, p01 1.000000, median 1.000000.
- Full: 943,967 posts encoded in 1,182 s (799 posts/s, 2.17 GiB peak), written
  by `03_write.py` as 19 runs under `embeddings/.../dt=2026-10-10/`.
- Coverage of text-eligible posts, 30 d window ending 2026-10-09: 140,007 of
  583,982 (24.0%) before, 583,982 of 583,982 (100%) after, every collection day
  100%. All 1,287,066 posts in the window have a vector.

## Memory at full coverage (Mac, `--days 30 --out-dir`, nothing to R2)

| coverage | text trace edges / users | wall | peak RSS |
|---|---|---|---|
| 24% (139,995 vectors) | 10,451 / 4,797 | 775 s | 5.3 GB |
| 100% (583,982 vectors) | 102,023 / 24,517 | textsim 2,423 s, rank 413 s | 8.3 GB (sets at the trace) |

The text trace now loads vectors as FLOAT, keeps the latest row per post with
an aggregate, and `text_similarity_network` no longer widens a caller-supplied
float32 matrix. The 8.3 GB is with those changes. `cloud` is not the Mac: its
24% run peaked at 11.85 GiB against 6.3 GB here, so 8.3 GB does not transfer.
A 30 d run at full coverage on `cloud` needs measuring before it is trusted
under `MemoryMax=12G`; the 21 d window (7.97 GiB at 10.6% coverage) needs the
same measurement.
