# Incitement backlog and keep-up on the RTX 3070 (`kma.incitement_backlog`)

Scores posts that `kma.incitement.score_new` has not reached, on mike-pc, with the same
model, hypotheses, schema and `incitement/` layout. `incitement._pending`, `hatespeech`
and every other reader see the rows unchanged. tf1 scores 500 posts per 2 h on CPU;
new posts arrive at 36k-67k a day, so without this job the unscored set only grows.

## What it does

Pending: the latest row per post with non-empty text and no row under
`incitement/platform=x/` (the same rule as `db.pending_posts`). Order:

1. Kenya-relevant first: relevance classifier `p_kenya >= 0.5` (model
   `kenya-relevance-afroxlmr-2026-09-13`), or the `measure.domain_bucket` keyword gate
   for a post with no relevance score.
2. Newest `collected_at` first inside each group.

The ordered ids and texts are written once per run id to `plan.parquet`. Chunk i is plan
rows `[i*N, (i+1)*N)` (`N` = 10,000) and goes to
`incitement/platform=x/dt=<run day>/run=<run_id>-cNNNN.parquet`. Lexicon hits come from
`incitement.scan_text`, scores from `incitement.nli_scores`, the table from
`incitement.score_table`; `score_new` uses the same three, so the schema cannot drift.
A test pins the schema equal.

Inference: fp16, `dtype=torch.float16` on the same `zero-shot-classification` pipeline,
texts sorted by length inside a chunk so a batch carries little padding, batch 32. If a
chunk pushes `torch.cuda.max_memory_allocated()` past 7 GiB, or raises out of memory, the
batch halves (floor 8, then exit 4).

## Measured (2026-10-10, RTX 3070, 5,000 random already-scored posts)

Throughput on the first 1,000 posts, posts/s, peak allocated VRAM:

| dtype | order | batch 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| fp16 | length-sorted | 165 / 1.24 GiB | 194 / 1.94 | 189 / 3.34 | 123 / 6.14 | 20 / 7.54 (spilled) |
| fp16 | arrival order | 123 / 1.24 | 101 / 1.94 | | | |
| fp32 | length-sorted | 87 / 2.42 | 88 / 3.79 | 13 / 6.53 (spilled) | | |

On all 5,000 at batch 32: fp16 185 posts/s, 1.94 GiB; fp32 87 posts/s, 3.79 GiB. Above
about 6.5 GiB allocated Windows spills to system RAM and the rate collapses 6x to 10x,
which is why the ceiling check exists. Batch 32 is the choice: fastest, and 5x under the
limit.

Agreement against the scores already in R2 (produced on CPU by tf1), per score, 5,000 posts:

| | mean abs diff | p99 abs diff | max abs diff | flip of max(menace) at 0.5 | at 0.9 | same top hypothesis |
|---|---|---|---|---|---|---|
| fp16 vs stored | 0.0018 | 0.0102 | 0.0358 | 0.22% | 0.10% | 99.46% |
| fp32 vs stored | 0.0017 | 0.0094 | 0.0252 | 0.22% | 0.14% | 99.44% |
| fp16 vs fp32 | 0.0017 | 0.0094 | 0.0316 | 0.24% | 0.12% | 99.54% |

Correlation per hypothesis 0.9998 to 1.0000 in all three. fp16 is no further from the CPU
rows than fp32 is, so the residual is not a half-precision effect. `max(menace)` is the
max of dehumanisation, violence_call and othering, the quantity `measure.coded_suspect`
thresholds. Bench script: `analysis/investigations/2026-10-10-incitement-gpu/bench.py`.

## Backlog run (2026-10-10 23:22 to 01:22 London)

One wrapper attempt on pi0, run `20261010T222220Z`: 1,476,747 pending (473,400 Kenya-relevant
first), 148 chunks, 7,138 s wall (1 h 59 m, 207 posts/s including the plan), peak 1.94 GiB
VRAM, no retries. A 2,000-post trial run at the start went through the same path. Checked
afterwards against a fresh mirror of R2:

- 2,496,153 distinct posts with text, 2,490,938 distinct scored, 5,215 unscored.
  All 5,215 were collected after the plan was taken (236 on 10-10, 4,979 on 10-11); none
  from 09-28 to 10-09 remain. The first scheduled run takes them.
- 2,491,938 `incitement/` rows. 1,000 ids have two rows: 500 are a GPU chunk plus a tf1 row
  (tf1 scored them while the backlog ran), 500 are two tf1 rows. No id has three, and no
  chunk key was written twice. No scored id is missing from `posts/`.

## Crash safety

- Chunk: written to `cNNNN.parquet.tmp`, renamed, then one PUT (R2 applies a PUT whole).
  A flag file marks it uploaded.
- Resume with the same run id: the plan is reused (not rebuilt, even if tf1 has scored
  more posts since), finished chunks are skipped, a chunk scored but not uploaded is
  re-sent without rescoring, and a chunk that was mid-upload is overwritten at the same
  key. No path adds a second copy of a chunk.
- Marker `incitement/_backlog/platform=x/run=<run_id>.json` (outside the readers' glob)
  records status, chunks, sizes and a read-back of every chunk (rows equal distinct ids).
  pi0 then checks the marker against its own bucket listing.
- A different run id re-plans from the current R2 state, so a run that was cut short is
  finished by the next one.

tf1's own pass still writes while this runs. A post both score ends up in two rows
(different `run=` files). `_pending` is unaffected; readers that must have one row per
post take the newest `scored_at`.

## Where it runs

mike-wsl holds no R2 keys. pi0 signs short-lived urls for `posts/`, `incitement/` and
`relevance/` (GET) and for this run's chunk and marker keys (PUT), 36 h; the job mirrors
the inputs (about 1 GB, 8.7k files, first mirror 9 min, later ones only the new files)
and uploads through the PUT urls. Layout, cron line and retries: `deploy/pi0/README.md`.
Runtime on mike-wsl: `~/projects/kenya-incitement/` (clone, `pyproject.toml` from
`deploy/mike/incitement/`, `cache/`, `logs/`, `job.lock`).

## Keep-up job

Wednesday and Saturday 02:10 (pi0 cron), staggered from the Tuesday/Friday topics job.
The wrapper refuses to start while a `kt-*` tmux session or a busy GPU says the topics
job is running, and retries with backoff for up to 20 h. Gauges
`kma_incitement_{last_success_timestamp_seconds,last_attempt_timestamp_seconds,
scored_posts,pending_at_start_posts,last_attempt_ok}` in
`/var/lib/node_exporter/textfile/kma_incitement.prom`; alert `KenyaIncitementStale`
(5.5 days) in the iot repo.

A steady run scores what arrived in 3.5 days (about 125k-235k posts at the 36k-67k a
day rate), which at 185 posts/s is 11 to 21 minutes plus the mirror update.

## tf1

Once the backlog is clear and a few cron runs have succeeded, tf1's incitement pass adds
nothing the GPU job does not do faster. Disabling it frees its CPU and removes the
duplicate rows above. Do it only after `KenyaIncitementStale` has been quiet for a
couple of weeks, because tf1 is the only fallback if mike-pc is down.
