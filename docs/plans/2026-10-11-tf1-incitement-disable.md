# tf1 incitement pass: duplicate scoring and the safe disable

2026-10-11. Branch `fix/flagged-ops`. Nothing here has been run on tf1; Tom approves each step.

## Finding: 1,000 posts have two incitement rows

Measured on R2 `incitement/platform=x/` (2,494,438 rows, 2,493,438 distinct posts):

| posts | runs | gap |
|---|---|---|
| 500 | tf1 `20260901T094346Z` and tf1 `20260901T104758Z` | 64 min |
| 500 | GPU chunk `20261010T222220Z-c0047` and tf1 `20261010T231610Z` | 13 min |

Cause of the second pair: `score_new` reads its pending set, spends minutes in the CPU NLI
(0.7 to 1.1 posts/s), then writes. The mike-pc backlog run scored the same newest posts
in that gap. Cause of the first pair is not provable from the data: both are tf1 passes
an hour apart. `db.pending_posts` treated ANY error reading the scored prefix as "nothing
scored yet" and dropped its anti-join, which rescores the newest 500; that is the
likeliest mechanism and is now closed (below).

Nothing double counts. Every reader keeps the latest row per post
(`db.latest_incitement`, `hatespeech._nli_for`, the refresh query). The enrichment card
in `kma.dashboard` is labelled a row throughput ratio, so 1,000 extra rows in 2.49 M is
inside what it states. Tests: `analysis/tests/test_incitement_duplicates.py`.

Code changes:
- `db.pending_posts` now raises on a failed read of the scored prefix; only "no files" or
  a missing table means everything is pending.
- `db.drop_scored` is called by `incitement.score_new` just before the write, so the
  overlap shrinks from the inference time to one listing. Not a lock.
- `ENRICH_INCITEMENT=0` turns the incitement pass off in the `--loop` worker
  (`enrich.incitement_enabled`). Default on, so nothing changes until it is set.

The GPU backlog plan is frozen per run on purpose (resumable chunks), so it can still
write a post tf1 scored first. Harmless for the same reason.

## Consequence of disabling: coded_suspect ordering

`PASSES` runs incitement before hate because `hatespeech._measure_frame` resolves
`coded_suspect` at scoring time from the NLI rows. With tf1's pass off, a post that tf1
hate-scores before mike-pc's Wednesday or Saturday run writes its NLI gets
`coded_suspect=False`. `python -m kma.hatespeech --refresh-measure` (CPU, no model) fixes
those rows: it re-measures rows whose NLI is newer than their hate score. Run it after
each Wed/Sat incitement run, or accept up to 3 to 4 days of `coded_suspect=False` on new
posts. Scheduling it is a separate decision for Tom; this procedure does not add it.

## Install procedure (needs Tom's approval; writes on tf1)

tf1 builds from `~/kenya-monitor-2027` (not a git repo), compose project `analysis`,
service `enrich`, container `analysis-enrich-1`, env from `~/kenya-monitor-2027/.env`.
Only this container is touched.

1. Ship the code (after this branch is on master; `<sha>` is master's head):
   ```
   ssh tf1 'cp -a ~/kenya-monitor-2027/analysis/src ~/backups/analysis-src-pre-incitement-off-20261011'
   rsync -a --exclude __pycache__ analysis/src/kma/{enrich,incitement,db}.py tf1:kenya-monitor-2027/analysis/src/kma/
   ```
2. Back up and set the knob:
   ```
   ssh tf1 'cd ~/kenya-monitor-2027 && cp -p .env .env.pre-incitement-off-20261011 \
     && printf "\nENRICH_INCITEMENT=0\n" >> .env && tail -n 2 .env'
   ```
3. Rebuild and recreate only `enrich`:
   ```
   ssh tf1 'cd ~/kenya-monitor-2027/analysis && GIT_SHA=<sha> docker compose up -d --build enrich'
   ```
4. Verify:
   - `docker logs analysis-enrich-1 2>&1 | grep "incitement pass disabled"` shows one
     WARNING line at startup.
   - After the first full cycle: `cycle N done` appears, and `incitement-scored` does not.
     `embedded`, `labelled` and `hate-scored` lines still appear.
   - `docker ps` shows the container up with 0 restarts; peak memory lower than before.
   - R2: no new `incitement/platform=x/dt=*/run=<stamp>Z.parquet` without a `-cNNNN`
     suffix after the change (tf1's rows have no suffix, mike-pc's do).
   - After the next Wed/Sat 02:10 pi0 run: pi0 gauge `kma_incitement_last_attempt_ok 1`.
5. Rollback: `cp -p .env.pre-incitement-off-20261011 .env`, then step 3 again.
