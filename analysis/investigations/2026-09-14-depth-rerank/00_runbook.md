# Re-ranking on deepened data: the steps, in order

The question: the 2026-09-14 ranking check found 99 of 100 deepened accounts
left v2's top 500. That was 100 accounts. This deepens the whole persisted top
500 and asks what the ranking looks like when its head is no longer thin.

Each step is a command that has been run at least once before at smaller scale.
Timings are from this run and belong in `findings.md`, not here.

## 1. Deepen the target set

    ssh pi0 docker exec -d -e PYTHONUNBUFFERED=1 collector-monitor-1 \
        sh -c "/app/.venv/bin/monitor deep-timelines --limit 500 > /tmp/deep500.log 2>&1"

Detached, because the pass is hours and an ssh drop must not take it with it.
Targets come from the latest persisted `coord2/kind=scores` run. The ledger at
`state/deep_timeline.json` blocks accounts deepened inside
`DEEP_TIMELINE_REFRESH_DAYS`, so the 120 already done are skipped rather than
repeated.

Watch: `no_posts` and `failed`. Both were 1/120 and 0/120 on the last pass, and
a high rate at this scale means accounts written off permanently on soft
failures.

## 2. Embed what the pass collected

    uv run python investigations/2026-09-12-embedding-backfill/01_export_pending.py
    uv run python investigations/2026-09-12-embedding-backfill/02_encode.py
    uv run python investigations/2026-09-12-embedding-backfill/03_write.py

`02_encode.py` picks cuda, then mps, then cpu, and refuses to write unless
re-encoding already-embedded posts reproduces their stored vectors. On the mac
that is ~345 posts/s on MPS.

## 3. Snapshot

    uv run python -c "from kma import bench; bench.snapshot('2026-09-XX-deep500', rows=False)"

A full snapshot, not a derived one: this pass adds posts, not just vectors.
`rows=False` skips footer reads, so the manifest cannot verify a pinned read -
which is the trade the previous two runs took as well.

## 4. Text trace

    uv run --with modal modal run modal_textsim.py --snapshot 2026-09-XX-deep500 --threshold 0.85

## 5. Rank

    uv run python -m kma.coord2_run --snapshot 2026-09-XX-deep500 --top 500 --persist

## 6. The check

    uv run python investigations/2026-09-14-depth-rerank/01_rank_check.py \
        --before 20260914T090118Z --after <new run>

Reports the treated arm against two controls: the churn floor between the two
runs, and untreated accounts matched on before-rank band. Reproduced the
earlier ad-hoc finding exactly on the 100-account pass (100 treated, 1 retained,
+406 ranks, churn floor 207/500).
