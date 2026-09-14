# Re-ranking on deepened data

Follows the ranking check in `docs/plans/2026-09-08-collector-depth.md`, which
found that 99 of 100 deepened accounts left v2's top 500. That pass covered 100
accounts. This one covers the whole persisted target set.

## The design change this run makes: a holdout

The earlier check could compare treated accounts against untreated ones only
because the pass happened to cover 100 of the top 500. A pass that deepens its
whole target set has no such comparison left, and the comparison is load
bearing: over the same pair of runs, untreated accounts retained 51.5% of their
top-500 places. A recomputed ranking churns on its own.

So `monitor deep-timelines --holdout 0.2 --holdout-seed 0` takes the ranked
selection, draws 20% of it uniformly at random, and deliberately does not fetch
those accounts. They are written to `deep_timelines/` with `status=holdout` and
`posts_written=0` before the first fetch, so the split exists in the record even
if the pass aborts.

Uniform over the whole ranked list, not a tail slice: the strata put the
thinnest accounts first, so holding back the tail would make the control the
accounts least like the treated ones.

## Reproducing the earlier finding with the tooling

`01_rank_check.py` against the two runs the ad-hoc check used:

    before 20260914T025108Z (snapshot pinned 2026-09-12 23:05)
    after  20260914T090118Z (snapshot pinned 2026-09-14 09:44)

| | in top 500 before | in top 500 after | retained |
|---|---|---|---|
| treated | 100 | 1 | 1.0% |
| untreated | 400 | 206 | 51.5% |

Churn floor 207 of 500 shared. The one surviving treated account moved 18 ->
424. Matched on before-rank band, the treated accounts in ranks 1-100 retained
1.0% against 0.0% for the two untreated accounts in that band - which is the
reason the holdout matters: n=2 is not a control.

The treated window is taken from the two runs' SNAPSHOTS, not their clocks. The
before-run was computed after the depth pass had finished and was still blind to
it, because its snapshot predated the new partition.

## The pass

`monitor deep-timelines --limit 500 --holdout 0.2 --holdout-seed 0`, 2026-09-14.

| | |
|---|---|
| selected | 487 (the ledger blocked 13 of the top 500, already deepened) |
| holdout | **97** |
| treated | 390 |
| deepened | 384 |
| no_posts | 6 |
| failed | 0 |
| posts written | **99,066** |
| authors written | 6,193 |
| floor_cleared | 0 |

**The terminal-status rate holds at scale.** `no_posts` is 6 of 390 (1.5%)
against 1 of 120 (0.8%) on the bounded pass, and `failed` is still 0. A by-id
timeline fetch cannot distinguish suspended from protected from empty and all
three are recorded terminal, so a high rate here would mean accounts written off
permanently on soft failures. It is not high.

`floor_cleared` is 0 again, as predicted: the activity floor runs before
centrality, so every account in a persisted `kind=scores` run already clears it.
What the pass buys is 254 posts per account of behavioural history for accounts
previously ranked on 2 to 19 observations.

**Timings, which are the numbers the per-cycle depth budget needs.** Selection
took 22 minutes (10:27 to 10:49 UTC) against the ~21 minutes the parent-backfill
pass measured, so the candidate query cost is stable and is paid once per pass
whatever the limit. Fetching took 65 minutes for 384 accounts - **5.9 accounts
per minute**, or about 10 seconds per account at depth 200.

At that rate a 50-account in-cycle pass is 22 minutes of selection plus 8
minutes of fetching: 73% of its wall clock on selection, which is the argument
for the 12-hour cadence rather than a per-cycle one.

## The first attempt, and what killed it

Three heavy R2 readers at once - two replay reproductions on the mac and the
depth pass on pi0 - and all three died within minutes of each other on DNS and
connection failures (`Could not resolve hostname` on pi0, `Could not connect to
server` on the mac). Resolution was healthy again immediately afterwards. Run
one heavy R2 job at a time; the bucket is not the bottleneck, the resolver is.
