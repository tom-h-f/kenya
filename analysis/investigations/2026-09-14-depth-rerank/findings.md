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

## This run

(pending: the 500-account pass with the holdout, embeddings, snapshot, trace,
rank, check)
