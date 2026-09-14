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

## The answer

Snapshot `2026-09-14-deep500` (28,292 objects), 91,983 posts embedded on the mac
at 335/s with cosine 1.000000 against production, text trace 111,222 edges over
27,024 users (against 109,479 / 26,579 before), v2 run
`coord2/kind=scores/dt=2026-09-14/run=20260914T150111Z`.

| arm | in top 500 before | in top 500 after | retained |
|---|---|---|---|
| treated (deepened) | 390 | **0** | **0.0%** |
| holdout (selected, not fetched) | 97 | 22 | 22.7% |
| untreated | 12 | 3 | 25.0% |

Matched on before-rank band, treated retention is 0.0% in every one of the five
bands. The holdout - drawn from the same ranked selection by the same rule,
differing only in having been fetched - kept 22.7%.

**The top 500 does not survive deepening.** The earlier 100-account result was
not a fluke of that sample: at four times the scale, with a proper control, the
effect is total.

### What this means, and it is not "deepen everything"

The accounts v2 ranks highest are the ones it knows least about, and giving them
history removes them. An account holding two posts that both touch viral objects
has a one-hot co-action vector, so its cosine against every other toucher is 1.0
whatever TF-IDF does. Two hundred posts of real behaviour make it read as
ordinary, which it is.

So deepening is not a coverage improvement that happens to move the ranking. It
is a destructive test, and the ranking fails it. **v2's centrality over this
corpus ranks sparsity, not coordination**, for as long as the activity floor
sits at 2 entities.

### The ranking's head is its least stable part

The control arm shows this independently of any treatment:

| before rank | control retained |
|---|---|
| 1-100 | 5.6% |
| 101-200 | 9.5% |
| 201-300 | 4.5% |
| 301-400 | 50.0% |
| 401-500 | 50.0% |

Untreated accounts in the top 300 churn out at about ten times the rate of
untreated accounts in the bottom 200. The part of the ranking anyone would act
on is the part that does not hold still.

### The recommendation: a stability criterion, not a triage budget

The user's own test was "if the top 500 survives deepening, it is real; if it
churns again, the method needs a stability criterion rather than a triage
budget." It churned, so:

- **Raise the activity floor to where the ranking is stable.** `MIN_ENTITIES` is
  2, which is the value that admits the one-hot pathology. The floor should be
  set empirically - re-run at 5, 10, 20 and measure holdout retention at each -
  rather than left at the smallest number that is not 1.
- **Report rank agreement under a depth perturbation beside any ranking.** The
  holdout makes this cheap and repeatable: deepen a random share of the
  selection, re-rank, and publish what fraction of the untreated arm held its
  place. A ranking whose control retention is 22.7% should not be presented as
  a finding.
- **Do not deepen the whole target set again.** It consumes the candidates the
  next measurement needs, and the 30-day refresh TTL means the arm cannot be
  rebuilt for a month.

### What it does NOT overturn

The A2 adjudications remain valid as "what the method surfaces", which is how
they are reported. The benchmark reproduction is untouched - on the IO datasets
the operations' accounts sit in a dense core with real history, which is
precisely the condition this corpus fails.
