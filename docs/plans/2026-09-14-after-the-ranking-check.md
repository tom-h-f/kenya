# After the ranking check: five items, in order

The ranking check (`2026-09-08-collector-depth.md`, "The ranking check, and it
inverts this task's premise") found that deepening an account makes it FALL out
of v2's ranking: of 120 deepened accounts, 100 were in the top 500 before and 1
after, against a churn floor of 293/500. The accounts v2 ranked most
confidently were the ones we knew least about, and the rankings were inflated by
sparsity.

That result sets the order of everything below. Nothing downstream of the
ranking - adjudication, publication, a dashboard - means much until the ranking
has been measured on a corpus where its top accounts are not thin.

## Todo

- [ ] **1. Re-rank on deepened data.** Deepen the current top 500 (~5,000
      requests), rebuild the text trace on a snapshot that holds the new
      partitions, re-run v2, and report what survives. Decide from the result
      whether the method needs a stability criterion rather than a triage
      budget.
- [ ] **2. Depth as a standing cost.** A small per-cycle depth budget inside
      `run_scheduler`, so coverage tracks collection to August 2027 instead of
      being overtaken between campaigns.
- [ ] **3. Finish or close C1.** Reproduction is 0.641 against a 0.95 bar with
      three measured suspects left. Either spend the session on them or write
      down that policy replay is indicative rather than exact.
- [ ] **4. The control arm.** A random sample of Kenyan discourse collected
      independently of the target list: the one thing that would make any
      prevalence statement publishable.
- [ ] **5. The dashboard.** Unblocked once 1 settles the ranking.

## 1. Re-rank on deepened data

The premise: if the top 500 survives deepening, the ranking is real; if it
churns the way the first 120 did, the method is ranking sparsity and needs a
stability criterion.

Steps, each of which has already been run once at smaller scale:

1. `monitor deep-timelines --limit 500` on pi0, against the latest persisted v2
   run (`coord2/kind=scores/dt=2026-09-14/run=20260914T090118Z`). The candidate
   query costs ~21 minutes and is paid once per pass, so this runs as one pass,
   not ten. Re-check `no_posts` and `failed` rates at scale - both were 1/120
   and 0/120 before, and a high rate would mean accounts written off on soft
   failures.
2. Embed the new posts (locally, MPS, ~345/s) so the text trace sees them.
3. A derived `bench` snapshot pinning the new `deep_timeline` partition.
4. Text trace on Modal, then `coord2_run` at `top=500`.
5. Join the new scores against `deep_timelines/` and report rank movement, as
   the depth plan requires of any post-pass re-run.

What the answer changes:

- **Top 500 largely survives** - the ranking is stable under depth once the
  thinnest accounts are settled, and the triage budget stands as it is.
- **Top 500 churns again** - the ranking is a function of how much we happen to
  hold per account, and the report needs a stability criterion (rank agreement
  under a depth perturbation) rather than a fixed cut at 500.

Either answer is publishable as a negative or positive result about the method;
the current state, where we know the first 120 churned and nothing else, is not.

## 2. Depth as a standing cost

Both depth passes are `run_once` commands deliberately kept OUT of
`run_scheduler`'s cycle, because at ~10 requests per account they would become
the dominant consumer of a cycle (see `scheduler.run_deep_timelines_once`). That
was the right call for a 167,220-id backlog. It is the wrong call for a
programme that collects until August 2027, because:

- Hydration coverage went 44% -> 49.8% in a bounded pass and 60% was measured
  unreachable as a one-off; the denominator grows as collection continues.
- Deep-timeline yield decays - the accounts worth deepening are the thin ones,
  and every pass consumes the thinnest.
- A one-off backfill is overtaken by the collection that follows it.

So the design question is the budget, not whether: a per-cycle cap small enough
that baseline collection is unaffected, applied on elapsed time rather than
cycle count (`_cycle_estimate_s` is the precedent - cycle counters reset on
every restart). The ~21-minute candidate query is the real constraint: it makes
tiny frequent passes wasteful, so the cadence has to amortise it.

## 3. Finish or close C1

Recall 0.641 with the toxic arm modelled, against a 0.95 bar. Three suspects,
each measurable against data already recorded:

- **Pass timing.** `census_pass_times` dates a pass by its first engagement
  write, minutes after selection ran, so the replay stands late and sees posts
  the live selector could not. `census_runs` rows carry their own timestamp.
- **Rate-limit truncation.** The replay fetches its whole selection; a truncated
  live pass fetched a prefix. `census_runs.fetched_retweeted` records what
  landed.
- **Empty fetches.** An object whose fetch returned no retweeters writes no
  engagement row and is scored as not-fetched. `census_ttl/` records selection
  independently of outcome, for every pass since 2026-09-08.

Order: timing first (it is a systematic offset and would move every pass),
then empty fetches, then truncation. If the three together do not clear 0.9,
close C1 explicitly: policy replay is indicative, not exact, and candidate
collection policies get scored on direction rather than on per-id agreement.

## 4. The control arm

`2026-09-13-publishable-statistics.md` finds that everything except prevalence
is publishable today, and that prevalence needs "a random sample of Kenyan
election discourse collected independently of the target list". That is a
collection-design problem, not an analysis one, and it is the only route from
"here is what our detector surfaces" to "here is what is happening".

`runner.census_discovered_handles` is the precedent and the argument: its
`ORDER BY random()` is load bearing, because selecting on the outcome being
measured manufactures the answer. A control arm is the same idea applied to
posts rather than handles, and it needs a defensible sampling frame - what
counts as Kenyan election discourse, drawn how, at what rate - before any code.

## 5. The dashboard

Blocked on an R2 token scoped to both buckets and a Zero Trust application,
both deliberately unrequested until v2 had something worth publishing. Item 1
decides whether it does.
