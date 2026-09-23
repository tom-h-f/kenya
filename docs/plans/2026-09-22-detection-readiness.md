# Detection readiness: from validated method to a running monitor

## The gap

Written 2026-09-22. The question was whether we have everything needed to
detect coordination when it starts - not whether there is any now.

The method is validated: v2 reproduces 6 of 6 IOHunter benchmark datasets, and
the blind A2 read showed it surfaces Kenya-relevant coordination where v1 does
not. What is missing is operational. Measured on this date:

- **The continuous detector is v1.** `analysis-enrich-1` on tf1 refreshes v1
  clusters a few times a day (452-468 clusters, ~2,900 accounts). v1's recall on
  the IO archive is 18.6%, and it was 0 of 11 Kenya-relevant in the
  size-matched blind read.
- **v2, adjudication and windowed detection run only by hand**, against frozen
  snapshots. `modal_coord2.py` takes a snapshot id; no schedule exists on tf1
  or Modal for any of them.
- **Nothing alerts.** Clusters and trend candidates land in R2 and nobody is
  told.
- **Nothing tests the chain.** Benchmark reproduction tests the method; nothing
  tests that the live pipeline would surface a campaign.

## Decisions

Made 2026-09-22 so the work below can run without waiting on them. Each can be
reversed; each says what would reverse it.

1. **v2 runs daily on Modal over a rolling window of the live corpus.** Daily
   because a coord2 pass takes over an hour, and Kenyan hashtag campaigns run
   for hours to days. The window builder reuses the snapshot code, so a
   production run is still reproducible from its window id.
2. **Adjudication stays on Modal; alerting runs on tf1.** ntfy lives on tf1 and
   is tailnet-only, so Modal cannot reach it, and the 2026-09-08 decision keeps
   `ANTHROPIC_API_KEY` off tf1. Modal writes verdicts to R2; a small tf1 job
   reads new verdicts and posts to ntfy topic `kenya-leads`. Secrets stay where
   they already are.
3. **Adjudication budget: at most 10 candidates per daily run**, ranked, new or
   materially changed since the previous run. The cap is a cost bound, and the
   run reports what it skipped.
4. **The canary never touches the archive.** A synthetic coordinated group,
   shaped from a real IO archive operation, is injected into the detector's
   input in memory, weekly, and must come out as a `[CANARY]` alert. A canary
   that does not arrive is itself an alert.
5. **Windows: daily at floor 5 for detection, weekly at floor 10 for the stable
   series** - pending the `MIN_ENTITIES` sweep, which can overturn either
   floor. Daily at floor 5 held 310 accounts over 60 days.
6. **Re-point collector promotion at v2 community ranks, with a hard cap on
   targets added per pass.** The cap answers the 6.6x targeting blowup that
   froze the Leiden resolution.
7. **The follow crawl moves off the every-cycle path.** Since the 2026-09-21
   fix it runs to completion, and cycles went from 110-150 min to 367-421 min.
   Search coverage drops ~9x after day 7, so collection frequency comes first.
8. **Concealment features are validated on the IO archive before use.** A
   signal is kept only if it separates known operation accounts from matched
   controls - the same test that refuted the self-amplification filter.
9. **v1 keeps running until v2 has run daily for two weeks**, then retires
   (Track C, C5).

## Workstreams

Six, in parallel, one branch each. None deploys: every production change -
pi0 rsync, `modal deploy` of a schedule, a tf1 service - waits for review.

| # | Branch | Delivers |
|---|---|---|
| A | `feat/v2-daily` | rolling-window builder and a daily Modal pipeline: textsim, coord2, community scores to R2 |
| B | `feat/leads-alerts` | candidate diff, bounded adjudication, verdicts to R2, tf1 ntfy poller |
| C | `feat/canary` | synthetic IO-shaped group, in-memory injection, end-to-end check |
| D | `feat/windowed-floor` | `MIN_ENTITIES` sweep, window decision, windowed scoring entry point |
| E | `fix/collection-cadence` | follow-crawl cadence, promotion re-point with a cap, control-arm budget |
| F | `feat/concealment` | deletion base rate, creation-burst, handle/bio change and avatar-reuse features, IO-validated |

B and C build against the existing `coord2/kind=scores` schema, so they need
not wait for A. D's windowed scoring is wired into A's pipeline after both land.

## Out of scope by decision

WhatsApp, TikTok and Facebook. Kenya's documented operations coordinate there
and surface on X as trends; this monitor sees the X end only, and anything
published says so.

## Merged and deployed, 2026-09-23

All six branches merged. Both suites green: 308 collector, 806 analysis.

**Deployed to pi0** (steps 1 and 4): the cadence work and the random re-check
arm. Nothing else is live - the Modal schedules and the tf1 poller are written
and unstarted, waiting on the spend limit and an ntfy token.

**Four seams the branches could not close alone**, none of which any single
workstream owned:

1. Planted canary posts enter the window through `snapshot_networks`'
   `transform_view`, before any trace is built, so the canary is carried by
   co-action rather than by a trace patched in afterwards.
2. The leads diff reads `kind=daily_members` - the daily pass's full
   membership. Leiden ids are not stable between runs, and the listing carries
   only the accounts that made the triage budget.
3. A canary candidate gets `canary.packet` instead of `dossier.build`, so the
   reader judges planted evidence rather than an empty packet.
4. `kind=scores` carries a `community` column, and the collector's promotion
   ranks within community before it takes its cap. Global centrality
   concentrates in one co-retweet block (2026-09-12), so a cap taken in global
   order could spend every slot inside it.

**The text trace has a time axis** (step 3). It was the largest gap in D's
result: over half the fused edges globally, and invisible to windowed detection
because the persisted trace is one row per pair with no timestamps. The GPU
pass now buckets by the later post of each qualifying pair and writes the
pair-level object from the same search, so one quadratic pass serves both and
the two cannot disagree.

**The deletion denominator is fixed at the source** (step 4). `collect_metrics`
re-checks 50 uniformly sampled posts a pass beside the ~400 selected on
engagement, tagged `arm='random'`. The engagement arm drew 0 of 467 re-checks
from the control partition on 2026-09-22, so the only exogenously sampled
partition was absent from the base rate by construction. `kma.deletions` reports
the arms separately and refuses to pool them.

### Still open

- **Windowed detection does not yet consume the timed text trace.** The trace
  and a slicing loader exist (`coord2_run.load_textsim(..., bucket="day")`);
  wiring it into `kma.windowed` is the next step, and D's floor decision was
  made without this trace.
- The Modal spend limit stops the full-scale runs AND has stopped relevance
  scoring in production since 2026-09-14.
- The canary's `[CANARY] MISSING` alert will fire correctly on day one, until a
  daily run with the wiring above has actually produced a listing.
