# Route 2: finding the campaign hashtags our keyword list cannot see

**Status: redesigned 2026-09-15 after the original premise was measured and
failed.** The replacement is cheaper, needs no new endpoint, and has already
produced a result.

## The problem

We collect by 34 fixed keywords in `collector/config/targets.yaml`. A
manufactured campaign invents its own hashtag, which is by construction not on
that list. Kenya's documented disinformation-for-hire pattern is to drive a
hashtag into trending and then delete, so the window in which it is visible is
short - and search coverage collapses ~9x past day 7 (measured 2026-09-14), so
anything missed live is effectively unrecoverable.

## What was tried first, and why it is dead

Pull X's own trend list via twscrape's `trends()`.

**Measured 2026-09-15 on pi0 against the live pool: it does not work.**

```
API unknown error: 200 - GenericTimelineById - (-1) Internal server error
(no raw responses yielded)
```

Both `trending` and `news` return nothing. The failure is X rejecting the
GraphQL operation under an HTTP 200, not an empty trend list, so it is not a
locale problem and cannot be fixed by changing which account asks. The op id in
this twscrape build is stale or the endpoint now needs session state that
cookie-auth accounts do not carry.

`search_trend(q)` still works, but it takes a query - it censuses a trend you
already know about. It discovers nothing.

So the original open question ("do our accounts see Kenyan trends?") is
answered in a way that removes the route as designed. Do not spend more on it
without first re-testing `trends()` after a twscrape upgrade.

## The replacement: discover hashtags in the control arm

The control arm samples 15-minute windows of Kenyan political discourse at
random and censuses each one whole. That is an **exogenous** sample - nothing
about a post decides whether it is collected except the minute it was posted
in - which makes it a legitimate discovery surface, and one our keyword list
cannot bias.

### It already worked

502 control-arm posts, 410 authors, collected 2026-09-14/15:

| hashtag | posts | authors | reachable by our keyword list |
|---|---|---|---|
| `#LindaMwananchiNairobi` | 24 | 11 | yes |
| `#BaBuKwaSababu` | 7 | 7 | yes |
| `#SisiNdioSifuna` | 4 | 4 | yes |
| `#NairobiNaSifuna` | 4 | 4 | yes |
| `#KaNairoTunamuOk` | 3 | 2 | **no** |
| `#NairobiNaFei` | 2 | 1 | **no** |
| `#LindaMwananchi` | 2 | 2 | yes |

**At least 2 of 7 repeated hashtags are outside our keyword list.** That is a
floor, not an estimate: the "reachable" column uses a generous substring test,
and X does not match the phrase `"Linda Mwananchi"` against the single token
`#LindaMwananchiNairobi`, so several of the "yes" rows are probably also misses
in practice. The precise test is a search query per tag, and it is worth running
before quoting any number.

`#LindaMwananchiNairobi` is 24 of 502 control posts - 4.8% of a random sample of
Kenyan political discourse - from 11 authors, 2.2 posts per author.

### Design

1. **Hashtag frequency over the control arm**, per tag per window, as a standing
   query. No new collection: the arm already writes `posts/type=control`.
2. **Emergence, not volume.** The quantity of interest is a tag appearing in
   sampled windows where it did not before, normalised by how many windows were
   sampled. A tag that has always been there is a topic; a tag that appears in
   three windows this week and none before is a candidate.
3. **Census the candidates** with `search_trend(q)` into `posts/type=trend`, a
   TARGETED partition - selection is conditioned on the tag having been noticed,
   so it must stay out of every prevalence denominator.
4. **Record every candidate considered**, including ones not censused, to a
   `trend_candidates/` prefix. The series of what emerged is a finding in itself
   and cannot be reconstructed later once the 7-day search window closes.

### The selection rule has to be fixed in advance

Otherwise this becomes a second keyword list chosen on outcomes, which is the
bias the control arm exists to remove. Three defensible rules, and this is a
decision rather than a detail:

- **Volume floor**: any tag in >= N sampled windows. Simple, but tracks what is
  popular rather than what is new.
- **Emergence**: any tag whose window-rate this period exceeds its prior rate by
  a stated factor. Matches the threat shape, needs a baseline period.
- **Author concentration**: any tag whose posts-per-author exceeds a floor.
  `#LindaMwananchiNairobi` at 2.2 would clear a floor of 2; ordinary discourse
  tags sit near 1.0. This is the most campaign-specific of the three and the
  most likely to be accused of assuming the conclusion.

Recommendation: **emergence, with author concentration recorded beside it but
not used for selection.** Emergence is about timing and does not presuppose
coordination; concentration is then an independent property of the tags it
finds, which keeps it usable as evidence rather than as a selection criterion.

## Cost

Small. The discovery query is over `posts/type=control`, which is currently 502
rows. The census arm reuses `search_trend` and the control arm's structure
(bounded per pass, ledger of what has been sampled, its own partition).

The binding constraint is control-arm volume: ~18 posts per window, 16 windows
per day, so ~290 posts/day to discover from, of which only 9.8% carry a hashtag
(49 of 502). Discovery power grows with the control arm, so this route gets
better for free as route 4 accumulates.

## What would make this stronger

Raising `CONTROL_WINDOWS_PER_PASS`. Windows are far under the cap (largest 71
against 100, 0 of 28 truncated), so the arm is cheap per window and the cap is
not the limit - the pass budget is. Doubling windows doubles both the
denominator and the discovery surface.

## Unresolved

1. Selection rule: emergence, volume, or concentration (recommendation above).
2. Should the discovery surface be control-arm only, or control plus baseline?
   Baseline is larger but keyword-biased, so a tag found there may be an
   artefact of our own list. Control-only is cleaner and slower.
3. Re-test `trends()` after a twscrape upgrade? If it ever works it is strictly
   better for discovery than sampling, because it sees the whole platform.

## Built and run live, 2026-09-15

`kenya_monitor.trend_discovery`, `trend_candidates/`, `posts/type=trend`, and a
cycle step behind the control arm (`TRENDS_EVERY_HOURS` 12).

First live pass, `run_trends_once(limit=2, per_tag=40)`:

```
trend discovery: 4 candidate(s) from 4 recent and 23 prior windows
trend: #lindamwananchinairobi emergence inf (36 posts, 14 authors, 2.6 per author) -> 43 collected
trend: #nairobinasifuna       emergence inf ( 8 posts,  8 authors, 1.0 per author) -> 42 collected
{'candidates': 4, 'censused': 2, 'posts': 85, 'authors': 55}
```

**The selection rule behaved as designed, and the separation it was built to
preserve showed up immediately.** Both tags emerged identically - no prior
windows, so emergence is infinite for both - and emergence alone chose them.
Author concentration, which is recorded and deliberately never used to select,
then separates them cleanly: 2.6 posts per author against 1.0.

That is the whole argument for the split. Had concentration done the selecting,
2.6 would be a property of the selection rule and worthless as evidence. Because
emergence selected, concentration is an independent observation about a tag that
timing found - which is what makes it usable.

One caution on reading it: at 4 recent windows the prior is thin, so "emergence
inf" currently means "not seen in the 23 prior windows", which is a weaker
statement than it will be once the control arm has more history. The number to
watch as the arm grows is how many candidates a pass returns; 4 from 4 windows
is a rate that will not hold.
