# Finding campaigns: four routes from co-action to concealment

## Why these four

The detector works. It reproduces 6 of 6 IOHunter benchmark datasets within two
published SD, and on this corpus it surfaces real coordination: the Sifuna
campaign with matching hashtags, opposition amplification, IEBC and election
content, an Ol Kalou by-election group.

It has also never called anything an influence operation, and the blind reader's
rationale is the same every time: **real-looking aged accounts, diverse avatars,
no concealment of origin.**

That is the reader telling us what we are missing. We measure *co-action* -
accounts doing the same thing at the same time. Co-action separates "together"
from "not together"; it cannot separate a campaign from an operation, because an
open campaign is also people acting together. What separates them is
**concealment**, and we collect almost no evidence of it.

Two further facts set the order below:

- **Search coverage collapses ~9x past day 7** (measured 2026-09-14: 72, 20, 24,
  33, 23, 34, 18 posts per window at ages 1-7 days, against 2, 3, 3, 5, 5, 4 at
  8-13). Anything not collected while it is live is effectively unrecoverable.
  Routes that catch campaigns *as they happen* beat routes that look back.
- **The trace best matched to Kenya's documented threat is the one we feed
  least.** `hashtag_sequence` has 459 users and 3,992 edges against
  `co_retweet`'s 13,859 and 513,627. The implementation is correct; the evidence
  is missing.

Ordered by value over cost. Routes 1-3 are independent and can run in parallel;
route 4 is downstream of the control arm accumulating.

Each route has its own plan, and the reconnaissance behind them was measured on
2026-09-15 rather than assumed:

| route | plan | status after measurement |
|---|---|---|
| 1. Deletions | `2026-09-15-deletion-tracking.md` | **cheapest**; the signal is already collected and discarded |
| 2. Trends | `2026-09-15-trend-collection.md` | **redesigned**; original premise measured dead, replacement already produced a result |
| 3. Windowed detection | `2026-09-15-windowed-detection.md` | **viable**; daily windowing holds 701 accounts |
| 4. Control as null | `2026-09-15-control-as-null.md` | accumulating; 28 windows, 0 truncated |

## What the reconnaissance changed

**Route 2's premise was wrong and is now replaced.** twscrape's `trends()`
fails against the live pool - `GenericTimelineById` returns `(-1) Internal
server error` under HTTP 200, for both `trending` and `news`. That is X
rejecting the operation, not a locale problem, so no change of account fixes it.

The replacement is better. The control arm samples Kenyan political discourse
exogenously, and hashtag frequency over it discovers campaign tags directly. It
already has: in 502 randomly sampled posts, `#LindaMwananchiNairobi` appears 24
times from 11 authors (4.8% of the sample), and at least 2 of the 7 repeated
tags - `#KaNairoTunamuOk`, `#NairobiNaFei` - are outside our 34-term keyword
list. Discovery needs no new endpoint and gets better for free as route 4
accumulates.

**Route 3 is viable, which was not obvious.** The fear was that daily windows
would leave too few accounts above the activity floor. Measured over the last 60
days of snapshot `2026-09-14-deep500`:

| window | >= 2 entities | >= 5 | >= 10 |
|---|---|---|---|
| per day | **701** | 310 | 179 |
| per week | 2,855 | 1,397 | 941 |

Daily holds at floors 2 and 5; floor 10 needs weekly. That sets the
window/floor trade concretely rather than by argument.

**Route 1 is cheaper than it looked.** `XCollector.refresh_metrics` already
calls `tweet_details` on ~400 held posts several times a day and does
`if tw is None: continue`. We are observing disappearances and discarding them.

---

## Route 1: track deletions

**The strongest concealment signal available, and we are currently discarding
it at the point of collection.**

Kenya's documented disinformation-for-hire pattern is: drive a hashtag into
trending, get paid, delete. A post we hold that later returns nothing is
evidence no amount of graph sparsity can make ambiguous - unlike centrality, it
does not degrade when we know little about an account.

### What already exists

`runner.collect_metrics` re-fetches the top 5% of posts by engagement from the
last 5 days, ~400 per pass, several times a day. So we already revisit posts on
a cadence. `XCollector.refresh_metrics` then does this:

```python
tw = await self.api.tweet_details(int(pid))
if tw is None:
    continue
```

The disappearance is observed and thrown away. Nothing downstream can tell "we
did not check" from "we checked and it was gone".

### The design

1. `MetricSnapshot` gains a status: present, or absent. `refresh_metrics` yields
   a row either way, so absence becomes data rather than a gap.
2. **Absence is not deletion, and the schema must not pretend otherwise.**
   `tweet_details` returning None conflates: post deleted, author suspended,
   author went protected, and our own request failing. This is the same
   ambiguity `deep_timelines` has with `no_posts`, and it was resolved there by
   recording the outcome and refusing to infer. Do the same: record `absent`,
   and resolve the cause separately by checking the AUTHOR. An author who still
   resolves and still posts, whose post is gone, is a deletion. An author who no
   longer resolves is a suspension, which is a different and also interesting
   signal.
3. A retry before recording absence, bounded, so a rate-limit blip is never
   written as a deletion.
4. Deletion rate per account becomes a feature, joined on `author_id`.

### What to measure, in order

- **Base rate first.** What share of held posts vanish within 7 days, across the
  whole corpus? Without that number, any per-account rate is unreadable.
- Deletion rate by partition (`baseline`, `targeted`, `control`). The control arm
  is what makes this a real comparison: it gives the rate among ordinary Kenyan
  political posts, sampled exogenously.
- Deletion rate of accounts the detector surfaced, against the control arm rate.

### Cost and risk

Days, not weeks - the collection path exists and the cadence exists. The main
risk is the ambiguity above: a status column that silently means four things
would be worse than no column, because it would look like evidence.

**The bias to state when publishing:** `collect_metrics` samples the top 5% by
engagement, so any deletion rate computed from it is the rate among
*high-engagement* posts, not among posts. That may well be the interesting
population, but it is not the corpus.

---

## Route 2: collect trends, not just terms

**We sample 34 fixed keywords. A manufactured campaign is, by construction, not
on that list.**

### What already exists

twscrape carries `trends(trend_id)` and `search_trend(q)`. No new scraping
infrastructure is needed, and `search_trend` sets `querySource=trend_click`,
which is how a real client reaches a trend.

### The design

A bounded per-cycle arm, shaped like the control arm because the same
discipline applies:

1. Pull the current trend list each cycle. Record **every** trend seen, with its
   timestamp and volume, to a `trends/` prefix - including ones we do not
   collect. The series of what trended is itself a finding, and it cannot be
   reconstructed later.
2. Select trends to census. The selection rule has to be stated and fixed in
   advance, or this becomes another keyword list chosen on outcomes.
3. Collect posts per selected trend into `posts/type=trend`, a TARGETED
   partition. Selection is conditioned on the platform's own amplification, so
   it does not belong in a prevalence denominator.
4. Re-check trends after they fall out of the list. A trend that disappears
   faster than its volume predicts, or whose posts delete (route 1), is the
   documented attack shape.

### The open problem, and it decides whether this works

`trends()` returns the trend list **as the querying account sees it**, which is
geo- and interest-personalised. Our pool accounts are of unknown and probably
non-Kenyan locale - the login errors show Turkish and other addresses. So this
may return trends for the wrong country.

Test before building: call `trends("trending")` from several pool accounts and
see what comes back. If it is not Kenyan, the options are to find a Kenyan
`trend_id` (the API takes an arbitrary timeline id, so a location-specific one
may exist), or to set the account locale, or to abandon the arm. **One
afternoon's measurement decides this, and it should happen before any code.**

### What this also fixes

`hashtag_sequence` is starved at 459 users. Trend collection is the natural
source of hashtag-bearing posts, so this route feeds the trace best matched to
the documented threat.

---

## Route 3: make detection time-resolved

**Campaigns are events. The current graph is a three-month aggregate, so a
six-hour campaign is diluted into invisibility.**

This is also the route that most directly attacks the sparsity pathology the
2026-09-14 depth test exposed. An account ranked on two lifetime observations is
untrustworthy; an account with twenty co-actions inside one hour is a dense
local signal, and density is exactly what deepening destroyed in the global
ranking.

### The design

`coord2_run` already takes `--days`, so windowed runs need no new detector.

1. Run the same five traces over daily (and weekly) windows rather than the
   whole snapshot.
2. Rank on **burstiness** - concentration of an account's co-action in time -
   rather than lifetime centrality. An account whose co-retweets are spread over
   three months is a habitual retweeter; one whose co-retweets fall inside two
   hours with the same partners is the thing we are looking for.
3. Persist per-window scores so a campaign has a start and an end, which is what
   makes it reportable as an event rather than a list of accounts.

### What to measure

- Does a windowed ranking survive the depth test that the global one failed?
  Re-run `01_rank_check.py` against windowed scores using the existing holdout.
  This is the honest test and it is cheap, because the holdout already exists.
- How many accounts clear `MIN_ENTITIES` inside a single day? This may be the
  binding constraint - the corpus is wide and thin, and a daily window makes it
  thinner. If the answer is "almost none", weekly is the floor and that is a
  finding about what this collection can support.

### Sequencing note

Do the `MIN_ENTITIES` sweep first (5, 10, 20, reporting holdout retention at
each). It is a prerequisite for trusting any ranking, windowed or not, and it is
one number per run.

---

## Route 4: use the control arm as a null model

**Now that collection is exogenous, "unusual" can mean unusual relative to
normal Kenyan political discourse, rather than central in a sparse graph.**

The control arm went live 2026-09-14 and samples 15-minute windows of a
pre-registered institutional frame at random, censusing each whole. It was built
as a denominator for prevalence. It is also a null distribution.

### The design

For any account-level or post-level feature - deletion rate (route 1), posting
regularity, account age at first activity, near-duplicate rate, toxicity - the
control arm gives the distribution among ordinary accounts in the same
discourse. A surfaced account is then reportable as a percentile against that
distribution rather than as a rank in a list.

This turns the output from a ranking into a **test**, which is the difference
between "these 500 accounts scored highest" and "this account's deletion rate is
above the 99th percentile of Kenyan political accounts".

### What has to happen first

Accumulation. 28 windows and ~400 posts as of 2026-09-15, growing 4 windows per
6-hour cycle inside a rolling 7-day horizon. Sample-size arithmetic before any
claim: the number of windows needed for a percentile estimate at the precision
we want to quote is a calculation, not a guess, and it should be done before the
first comparison rather than after.

**One caveat that must travel with every use of it.** The frame is
institutional-Kenyan-politics, not "Kenyan discourse". A rate computed against it
is conditional on that frame, and the frame is stated beside the rate.

---

## What none of this buys

None of these routes will manufacture an influence operation if there is not one
here. The election is August 2027 and this is September 2026; Kenya's documented
campaigns cluster around electoral events, and we are between them.

**Finding nothing in 2026 is a legitimate result**, and the honest published
version of it - a benchmark-validated detector, pointed at Kenyan election
discourse for three months, surfacing open campaigning and engagement pods
rather than influence operations - is more defensible than most of this field.
The work above is what makes the detector calibrated and pointed at the right
phenomenon when a campaign does start.

---

## Unresolved questions

Questions 1 and 4 from the first draft are answered above and removed. What is
left is decisions, not measurements - each is in its own plan too.

1. **Trend selection rule** (route 2). Emergence, volume floor, or author
   concentration? Recommendation: emergence, with concentration recorded beside
   it but not used to select, so concentration stays usable as evidence rather
   than becoming the selection criterion.
2. **Absent-vs-deleted resolution** (route 1) costs one request per absent post
   to check the author. Acceptable, or should absence stay unresolved and be
   reported with the ambiguity stated?
3. **Window width** (route 3). Daily at floor 2-5, or weekly at floor 10? The
   population table above is the trade; the `MIN_ENTITIES` sweep decides which
   floor is defensible.
4. **Order of work.** Route 1 is the only one that needs no ranking at all, so
   it can run in parallel with the `MIN_ENTITIES` sweep. Routes 2-4 are all
   improved by the control arm growing, which argues for raising
   `CONTROL_WINDOWS_PER_PASS` early since it is a one-line change that makes
   three routes better.
