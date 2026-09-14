
## Built 2026-09-14

`kenya_monitor.control`, `monitor control`, `config/control_frame.yaml`,
`control_runs/`, and a fourth value in `kma.db.SCOPES`.

Decisions the implementation settled, with the argument for each:

- **The frame ships as the Kenyan-politics one** - institutions and offices
  (IEBC, Bunge, Parliament, Senate, National Assembly, Governor, Senator,
  county government, by-election, voter registration) conjoined with a Kenya
  anchor group. Not campaign language: a frame built from slogans
  over-represents slogan talk, which is the bias `targets.yaml` has and this
  arm exists to avoid. Swap it by editing the file, but a change starts a NEW
  frame and rates either side of it are not comparable.
- **Windows are aligned to the epoch, not to `now`.** Two passes minutes apart
  must see the same grid, or "already sampled" is unanswerable and each pass
  draws from a different population.
- **`SEARCH_PRODUCT` is already `Latest`.** The census claim needed
  reverse-chronological results rather than relevance ranking, and the
  collector has been set that way since long before this - so the design's
  central assumption was already satisfied rather than newly assumed.
- **Empty windows are recorded.** The denominator is sampled windows, so a
  window with no frame posts is an observation; dropping it would bias every
  rate upward.
- **Truncated windows are recorded, kept, and flagged.** A window at the cap is
  a ranked sample of an unknown larger set and the census claim does not cover
  it. Dropping them instead would bias the sample toward quiet windows, which
  is worse than knowing.
- **`all` still means what it meant.** The control arm is reachable only by
  naming the `control` scope. Adding a partition must not silently change what
  an existing number means.
- **In the cycle from the start**, unlike the depth arms. X search reaches 14
  days, so a one-off pass samples the last fortnight and nothing else; coverage
  of the collection period accumulates only by running continuously.

## The one number still to measure

Whether a 15-minute window of this frame fits under the 100-post cap. If it
does, each sampled window is a census and the arm works as designed. If it does
not, `truncated` will say so on the first pass and the window narrows until it
stops. That measurement needs the account pool, so it needs a deploy - it
cannot be taken from a laptop.

## First live passes, and the finding that changed the design (2026-09-14)

Twenty windows sampled, 249 posts, **0 truncated**. The census claim holds
comfortably: the busiest 15-minute window returned 39 posts against a cap of
100, so narrowing is not needed and the results are whole windows rather than
ranked prefixes.

But the first 20 windows showed something else. Windows 11-14 days old returned
1-3 posts; windows 2-7 days old returned 13-39. Time of day did not explain it -
a window at 18:15 Nairobi time returned 3 posts at age 8 and another at the same
hour returned 39 at age 4.

Measured directly by running the frame query over the SAME slot (12:00-12:15
UTC) at every age from 1 to 13 days, holding query, cap and time of day fixed:

| age (days) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| posts | 72 | 20 | 24 | 33 | 23 | 34 | 18 | **2** | 3 | 3 | 5 | 5 | 4 |

**A ~9x cliff between day 7 and day 8.** X search accepts queries back to 14
days and returns comparable results for about 7.

Two consequences.

**For the control arm**, `HORIZON_DAYS` is 7, not `collectors.x.MAX_AGE_DAYS`'s
14. Sampling uniformly over 14 days would have made every rate a function of how
old the sampled window happened to be - the exact confound the arm exists to
remove, arriving by a route nobody would have looked for. Window age stays
recoverable per row (`window_start` against `collected_at`) so an analysis can
condition on it rather than trust the boundary, and it should: even inside the
band, age 1 returned about twice what ages 2-7 did.

**For the collector at large, and this is the bigger one.** `backfill_windows`
sweeps windows from `SEARCH_RECENT_DAYS` out to 14 days on the first cycle of
each UTC day, and the baseline `search` partition is built from them. If the
same decay applies there - and there is no reason it would not, it is the same
endpoint with the same operators - then the older backfill windows have been
contributing roughly a tenth of the posts the recent ones do, and the baseline
corpus is far more recency-weighted than its window schedule suggests. That is
worth measuring against the corpus itself rather than assuming: the test is
posts-per-window-day by collection age in `posts/type=search`, which needs no
new collection at all.
