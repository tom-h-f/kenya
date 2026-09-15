# Route 4: turn the control arm from a denominator into a test

## The argument

The control arm was built to give prevalence a denominator. It is also a **null
distribution**, and that is the more immediately useful thing.

Every output this project produces is currently a ranking: these 500 accounts
scored highest. A ranking cannot say whether the top account is unusual, only
that it is top. With an exogenous sample of ordinary Kenyan political discourse,
any account-level feature becomes a percentile against normal - which turns
"scored highest" into "above the 99th percentile of Kenyan political accounts",
a claim that survives the corpus changing shape underneath it.

## What exists

Live since 2026-09-14, sampling itself every 6 hours without intervention.

| | |
|---|---|
| windows sampled | 28 |
| posts | 502 |
| authors | 410 |
| truncated windows | **0** |
| posts per window | min 0, median 14, mean 17.9, max 71 |
| empty windows | 1 |
| window age range | 1.4 - 13.7 days |

0 of 28 truncated against a cap of 100 means every sampled window is a census
rather than a ranked prefix - the property the whole design rests on. The oldest
windows predate the 7-day horizon fix and should be filtered by age at analysis
time, which is why `window_start` and `collected_at` are both recorded.

## What it can be a null for

Any feature computable on both the target set and the control arm:

- **Deletion rate** (route 1). The strongest pairing: "surfaced accounts delete
  at N times the rate of ordinary Kenyan political accounts" is a concealment
  claim with a real denominator.
- **Near-duplicate rate.** Already computable from the text trace machinery.
- **Posting regularity**, account age at first activity, follower/following
  ratio - the ordinary inauthenticity features, but with a defensible baseline
  for the first time.
- **Toxicity and relevance**, which are the two classifiers already in
  production.

## The arithmetic that has to happen before any claim

At ~18 posts per window and 16 windows per day, the arm accumulates ~290
posts/day from ~230 distinct authors. Quoting a 99th percentile needs enough
authors that the tail is populated: **the required sample size is a calculation,
not a guess, and it should be done before the first comparison rather than
after.** A percentile quoted from 410 authors has a wide interval and the
interval must be published with it.

Two cheap ways to accelerate, both worth costing:

- Raise `CONTROL_WINDOWS_PER_PASS` from 4. Windows are far under the cap, so the
  arm is cheap per window and the pass budget is the only limit.
- Widen the window from 15 minutes. The busiest window returned 71 of a 100
  cap, so 30 minutes would still census most windows while doubling the time
  sampled per request. **Measure first**: a wider window truncates more often,
  and a truncated window stops supporting the census claim.

## The caveat that travels with every use

The frame is **institutional Kenyan politics** - IEBC, Bunge, Parliament,
Senate, National Assembly, Governor, Senator, county government, by-election,
voter registration, conjoined with a Kenya anchor group. It is not "Kenyan
discourse". Every rate computed against it is conditional on that frame, and the
frame is stated beside the rate or the number is misleading.

Changing `control_frame.yaml` starts a NEW frame and rates either side are not
comparable. If the frame is ever widened, the old and new arms are two
populations, not one longer series.

## Unresolved

1. Required sample size for the percentile claims we actually want to make -
   this needs deciding before comparisons start.
2. Raise windows per pass, widen the window, or both? Widening needs a
   truncation measurement first.
3. Should a second frame run in parallel - a broader Kenyan-discourse frame
   alongside the institutional one? It would let political prevalence be a rate
   WITHIN Kenyan discourse, which is the strongest version of the claim, at the
   cost of a second arm and a harder census.
