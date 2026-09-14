# A control arm, so prevalence has a denominator

`docs/analysis/2026-09-13-publishable-statistics.md` finds that four things are
publishable today - the benchmark reproduction, the relevance gate's
human-measured error rate, what the detector surfaces as its own output, and the
negative results - and that prevalence is not, because there is no denominator:
every partition we hold was collected because something about it looked
interesting.

This is the design for the one thing that changes that.

## What is wrong with the denominators we have

`BASELINE_TYPES` exists to separate sampling-biased collection from the rest,
and it does that job. What it does not do, and was never claimed to do, is make
the baseline a sample of Kenyan discourse. `search` is 34 pre-chosen keywords;
`timeline` is 30-odd curated handles; `replies` and `hydrated` hang off those.
So a rate computed over the baseline is "the rate among posts matching our
keyword list", which is a statement about the keyword list.

Two further facts make the baseline unusable as a denominator even on its own
terms. Its composition moved from 70.7% search / 12.1% replies to 14.3% / 72.6%
on 2026-08-06 while replies carry 3.2x the hate rate, which is what made the
raw toxicity series unpublishable. And `min_faves` is applied to search, so the
frame is engagement-filtered as well as keyword-filtered.

## The constraint: X gives no random sample

There is no sampled stream and no `place_country` operator on this access path.
`lang:sw` is useless here - roughly 0% of Swahili posts carry the tag, which is
why `build_query` never emits it. The only levers are a query string and a
`since`/`until` window.

## The design: census a random window, rather than sample a ranked result

Search results are ranked, so the first K results of a broad query are not a
random K. But that is only a problem while the result set is larger than what
one pass can take. Narrow the window enough and the ranking stops mattering:
if a query over a 15-minute window returns fewer posts than the page cap, what
comes back is every post in the frame for that window - a census, not a sample.

So:

1. Fix a broad FRAME QUERY, pre-registered and never tuned on an outcome.
2. Pick windows uniformly at random from the collection period.
3. Take every frame post in each sampled window.
4. The denominator is "frame posts in sampled windows"; every rate is
   conditional on the frame, and the frame is stated beside the rate.

Randomness enters through the window, not through the results, so the sample is
exogenous to the detector in the way `census_discovered_handles`' `ORDER BY
random()` is - which is the precedent and the argument.

The one number this design rests on is whether a frame query over a short
window actually fits under the page cap. That is measurable before anything is
built: run the candidate frame query over a handful of 15-minute windows at
different times of day and count. If it does not fit, the window narrows until
it does, and if that runs out of road the design falls back to an account-level
frame (below).

## The decision that is not mine to make: what the frame is

Three candidate frames, in decreasing breadth and increasing defensibility of
"Kenyan":

- **Kenyan-discourse frame.** A small OR-group of high-frequency Kenyan Swahili
  and Sheng function words. Broadest, closest to "Kenyan discourse", and the
  hardest to fit under a page cap. Says nothing about politics, so political
  prevalence becomes a rate WITHIN it, which is the strongest version of the
  claim.
- **Kenyan-politics frame.** A pre-registered OR-group of institution and office
  words that are not campaign slogans: IEBC, "Parliament", "Senate", county
  names. Narrower, fits a window more easily, and the denominator is "Kenyan
  political talk" rather than "Kenyan talk".
- **Account frame.** Sample accounts at random from the authors already
  observed, and take a bounded slice of each timeline. The denominator becomes
  accounts rather than posts, so the publishable claim changes shape: "X% of
  accounts that entered our corpus posted Y", not "X% of posts are Y".

Recommendation: the Kenyan-politics frame. It is the one whose denominator
matches the claims this project actually wants to make, it is the most likely
to fit a window census, and its terms can be fixed in advance without being
tuned - which is the property that makes the whole thing work.

## Provenance

Rows land in `posts/type=control`, its OWN partition, in neither
`BASELINE_TYPES` nor `TARGETED_TYPES`. Not baseline, because mixing it into the
existing denominator would repeat the 2026-08-06 composition break that made the
toxicity series unpublishable. Not targeted, because it is the opposite of
targeted: nothing about a post decides whether it is collected except the window
its timestamp falls in. `db.SCOPES` gains a third value and every scope-aware
reader has to name which one it means, which is the point - a control arm that
can be silently unioned into a denominator is not a control arm.

Beside the posts, one row per sampled window recording the frame query, the
window bounds, the cap, and how many posts came back. Without that a later
reader cannot tell a quiet window from a truncated one, and the difference is
the whole validity of the census claim.

## What this does NOT buy

It does not make the detector's output a prevalence estimate. v2 ranks accounts
within the corpus it is given; running it over the control arm would measure
something else. What the control arm supports is rates of measurable ATTRIBUTES -
toxicity, relevance, near-duplication - computed on a frame with a known
sampling probability. Any coordination claim still rests on the detector, and
still travels with the detector's caveats.
