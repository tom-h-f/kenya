# Route 1: record the deletions we are already observing

## The argument

The blind reader has declined to call anything an influence operation, with the
same rationale every time: *real-looking aged accounts, diverse avatars, no
concealment of origin*. We measure co-action, which separates "together" from
"not together" but cannot separate an open campaign from a covert one. What
separates them is concealment, and deletion is the concealment signal this
project can actually reach.

Kenya's documented disinformation-for-hire pattern is: drive a hashtag into
trending, get paid, delete. A post we hold that later returns nothing is
evidence no amount of graph sparsity can weaken - unlike centrality, it does not
degrade when we know little about an account.

## We are already collecting this and throwing it away

`runner.collect_metrics` re-fetches the top 5% of posts by engagement from the
last 5 days, ~400 per pass, several times a day. `XCollector.refresh_metrics`
then does this:

```python
tw = await self.api.tweet_details(int(pid))
if tw is None:
    continue
```

The disappearance is observed and discarded at the point of collection. Nothing
downstream can distinguish "we never checked" from "we checked and it was gone".

This is the cheapest high-value change available: the collection path exists,
the cadence exists, and the requests are already being paid for.

## Absence is not deletion

`tweet_details` returning None conflates four things:

1. the post was deleted
2. the author was suspended
3. the author went protected
4. our own request failed (rate limit, proxy, transport)

A status column that silently means all four would be worse than no column,
because it would look like evidence. This is the same ambiguity `deep_timelines`
has with `no_posts`, and it was resolved there by recording the outcome and
refusing to infer the cause.

**Resolution rule.** Record `absent` on the post. Then check the AUTHOR:

- author resolves and still posts, post gone -> **deletion**
- author no longer resolves -> **account suspended or deleted**, a different and
  also interesting signal
- our request errored -> retry, bounded, and never write absence on a raised
  request

The author check costs one request per absent post. At the base rate this is
small, but it is a real cost and it is the open question below.

## Design

1. `MetricSnapshot` gains `status` (`present` | `absent`) and, where resolved,
   `absence_cause` (`post_deleted` | `author_gone` | `unresolved`).
2. `refresh_metrics` yields a row either way, so absence becomes data.
3. Bounded retry before recording absence, so a rate-limit blip is never written
   as a deletion.
4. Author resolution as a second, separately bounded step - it must not be able
   to starve the metrics pass.
5. Deletion rate per account, joined on `author_id`, as a feature.

## What to measure, in order

1. **Base rate first.** What share of held posts vanish within 7 days,
   corpus-wide? Every per-account number is unreadable without it.
   *(Probe running 2026-09-15: two age bands, 2-4 days and 10-20 days, 60 posts
   each, with author resolution on a subsample. Numbers land here.)*
2. **By partition** - `baseline`, `targeted`, `control`. The control arm is what
   makes this a comparison rather than a number: it gives the deletion rate
   among ordinary Kenyan political posts, sampled exogenously.
3. **Surfaced accounts against the control arm.** This is the publishable form:
   "accounts the detector surfaced delete at N times the rate of ordinary Kenyan
   political accounts", with the interval.

## The bias that must be stated on publication

`collect_metrics` samples the **top 5% by engagement**. Any deletion rate
computed from it is the rate among high-engagement posts, not among posts. That
may well be the interesting population - a paid campaign's posts are meant to be
seen - but it is not the corpus, and quoting it as a corpus rate would be the
same error as the raw toxicity series.

**The clean fix is the control arm**: it is sampled by time, not by engagement,
so a deletion rate computed over control-arm posts has no engagement bias at
all. That makes route 4 a dependency for the publishable version of this, though
not for the feature.

## Why this route does not depend on the ranking

Everything else in the programme waits on the `MIN_ENTITIES` sweep, because
until the floor is right no ranking is trustworthy. Deletion rate is an
account-level property that needs no graph, no centrality and no ranking. **It
can run in parallel with the sweep**, and it is the only one of the four routes
that can.

## Unresolved

1. Is one extra request per absent post acceptable to resolve deleted from
   suspended, or should absence stay unresolved and be reported with the
   ambiguity stated?
2. Should the deletion check be extended beyond the top 5% by engagement -
   e.g. a random sample of held posts per pass - to remove the engagement bias
   at source rather than relying on the control arm?
3. Retention: deletions are only observable if we re-check before the post ages
   out of anything we can query. Is there a re-check schedule per post (once at
   24h, once at 7d), or is the existing top-5% cadence enough?
