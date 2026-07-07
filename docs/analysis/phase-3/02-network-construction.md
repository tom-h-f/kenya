# 02 - Network construction

Turn traces into a weighted account-account graph per channel.

## Bipartite -> projection

For a channel, the bipartite graph `B` links accounts to action objects (an
account is connected to every object it acted on). The account-account
projection connects two accounts with weight = number of objects they both acted
on (optionally time-constrained). This is done efficiently in DuckDB by
self-joining the trace table on `action_object`, never materialising the full
account x account matrix:

```sql
-- generic co-action projection (one channel)
WITH traces AS ( SELECT author_id, action_object, created_at FROM ... )
SELECT a.author_id AS src, b.author_id AS dst,
       count(*) AS weight,
       min(abs(epoch(a.created_at) - epoch(b.created_at))) AS min_gap
FROM traces a
JOIN traces b
  ON a.action_object = b.action_object
 AND a.author_id < b.author_id            -- unordered pairs, no self
 AND abs(epoch(a.created_at) - epoch(b.created_at)) <= :delta   -- omit for untimed
GROUP BY 1, 2
HAVING count(*) >= :min_repetition
```

`min_repetition` >= 2 (CooRTweet default): a single shared action is not
coordination. `delta` is the co-action window in seconds.

## Channel definitions

- **co-retweet.** `action_object = repost_of_id`. Untimed variant (shared
  amplification targets) and timed variant = **fast co-share** (`delta` small,
  e.g. 10-60s; CooRTweet default 10s). Fast co-retweet is the strongest classic
  signal.
- **co-reply.** `action_object = in_reply_to_id`. Accounts swarming the same
  target tweet.
- **co-hashtag** (Wave B). Unnest `hashtags[]` -> `action_object = hashtag`.
  Also compute the **hashtag-sequence** variant (Pacheco): per-post ordered tag
  multiset as the object, catching scripted identical tag sets.
- **co-URL** (Wave B). Unnest normalised `urls[]`.
- **co-mention** (Wave B). Unnest `mentions[]`.
- **text-similarity** (our advantage). Two sub-methods:
  1. *Near-duplicate clusters*: group posts with pairwise cosine >= `tau` (e.g.
     0.9) into content objects (connected components over the kNN graph from
     DuckDB `vss`), then project accounts sharing a cluster. `tau` measured.
  2. *Synchronised semantic co-post*: account pair whose posts have cosine >=
     `tau` within `delta`. More expensive; restrict to candidate pairs from the
     kNN graph, not all pairs.
- **handle/image.** Static sockpuppet signal: near-identical handles (edit
  distance / shared numeric-suffix pattern) or identical `profile_image_url`.
  Feeds characterization (05) more than the multiplex.

## Edge weighting

Two weightings, both retained:

- **Raw co-count** (default, feeds SVN which models counts directly).
- **TF-IDF cosine** (Pacheco co-retweet): represent each account as a TF-IDF
  vector over action objects (rare shared objects weigh more than popular ones),
  edge = cosine similarity. Downweights trivially-popular objects (e.g. a viral
  tweet everyone retweets) that inflate raw co-counts.

## Outputs

Per channel: a weighted edge list `(src, dst, weight, min_gap, n_objects)` plus
per-account activity degree (needed by the null model in 03). Persist to the
`coordination/` prefix (07).

## Scaling

Projection cost is dominated by popular objects: a tweet retweeted by N accounts
yields O(N^2) pairs. Mitigate by (a) capping/█downweighting objects with very
very high account degree (they carry little coordination signal - TF-IDF does this
naturally), (b) doing the self-join in DuckDB, (c) the `min_repetition` HAVING
filter early. Report pair counts before/after filtering; if a single object
explodes, exclude it and note it.
