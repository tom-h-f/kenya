# CIB-oriented collection strategy

Keyword/account search samples *content*; coordination detection (phase 3)
needs dense sampling of *behaviour around shared objects* and *social graph*
structure. This doc explains the strategic rationale; the full command and
env reference is in [Collection reference](README.md).

Decisions (2026-07-08): full stack in one build; adaptive promotion
auto-applies with caps and expiry (never edits the curated `targets.yaml`);
follow-graph collection supports flagged clusters, suspicion-ranked accounts,
and recursive BFS crawl with per-account state.

## 1. Retweet-inclusive Latest search

X search excludes native retweets by default and ranks by relevance, which
hides exactly the low-engagement fresh accounts CIB campaigns use. Every
keyword query now:

- appends `include:nativeretweets` (env `SEARCH_INCLUDE_RETWEETS`, default on)
- requests the `Latest` product (env `SEARCH_PRODUCT`), chronological not ranked

Retweet rows carry real `created_at`, feeding the timed fast co-share channel.

## 2. Snowball / census slices

Per pass, for objects already observed:

| step | source | output |
|---|---|---|
| retweeter lists | top `SNOWBALL_TOP_RETWEETED` reposted objects (lookback `SNOWBALL_LOOKBACK_DAYS`) | `engagements/` rows (account x object incidence, kind=`retweet`) + authors |
| reply threads | top `SNOWBALL_TOP_CONVERSATIONS` conversations by reply_count | `posts/type=replies` |
| original hydration | referenced `repost_of_id`/`quoted_post_id`/`in_reply_to_id` not yet held | `posts/type=hydrated` |

`retweeters()` returns Users with **no retweet timestamp**: engagements are
incidence-only (untimed co-retweet channel, SVN). Timing comes from the
RT-inclusive search rows. The two complement; neither replaces the other.

A state file (`state/snowball.json`) TTLs each object (`SNOWBALL_REFRESH_HOURS`)
so hot objects are not re-fetched every pass. This makes the account x object
incidence a **census for the chosen objects** - the hypergeometric null in
`kma.coordination` then tests real incidence instead of sampling luck.

## 3. Adaptive target promotion (auto, capped)

`state/dynamic_targets.json` holds promoted targets; merged with the static
`targets.yaml` at run time, never written back to it.

- **Hashtags**: last-24h count >= `DYNAMIC_HASHTAG_MIN_COUNT` and >=
  `DYNAMIC_HASHTAG_RATIO` x the prior-7d daily average (or brand new). Top K
  promoted as keywords.
- **Accounts**: members of the latest persisted coordination clusters
  (`coordination/kind=clusters` on R2) promoted to timeline targets.
- Caps: `DYNAMIC_MAX_KEYWORDS` / `DYNAMIC_MAX_ACCOUNTS`. Expiry: entries not
  re-confirmed for `DYNAMIC_EXPIRY_DAYS` are dropped. All promotions logged
  with their source ("hashtag-burst", "coordination-cluster").

## 4. Always-on collection (2026-07-08)

`monitor run` is a continuous worker: cycles of posts -> snowball -> metrics ->
follow crawl run back to back, forever. There is no wall-clock gap between
cycles - the throttle is the per-account pacing plus twscrape's rate-limit
rotation (it waits when the whole pool is limited). A short randomized
cooldown (`CYCLE_COOLDOWN_MIN_S`..`CYCLE_COOLDOWN_MAX_S`, default 60-300s)
keeps the cadence organic between cycles.

Burst detection (hourly volume z >= `BURST_ZSCORE` with at least
`BURST_MIN_POSTS` posts vs the prior-48h baseline) skips the cooldown so the
next cycle starts immediately - fast co-share evidence lives at second
resolution during bursts. Backfill windows join the first cycle of each UTC
day; account-pool maintenance (yaml sync, relogin, lock reset) runs on its own
`ACCOUNT_SYNC_HOURS` timer.

## 5. Follower edges (targeted)

`monitor follows` fetches `followers()` + `following()` for selected accounts.
Edges land in `follows/` as `(follower_id, followed_id)`; discovered users
update `authors/`.

**Target selection:**

| Mode | Command | Use when |
|------|---------|----------|
| Coordination clusters | `monitor follows` | Default; members of latest persisted clusters |
| Explicit handles | `monitor follows --handle X` | Analyst-directed |
| Suspicion rank | `monitor follows --top-suspicious N` | Triage bot-like accounts from Phase 1 heuristic |

Caps: `FOLLOW_FETCH_LIMIT` per direction per account, `FOLLOW_MAX_ACCOUNTS` per
invocation. Enables follower-overlap corroboration in analysis.

## 6. Recursive follow crawl

`monitor crawl-follows` walks the graph BFS across runs:

1. Seed from handles, suspicion rank, and/or uncrawled accounts already seen
   in `follows/` edges.
2. Crawl each due account; enqueue neighbours from new edges.
3. Record crawl time per `platform_user_id` in `state/follow_crawl.json`.
4. Skip accounts crawled within `FOLLOW_CRAWL_REFRESH_DAYS`.

Designed for cron-style repetition: each pass crawls up to
`FOLLOW_CRAWL_MAX_PER_RUN` accounts and expands the frontier. Use
`monitor crawl-follows --status` for ledger summary and pending queue size.

## R2 prefixes (collection)

```
engagements/platform=x/dt=/run=.parquet   (platform_post_id, platform_user_id, kind, collected_at)
follows/platform=x/dt=/run=.parquet       (follower_id, followed_id, collected_at)
posts/platform=x/type=replies/...         (snowballed reply threads)
posts/platform=x/type=hydrated/...        (hydrated referenced originals)
```

Analysis side: `kma.db.latest_engagements` / `latest_follows`;
`kma.coordination.traces("co_retweet")` unions engagement incidence (untimed
rows, NULL created_at - excluded from timed variants automatically).

## Rate budget

All fetches go through the shared twscrape pool with per-account pacing and
LRU rotation. Caps are env-tunable and deliberately conservative; measure the
pool's sustained throughput before raising them - do not guess limits.

## Non-goals (this build)

TikTok/Facebook actors, archival beyond `MAX_AGE_DAYS`, WhatsApp. Tracked in
the roadmap, not here.
