# Explorer data inventory

Measured 2026-10-10 from R2 (listings, parquet footers, windowed projected reads;
no `SELECT *`). Window for "21 d" figures: collection days 2026-09-19 to 2026-10-09,
the daily run `daily-2026-10-09-21d` (run `20261010T182001Z`). Nothing here is an
account id or handle. Row counts are collector rows unless "distinct" is stated.

## Corpus and freshness

| Prefix | Objects / size | dt range | Newest write | Grain | Notes |
|---|---|---|---|---|---|
| `posts/` | 8,144 / 908 MB | 07-16 .. 10-10 | 10-10 17:33 | one row per collected snapshot of a post | 30 d (09-10..10-10): 2,541,663 rows. 21 d window: 1,563,308 rows, 774,836 distinct posts, 327,223 distinct authors |
| `authors/` | 9,227 / 1.53 GB | 07-16 .. 10-10 | 10-10 17:33 | profile snapshot | latest per id: 3,340,811 authors; handle 100%, bio 68.7%, followers 100%, created_at 100%, location 47.3% |
| `embeddings/` | 1,739 / 5.08 GB | 07-16 .. 10-10 | 10-10 16:35 | one vector per post | 1,542,603 rows, 768-d, ~887 rows per object |
| `relevance/` | 41 / 46 MB | 09-12 .. 10-10 | 10-10 14:49 | `p_kenya` per post | promoted model `-09-13`: 2,480,344 posts; 32.7% have p_kenya >= 0.5 (of scored posts, not a prevalence) |
| `hatespeech/` | 1,103 / 43 MB | 07-25 .. 10-10 | 10-10 18:07 | label, `p_hate`, `hate_flag`, `flagged`, `coded_suspect` per post | 1,950,193 rows over 1,057,573 posts (re-scored rows duplicate). hate_flag 0.45%, flagged 0.48%, coded_suspect 0.03%, label != neither 3.8%, all over a corpus that is not a sample |
| `incitement/` | 515 / 25 MB | 07-17 .. 09-28 | 09-28 | NLI scores per post | 1,011,691 posts. Stale: nothing written for 12 days |
| `labels/` | 1,711 / 16 MB | 07-16 .. 10-10 | 10-10 17:02 | sentiment, emotion per post | 811,313 posts |
| `topics/` | 1 / 6 MB | 09-04 only | 09-04 | topic id per post | see Topics |
| `engagements/` | 6,547 / 114 MB | 07-16 .. 10-10 | 10-10 15:53 | (object, user, kind=retweet), incidence only | 30 d: 4,229,594 edges, 22,121 objects, 830,335 users. No timestamp, no tombstone |
| `follows/` | 5,609 / 49 MB | 07-16 .. 10-10 | 10-10 10:58 | (follower, followed) ever observed | all: 4,157,920 edges, 951,562 followers, 970,253 followed. 30 d: 1,413,828 |
| `metrics/` | 784 / 10 MB | 07-16 .. 10-10 | 10-10 16:50 | re-check snapshots | `status` column since 09-21, see Deletions |
| `deep_timelines/` | 200 / 0.7 MB | 09-08 .. 10-08 | 10-08 | per account deepened | ok 977 accounts (259,924 posts written), holdout 97, no_posts 13 |
| `control_runs/` + `posts/type=control` | 77 / 0.3 MB; 77 / 5 MB | 09-14 .. 10-10 | 10-10 | one row per 15-min window | 1,036 runs since 08-31, 29,213 posts, 20 truncated |
| `trend_candidates/` | 45 / 0.1 MB | 09-15 .. 10-10 | 10-10 17:04 | per tag per run | 337 rows, 45 runs, 118 distinct tags; latest run 21 tags, 3 censused |
| `census_runs/`, `census_ttl/`, `collection_runs/` | 551, 234, 115 | | collection_runs last 09-01 | collector counters | collection_runs stopped 09-01 |
| `coord2/kind=daily_*` | 1 object per kind | 10-10 | 10-10 | see Networks | two runs, both written 10-10 (30 d and 21 d); history starts now |
| `coord2/kind=scores` | 6 | 09-08 .. 10-10 | 10-10 | one row per listed account | 306 accounts / 16 communities (21 d), 357 / 21 (30 d) |
| `coord2/kind=textsim` | 10 | | 10-10 | user pairs, `bucket=day` variant | window 21 d: 1,631 pairs before the activity floor, 1,266 after |
| `coordination/` (v1) | 2,049 / 861 MB | to 10-10 | 10-10 10:47 | v1 clusters and edges | v1 retired 10-10. Not used by the explorer |
| `leads/`, `canary/` | do not exist | | | | no canary or lead artifact is in the bucket |

Embedding, relevance and incitement coverage of the window is not computed anywhere
persisted. It must be built (distinct posts in window joined to each prefix).

## Networks and edges

What the daily persists: `daily_runs` (1 row), `daily_accounts` (listed accounts),
`daily_communities` (every community of 4+ accounts), `daily_members` (every member).
Measured on the 21 d run:

| | 30 d run | 21 d run |
|---|---|---|
| accounts in graph | 14,938 | 10,941 |
| fused edges | 685,065 | 459,744 (measured by rebuilding: 109 edges carry more than one trace) |
| communities of 4+ / kept at floor 0.5 | 96 / 21 | 46 / 16 |
| accounts in kept communities | not measured | 10,179 (members persisted: 10,495 across all 46) |
| listed accounts | 357 | 306 |
| edges by trace | co_retweet 676,829; co_url 1,615; hashtag_sequence 2,085; fast_retweet 21; text_similarity 4,701 | 456,780; 551; 1,242; 15; 1,266 |
| networks / relevance / communities, seconds | 509 / 92 / 12 | 325 / 61 / 5 |

**Not persisted, must be built: the edge lists.** The daily keeps accounts,
communities and members, not who is linked to whom, not the trace rows, not the
evidence. Two ways to get them: (a) persist them in `v2_daily.rank` where the graph
already sits in memory (`daily_edges`, `daily_traces`; adds a few MB of parquet and no
recompute); (b) rebuild in the exporter. (b) was measured: 346 s and ~2 GB RSS on a
14-core Mac for `snapshot_networks` over the 21 d window, then 2.8 s for Leiden. It
reproduces the persisted membership exactly (label match 1.0, 10,495 of 10,495), so
the rebuild is deterministic. Recommendation (a): do not pay 346 s a second time.

Text-similarity evidence is also missing: the persisted trace has only (user pair,
day bucket, mean similarity), no post ids. For 1,266 edges the exporter recomputes the
post pair from embeddings of those users' posts.

Edge basis (21 d, kept communities, measured):

- intra-community co_retweet edges: 256,691. Share of retweeted posts both accounts
  retweeted: 100% >= 1, 7.4% (19,095) >= 2, 0.9% (2,207) >= 3, 0.1% (287) >= 5.
- Cross-community fused edges: 199,359, all between kept communities, aggregating to
  111 community pairs.
- Rule "shared >= 2 or any non-co_retweet trace": 42,445 edges over 7,547 nodes (21,385
  inside a kept community, 20,461 between kept communities). Rule "shared >= 3 or other":
  7,299 edges over 3,250 nodes.
- Top-5-per-node backbone inside kept communities: 37,422 edges over 10,179 nodes;
  77% of its edges rest on exactly one shared retweet.
- Per community, share of members covered by the 10 most retweeted shared posts: 0.18 to
  0.99 across the 11 large communities, so a few communities are one-or-two-post
  blocks and others are diffuse.
- Trace rows feeding the bipartite graphs (21 d, all users): co_retweet 219,302 rows /
  32,890 users / 121,658 posts; co_url 37,009 / 13,654 / 28,749 URLs; hashtag_sequence
  9,066 / 4,967 / 5,741 sequences; fast_retweet 477 / 322 / 292.

Questions answered: who is linked to whom and by what; which communities touch;
which posts, URLs or tag sequences bind two accounts (via evidence). Not answered:
direction or timing of co-retweets (`engagements/` has no time), why an account acted.

## Topics

`topics/` is one object written 2026-09-04: 663,294 posts, 1,592 topics plus noise (id -1)
holding 56.4% of rows (374,278), UMAP + HDBSCAN (`min_cluster_size=60`, `n_neighbors=15`),
593,035 distinct posts by approximate count. Per post, not time-sliced, not refreshed. It
covers posts collected to about the fit date, so none of the 21 d window except posts
already held. The fit costs hours of CPU and tens of GB (documented 2026-09-03), so it
does not run on `cloud`. Term labels are not persisted: they come from `topic_summary`
over post text and must be rebuilt (c-TF-IDF, cheap). New posts can be placed by nearest
topic centroid from embeddings, which is cheap in principle but its cost and quality are
not measured here; it needs a precision check on labelled examples before it ships.

## Hashtags and URLs

No extraction path exists beyond the persisted `hashtags` and `urls` list columns on
`posts/` (X entity list, `urls` is X's own expanded URL, third-party shorteners not
resolved). Trace code already reads them (`coord2.hashtag_sequence_traces`). Measured on
the 21 d window, distinct latest posts 774,836:

- with at least one hashtag: 57,427 (7.4%); distinct tags (lower-cased): 21,481.
- distinct URLs 28,749; distinct domains 5,618. Top domains: x.com 8,739, zurl.co 3,136,
  standardmedia.co.ke 1,682, youtu.be 1,630, citizen.digital 1,516.
- top tags are dominated by campaign-style tags (largest 8,159 posts by 2,445 accounts),
  not a measure of discourse share: the corpus is a targeted collection.

`trend_candidates/` gives emerging tags with `recent_rate`, `prior_rate`, `emergence`,
`censused` per run, rates from the control arm.

## Account profiles

`authors/` carries handle, display name, bio, location, followers, following, tweets,
listed, verified, blue, created_at, profile image URL; latest snapshot per id with
`collected_at`. Export needs only the ~10.5k members, not 3.3M authors.

## Deletions

`metrics/status` has rows since 2026-09-21. Present: top arm 49,422 rows (2,113 posts),
random arm 6,051 (6,024), unlabelled 2,420. Absent: author_gone top 350 rows (34 posts),
random 47 (47), unlabelled 26; post_deleted top 4 posts, random 2, unlabelled 1. Eight
deleted posts and about 90 gone authors in three weeks: too thin for a per-account view
and it would be an accusation. Aggregate arm-level rates with denominators are the
most that is supportable, and the top arm is engagement-biased (top 5%).

## Hate and incitement, the composition-drift caveat

`hatespeech/` is per post and carries several flag columns. The collector edits its own target list;
between July and August the baseline mix went from 70.7% search / 12.1% replies to
14.3% / 72.6% and replies carry 3.2x the hate rate (`docs/analysis/2026-09-13-publishable-statistics.md`).
Window composition today (30 d rows): replies 949,686, search 848,735, deep_timeline
251,950, census_timeline 202,063, timeline 188,344, hydrated 33,230, control 29,213,
parent_backfill 23,514, trend 8,442, cib_timeline 6,486. A flag count over this
corpus is a mix statistic. Only within-partition (baseline types by first-seen type, or
control) or standardised rates are defensible. Incitement is stale since 09-28.

## Embeddings and a 2-D map

1,542,603 vectors. Measured on this Mac (14 cores): 15,000 vectors read from R2 in 12.0 s;
PCA to 2-D 0.81 s; UMAP (cosine, 15 neighbours, seeded) 20.6 s, 0.57 GB. The `cloud` box
was not measured; expect it slower, to be measured. A post-level map of a few tens of
thousands of sampled posts is feasible on `cloud` in minutes. It answers "what is near
what" and little else; topics and tags answer the named questions more directly. Not v1.

## Canary, leads, change detection

`kind=daily_members` holds a full membership per run, enough for overlap matching, but
only two runs exist and both were written on the same day, so "new or vanished since
yesterday" has no data until about 2026-10-12. Canary and leads artifacts do not exist
in R2 (no `canary/`, `leads/` prefixes), so canary status renders "not available".

## Network and graph sizes, measured for the stack choice

Graphs from the 21 d run, kept communities, layout positions precomputed.

| Graph | Nodes | Edges | Layout cost | Browser |
|---|---|---|---|---|
| top 25 per community (the listing) | 306 | 4,810 | graphology FA2 0.9 ms/iter; d3-force 0.8 ms/tick | n/a |
| top-5 backbone | 10,179 | 37,422 | FA2 40.2 ms/iter; d3-force 26.8 ms/tick; igraph DRL 8.6 s, FR-grid 0.2 s, LGL 0.4 s, graphopt 120 s | n/a |
| all kept intra edges | 10,179 | 258,976 | FA2 42.9 ms/iter | JSON 5.4 MB |
| default rule (shared >= 2 or other trace) | 7,547 | 42,445 | igraph DRL, one-off | Sigma 3.0.2 WebGL: 120 fps over 180 camera moves (8.3 ms/frame), graph build 0.5 s, first render 3.0 s; JSON 0.63 MB raw, 0.14 MB gz |
| whole fused graph | 10,941 | 459,744 | n/a | Sigma did not finish loading inside about 2 minutes, never rendered. JSON 5.7 MB raw, 0.87 MB gz (with random positions) |

The browser numbers were taken in the desktop app's embedded browser, one run each; treat
as order of magnitude. DuckDB-WASM 1.29.0 `duckdb-eh.wasm` is 35.7 MB uncompressed
(CDN header); Workers static assets cap a file at 25 MiB, so it is not bundled with the
site. Compressed transfer size was not measured.

## Read cost, for the producer budget

Reading the 21 d window's posts for the 10,941 graph accounts (join against the id list,
all of `posts/` partitions in the window) took 287 s on this Mac over the home link,
yielding 176,019 distinct posts from 10,941 authors (50.3% reposts, 35.3 MB of text).
Post counts per account: median 6, p90 35, p99 159, max 3,373. Rebuilding the networks
took 346 s and about 2 GB. Neither is measured on `cloud`.

## What does not exist and must be built

| Need | Status |
|---|---|
| Fused edge list, trace rows, evidence per edge | not persisted; persist in the daily (preferred) or rebuild |
| Text-similarity post pairs | not persisted; recompute for ~1.3k edges |
| Community identity across runs | none; ids are per run. Build lineage by overlap |
| Topic term labels, topics for posts after 09-04 | not persisted |
| Window coverage by stage | not persisted |
| Per-account "amplifies" | derivable from posts reposts joined to originals; original coverage unmeasured |
| Hashtag and URL aggregates, per-day | derivable from `posts/` list columns; not persisted |
| Change detection | needs 2+ runs on different days |
| Canary, leads | absent |
