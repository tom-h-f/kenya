# Phase 3 - Coordination networks

Status: not started. Planned in detail.

> **Detailed research-grade plan: [`phase-3/`](phase-3/README.md)** (8 files:
> traces, network construction, statistical validation, multiplex/communities,
> characterization, evaluation, implementation). This file is the original
> high-level summary; the `phase-3/` series supersedes it.

The classic disinformation-campaign toolkit. Depends on Phase 0 (fuller capture +
structured fields) and reuses Phase 2 embeddings.

## Why

Detect accounts acting in concert to amplify narratives - the core of a
disinformation campaign. No single signal proves coordination; we stack
behavioural signals until the combination beats any organic explanation, then
surface the resulting account clusters.

## Method (CooRnet / CooRTweet style)

For each coordination channel, find actions repeated by different accounts inside
a short time window `delta`, build a weighted account-account graph, threshold,
then community-detect.

Edge types (all from DuckDB self-joins on existing columns):
- **co-retweet** - same `repost_of_id` by different `author_id` within `delta`.
- **co-hashtag** - overlapping `hashtags` sets (Phase 0) within `delta`.
- **co-URL** - same `urls` entry (Phase 0) within `delta`.
- **co-reply** - same `in_reply_to_id` within `delta`.
- **fast-retweet** - very small inter-arrival between original and repost.
- **near-duplicate text** - cosine over Phase 2 embeddings above a threshold, or
  MinHash if embeddings not ready.

## Deliverables

- `analysis/src/kma/coordination.py`:
  - `edges(con, kind, delta) -> relation` - weighted account pairs for one
    channel.
  - `coordination_graph(con, ...)` - union/weight channels, threshold, return an
    edge list.
  - `communities(edge_list)` - Louvain/Leiden clusters (via `networkx` or
    `igraph`).
  - `characterise(con, communities)` - join Phase 1 suspicion scores + Phase 2
    narratives per cluster (who pushes what, how synchronously).
- `analysis/notebooks/coordination.py` (marimo) - suspicious clusters, their
  narratives, member authenticity scores, timing.
- New deps via `uv`: `networkx` (or `python-igraph`).

## Parameters - MEASURE, don't guess

- `delta` (co-action window), edge-weight thresholds, near-dup cosine cutoff:
  all tuned empirically. `DECISION:` start from literature defaults (e.g. co-share
  window of seconds-to-minutes) then calibrate on our data.

## Caveats

- Even after Phase 0, twscrape capture is not exhaustive - clusters are
  suggestive, not a census. Document this in every output.
- Coordination is not inherently malicious (fan groups, campaigns organise
  openly). Characterisation (Phase 1 + Phase 2) is what separates authentic
  organising from inauthentic amplification.

## Verify

Confirm the method recovers an obviously coordinated set (e.g. bulk
identical-text amplifiers) before trusting subtler clusters. Notebook runs
end-to-end against live R2.
