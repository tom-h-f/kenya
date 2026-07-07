# 04 - Communities and multiplex integration

Each channel's validated graph (03) is one layer over a shared account node set.
Coordinated groups are dense communities; the strongest evidence is a group that
is dense in **multiple independent layers**.

## Community detection: Leiden

Use Leiden (`leidenalg` on `igraph`) over Louvain: Leiden guarantees
well-connected communities (Louvain can return internally-disconnected ones) and
is more stable.

- Quality function: CPM (Constant Potts Model) with a resolution `gamma`, or
  modularity. CPM gives a resolution knob with a clear interpretation (link
  density threshold) and avoids modularity's resolution limit.
- Sweep `gamma`; report community count, sizes, and stability across the sweep
  (robust clusters persist).
- Run per layer, and on an aggregated multiplex graph.

## Multiplex integration - two views, both kept

1. **Per-layer + corroboration (primary).** Detect communities in each layer
   independently, then score every account pair (or cluster) by the number of
   layers in which they are co-clustered / directly linked. A pair coordinated
   across `k >= 2` independent channels is far less likely to be organic than one
   channel alone (survey: multi-channel corroboration is the strongest available
   evidence short of ground truth). Emit a `corroboration` count per edge/cluster.
2. **Aggregated multiplex (secondary).** Sum/normalise layer weights into one
   graph (or use multiplex-aware community detection), Leiden over it, for a
   single consolidated clustering.

## Corroboration score

For a candidate cluster, record:
- `channels`: which layers support it (e.g. {co-retweet, text-sim}).
- `n_channels`: count of supporting layers.
- intra-cluster edge density per layer.
- share of member pairs that are SVN-validated (not just percentile).

Rank clusters by (n_channels, SVN density, size). High `n_channels` + high SVN
density = high-confidence coordinated cluster; single-channel + percentile-only =
weak, flag as tentative.

## Outputs

Cluster table: `cluster_id`, member `author_id`s, `channels`, `n_channels`,
per-layer density, size, and the SVN-validated internal edge share. Feeds
characterization (05). Persist to `coordination/` (07).

## Deps

`python-igraph`, `leidenalg` (add). `networkx` (already present) as a fallback /
for convenience metrics.
