# 07 - Implementation

## Module: `analysis/src/kma/coordination.py`

Builds on existing helpers: `kma.db` (`latest_posts`, `latest_authors`,
`latest_embeddings`, source globs, `BUCKET`), `kma.authenticity`, `kma.semantic`,
`kma.classify`. DuckDB-first for traces/projection; igraph/leiden + scipy/
statsmodels for validation and communities.

### API

```python
# --- traces + projection (02) -------------------------------------------
def traces(con, channel, platform="x"): ...
    # -> relation (author_id, action_object, created_at) for one channel

def projected_edges(con, channel, delta=None, min_repetition=2, weighting="count"):
    # DuckDB self-join projection -> (src, dst, weight, min_gap, ...)
    # weighting in {"count","tfidf"}

# --- statistical validation (03) ----------------------------------------
def validate_svn(edges, degrees, n_objects, method="fdr_bh", alpha=0.01):
    # hypergeometric p per pair + multiple-testing correction
    # -> edges with (p_value, validated) ; method in {"fdr_bh","bonferroni"}

def validate_montecarlo(con, channel, delta, n_iter=1000, alpha=0.01):
    # time-shuffle null for fast/text-sim channels

def percentile_filter(edges, q=0.995):  # CooRnet-style baseline

# --- multiplex + communities (04) ---------------------------------------
def build_layers(con, channels, **params): ...   # dict[channel] -> validated graph
def communities(graph, resolution=1.0):           # leidenalg CPM -> membership
def corroborate(layers): ...                      # per-pair/cluster channel support

# --- characterization (05) ----------------------------------------------
def scorecards(con, clusters): ...   # joins Phase 1 + Phase 2 -> ranked table

# --- evaluation (06) ----------------------------------------------------
def inject_synthetic(con, k, seed_objects, window, strength): ...
def evaluate_recovery(...); def null_baseline(...); def internal_validation(...)
```

### Text-similarity channel

Reuse Phase 2 embeddings + DuckDB `vss`: build the kNN graph (cosine >= `tau`),
connected components = content clusters, then project accounts over cluster
membership. Candidate-pair restriction (only pairs sharing a near-dup cluster)
keeps the synchronised-semantic variant tractable.

## Persistence (hybrid, matches the roadmap)

Coordination artifacts are expensive to recompute at scale, so persist per run to
a new R2 prefix, mirroring `embeddings/` and `labels/`:

```
coordination/platform=x/kind=edges/channel=<c>/method=<svn_fdr|svn_bonf|pct>/dt=/run=.parquet
coordination/platform=x/kind=clusters/dt=/run=.parquet
```

Edge lists carry `(src, dst, weight, p_value, min_gap, method)`; clusters carry
`(cluster_id, author_id, channels, n_channels, ...)`. Add `coordination_source()`
/ `latest_*` helpers to `kma.db`.

## Dependencies (add via uv)

`python-igraph`, `leidenalg`, `statsmodels`. Present already: `scipy`,
`networkx`, `duckdb`, `polars`, embeddings stack.

## Milestones

**Wave A (buildable now, current 18k):**
- M1 - co-retweet channel end to end: `traces` -> `projected_edges` ->
  `validate_svn` (Bonferroni+FDR) + `percentile_filter` -> Leiden communities ->
  `scorecards`. Persist. Notebook `notebooks/coordination.py`.
- M2 - text-similarity channel (embeddings near-dup) added as a second layer;
  corroboration across co-retweet + text-sim.
- M3 - co-reply + fast co-share layers; evaluation harness (06): synthetic
  injection, null baseline, internal validation, case studies (crypto/insurance
  positive controls).

**Wave B (as post-Phase-0 data accrues):**
- M4 - co-hashtag, co-URL, co-mention layers once field coverage is sufficient
  (track via `len(hashtags)>0` share).
- M5 - full multiplex corroboration + robustness sweeps; finalise the
  inauthenticity index weights against the M3 evaluation.

## Verification (per milestone)

Each milestone runs end to end on live R2 and passes its 06 checks: null-shuffle
gives ~empty Bonferroni network; synthetic injections are recovered above the
strength threshold; detected clusters beat random groups on bot/narrative
internal validation; the crypto/insurance clusters are recovered. Notebook
exports clean. Nothing is reported as "coordinated" on a single channel /
percentile-only basis.

## Compute

Projection self-joins run in DuckDB (scales with shared-object pairs, not
A^2). Popular-object explosion handled by TF-IDF downweighting + degree caps
(02). SVN p-values vectorised over tested pairs via scipy. Leiden on the
validated (sparse) graph is fast. Benchmark M1 on the full 18k before scaling
parameters - do not guess thresholds, sweep them (06).
