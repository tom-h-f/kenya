# 03 - Statistical validation of edges

The core of research-grade CIB detection: keep only co-actions that are
statistically surprising under a null model, not merely frequent. Two methods,
both computed and compared (04, 06).

## Primary: Statistically Validated Networks (Tumminello et al. 2011)

For an account pair, ask: given how active each account is, how surprising is
the number of objects they share? Model the shared count under a hypergeometric
null.

For accounts A and B on a channel with `M` distinct action objects total, where
A acted on `N_A` objects and B on `N_B`, the number of shared objects `X` under
random assignment follows a hypergeometric distribution. The one-sided p-value:

```
p(A,B) = P(X >= x_obs) = sum_{k=x_obs..min(N_A,N_B)} hypergeom.pmf(k; M, N_A, N_B)
```

Implement with `scipy.stats.hypergeom.sf(x_obs - 1, M, N_A, N_B)`.

**Multiple-testing correction.** There are up to O(A^2) pairs, so raw p-values
overcount. Two validated networks:

- **Bonferroni network**: keep edges with `p < alpha / n_tests` (alpha = 0.01).
  Very conservative; near-empty on random data (the desired false-positive
  control).
- **FDR network**: Benjamini-Hochberg at `q = 0.01` via
  `statsmodels.stats.multitest.multipletests(method="fdr_bh")`. Higher recall,
  still controlled.

Ship both; report the Bonferroni network as the high-precision core and FDR as
the sensitive view.

`n_tests` = number of candidate pairs actually tested (pairs sharing >= 1
object), not the full O(A^2), which keeps the correction from being needlessly
brutal (standard SVN practice).

### Time-constrained variant

For fast co-share, the object set is the same but co-occurrence requires the
pair within `delta`. The null is then over time-shuffled `created_at` (Monte
Carlo): shuffle timestamps within each account, recompute co-within-delta counts,
build the empirical null distribution of shared counts, take the p-value. This
is the null model the text-similarity and fast channels use where the
hypergeometric object model does not directly apply.

### Text-similarity channel null

Near-duplicate content clusters are the objects; apply the same hypergeometric
SVN over (account x content-cluster) incidence. For the synchronised-semantic
variant, use the time-shuffle Monte-Carlo null.

## Baseline: percentile / CooRnet-style

The widely-used simple filter, for comparison:

- Keep edges above a high weight percentile (e.g. top 0.5%), OR
- CooRTweet rule: `time_window` (default 10s) + `min_repetition` (default 2),
  keep all surviving edges.

No null model, so it cannot distinguish "surprising" from "popular". We report
the overlap between the percentile edge set and the SVN edge set - large
divergence is itself informative (percentile keeps popular-object noise SVN
rejects).

## Outputs and reporting

Per channel: validated edge lists (Bonferroni, FDR, percentile) with p-values
and weights. Report edge counts, graph density, degree distribution, and the
Jaccard overlap of the three edge sets. On time-shuffled input the Bonferroni
network must be ~empty - this is a required sanity check (06).

## Deps

`scipy` (installed), `statsmodels` (add). Hypergeometric SF over tested pairs is
cheap; the expensive part is enumerating tested pairs, done in DuckDB (02).
