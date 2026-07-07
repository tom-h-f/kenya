# 06 - Evaluation and validation

No ground truth for CIB in this dataset, so validity is established by
convergent evidence, not a single accuracy number.

## 1. Synthetic injection (primary quantitative test)

Plant a known coordinated cluster into the real data and measure recovery.

- Generate `K` synthetic accounts that co-act on a seed set of `S` objects
  (retweet the same tweets / post near-duplicate text) within a window `w`.
- Inject their traces into the real trace tables; run the full pipeline
  (02-04) unchanged.
- **Metrics**: precision / recall / F1 of recovered injected accounts; the
  survey's **weighted precision** (weighted average of positive rate over
  non-singleton clusters, penalising over-fragmentation that splits the injected
  group).
- **Detection ROC**: sweep injection strength - synchrony `w`, overlap `S`,
  cluster size `K` - to map the boundary of what the pipeline can detect. This
  quantifies recall limits (tie back to the sampling caveat in 01).
- Run against SVN and percentile filters to compare their operating points.

## 2. Null-model baseline (false-positive control)

Time-shuffle `created_at` within each account (destroying real synchrony) and
run the pipeline. The **Bonferroni SVN network must be near-empty**; a non-empty
result signals a bug or an inadequate null. Report validated-edge count on real
vs shuffled input.

## 3. Internal validation (convergent evidence)

Detected clusters should differ from random account groups of equal size:
- significantly higher Phase 1 `suspicion` / anomaly (permutation test, report
  effect size + p-value).
- significantly higher narrative homogeneity and near-duplicate rate (Phase 2).
If detected clusters are indistinguishable from random groups, the detection is
not capturing anything meaningful - a falsification test, not a vanity metric.

## 4. Case studies (qualitative)

- **Positive controls**: the crypto/finance and insurance/reinsurance
  promotional clusters surfaced by Phase 2 topic modelling are near-certainly
  coordinated spam; the pipeline should recover them. Their recovery is a
  real-data sanity check.
- Manually inspect the top-`k` high-`inauthenticity-index` clusters: document
  member handles, shared content, timing, and a human judgement (coordinated-
  inauthentic / legitimate-coordination / false-positive). Report precision@k.

## 5. Robustness / sensitivity

- Stability of clusters across `delta`, `tau`, resolution `gamma`, and
  correction (Bonferroni vs FDR) sweeps - robust clusters persist.
- Edge-set overlap (Jaccard) SVN vs percentile.

## Reporting

An evaluation notebook / section that runs 1-5 on live data and records: recovery
F1 vs injection strength, shuffled-vs-real validated-edge counts, internal-
validation effect sizes, and the case-study precision@k table. A cluster is only
reported as "coordinated" when multiple of these agree.
