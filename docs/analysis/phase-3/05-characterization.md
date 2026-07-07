# 05 - Characterization: authentic vs inauthentic

Detection (02-04) finds coordinated clusters. Coordination alone is not
malicious: fan groups, news outlets, and open campaigns coordinate legitimately.
Characterization separates inauthentic amplification from organic co-activity,
reusing Phase 1 and Phase 2 directly.

## Per-cluster scorecard

For each cluster from 04, compute:

**Authenticity (Phase 1, `kma.authenticity`)**
- distribution of member `suspicion` and `anomaly_rank` (mean, median, share > p90).
- account-age distribution and **creation-date burstiness**: many members created
  in a narrow window is a strong CIB signal (compute the tightest window holding
  X% of members).
- share of members that are default-image / empty-bio / digit-suffix handles.

**Narrative (Phase 2, `kma.semantic` / `kma.classify`)**
- **narrative homogeneity**: do members concentrate on one topic cluster?
  (entropy of member topic distribution; low = homogeneous).
- **near-duplicate rate**: share of member posts that are near-duplicates of each
  other (mean pairwise cosine, from embeddings).
- stance/sentiment alignment toward tracked targets (do members push the same
  stance in lockstep?).

**Coordination-intrinsic**
- supporting `channels` and `n_channels` (04).
- synchrony: median inter-arrival `min_gap` of member co-actions (tighter = more
  scripted).
- SVN-validated internal edge share.

**Impact**
- combined reach (sum/unique followers), aggregate engagement, and engagement-
  per-follower (amplification efficiency).

## Inauthenticity index

A transparent composite (documented weights, calibrated in 06), combining:
`bot-likeness` (mean suspicion) + `synchrony` (inverse median gap) +
`homogeneity` (narrative concentration) + `concealment` (account freshness /
sockpuppet handle-image signals) + `corroboration` (n_channels). Report the
components, not just the scalar - the breakdown is what an analyst acts on.

Explicitly do NOT auto-label: output ranked, explained scorecards for human
review. State that legitimate coordination scores non-zero and that the index is
a triage tool.

## Outputs

Ranked cluster scorecards (one row per cluster with all components + member
list), plus a member-level table joining each account's authenticity + dominant
topic + stance. Notebook `notebooks/coordination.py` renders the top clusters,
their narratives, member authenticity, and timing.
