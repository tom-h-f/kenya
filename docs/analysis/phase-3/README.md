# Phase 3 - Coordination networks (detailed plan)

Detecting coordinated inauthentic behaviour (CIB): groups of accounts acting in
concert to amplify narratives, where the coordination is concealed. This is the
richest and hardest part of the roadmap, so it is planned as a series of files.

Read in order:

| # | File | Stage |
|---|------|-------|
| 00 | this file | overview, methodology, literature |
| 01 | [data-and-traces.md](01-data-and-traces.md) | behavioural traces from our schema + data-availability matrix |
| 02 | [network-construction.md](02-network-construction.md) | bipartite -> account-account projection, per channel |
| 03 | [statistical-validation.md](03-statistical-validation.md) | null models, SVN, FDR/Bonferroni, percentile baseline |
| 04 | [communities-multiplex.md](04-communities-multiplex.md) | multiplex layers + Leiden + cross-channel corroboration |
| 05 | [characterization.md](05-characterization.md) | integrate Phase 1 + 2; authentic vs inauthentic scorecards |
| 06 | [evaluation.md](06-evaluation.md) | synthetic injection, internal validation, null baselines, case studies |
| 07 | [implementation.md](07-implementation.md) | `kma/coordination.py` API, persistence, deps, milestones |

## Why this is hard (and how we get it right)

Coordination detection is inherently probabilistic: no single co-action proves
coordination, since real users also co-share. The discipline is to (a) measure
co-actions against a **null model** so we only keep statistically surprising
links, (b) **corroborate across independent channels**, and (c) **characterize**
each cluster (bot-likeness, narrative homogeneity, synchrony) to separate
inauthentic amplification from ordinary organic co-activity (fan clubs, open
campaigns). Coordination is not by itself malicious - characterization is what
earns that label.

## The pipeline (Pacheco et al. 2021; Cresci survey 2024)

```
behavioural traces            (01)
   -> bipartite account x action network, per channel
   -> project to account-account graph, time-constrained   (02)
   -> filter edges by statistical validation (SVN) + percentile baseline (03)
   -> multiplex layers -> Leiden communities -> corroborate (04)
   -> characterize + score clusters (Phase 1 bot + Phase 2 narrative) (05)
   -> evaluate: synthetic injection, internal validation, null baseline (06)
```

## Decisions locked (2026-07-07)

- **Edge filtering:** statistically-validated networks (null model + FDR/
  Bonferroni) as primary, CooRnet-style top-percentile as a comparison baseline.
- **Sequencing (two waves).** Wave A on the current 18k posts now: co-retweet,
  text-similarity (embeddings), co-reply, fast co-share/timing. Wave B as
  post-Phase-0 data accrues: co-hashtag, co-URL, co-mention. See 01.
- **Graph model:** multiplex (one layer per channel) + Leiden, with cross-channel
  corroboration as the confidence signal.
- **Evaluation:** full - synthetic injection with recovery metrics, internal
  validation against Phase 1/2, null-model baselines, qualitative case studies.

## What is unique to our setup

- Phase 2 **embeddings** enable a text-similarity channel (semantic near-duplicate
  co-posting) that link/hashtag-only tools cannot do.
- Phase 1 **bot scores** and Phase 2 **topics/sentiment/stance** make the
  characterization stage (05) immediate rather than future work.
- Known-ish positive controls already surfaced: the crypto/finance and
  insurance/reinsurance promotional clusters from Phase 2 topic modelling are
  natural coordinated-content test cases (06).

## Key references

- Pacheco, Hui, Torres-Lugo, Truong, Flammini, Menczer. *Uncovering Coordinated
  Networks on Social Media: Methods and Case Studies.* ICWSM 2021.
  [arXiv:2001.05658](https://arxiv.org/abs/2001.05658)
- Mannocci, Mazza, Monreale, Tesconi, Cresci. *Detection and Characterization of
  Coordinated Online Behavior: A Survey.* 2024.
  [arXiv:2408.01257](https://arxiv.org/abs/2408.01257)
- Tumminello, Miccichè, Lillo, Piilo, Mantegna. *Statistically Validated Networks
  in Bipartite Complex Systems.* PLoS ONE 2011.
  [journal](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0017994)
- Nizzoli, Tardelli, Avvenuti, Cresci, Tesconi. *Coordinated Behavior on Social
  Media in 2019 UK General Election.* ICWSM 2021.
- Righetti et al. *CooRTweet: A Generalized R Software for Coordinated Network
  Detection.* Computational Communication Research 2025.
- Weber, Neumann. *The coordination network toolkit.* J. Computational Social
  Science 2024. [link](https://link.springer.com/article/10.1007/s42001-024-00260-z)
- *Detecting Coordinated Activities Through Temporal, Multiplex, and Collaborative
  Analysis.* 2025. [arXiv:2512.19677](https://arxiv.org/html/2512.19677v1)
