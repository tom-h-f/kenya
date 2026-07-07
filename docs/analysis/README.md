# Disinformation-analysis plan

Phased build of disinformation-tracking analysis on the collected X data.
Ordered by data-fit; each phase ships a `kma` module + a marimo notebook that
runs end-to-end against live R2.

| Phase | Focus | Status |
|---|---|---|
| [0](phase-0-collector-completeness.md) | Collector completeness (structured fields, full capture) | done |
| [1](phase-1-authenticity.md) | Account authenticity / bot scoring | done |
| [2](phase-2-semantic.md) | Semantic / narrative layer | done (topic tuning open) |
| [3](phase-3/README.md) | Coordination networks (CIB) | done (Wave B pending data) |

## How the analysis works (context)

Disinformation-campaign detection is probabilistic - no single signal proves
coordination, so signals are stacked until the combination beats any organic
explanation. Four families:

1. Account authenticity / bot scoring (Phase 1).
2. Coordinated inauthentic behaviour networks - co-retweet / co-hashtag / co-URL
   graphs + synchronized timing (Phase 3).
3. Temporal / burst analysis (woven through 1 and 3).
4. Narrative / semantic tracking - embeddings, topics, sentiment, stance (Phase 2).

Persistence is hybrid: expensive embeddings are written back to R2
(`embeddings/`); cheap bot scores and coordination edges compute live in DuckDB.

### Key references

- [Uncovering Coordinated Networks on Social Media](https://btrantruong.github.io/assets/pdf/uncover.pdf)
- [Coordination Network Toolkit](https://link.springer.com/article/10.1007/s42001-024-00260-z)
- [Network analysis of disinformation campaigns](https://arxiv.org/pdf/2005.13466)
- [Account-history features for bot detection](https://arxiv.org/pdf/2606.26127)
- [Botometer 101](https://pmc.ncbi.nlm.nih.gov/articles/PMC9391657/)
