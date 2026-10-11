# Transitive families: where text components were used as families

2026-10-11, branch `fix/flagged-families`.

Problem: a connected component of a similarity graph is transitive. A~B and B~C put A and C together
though they need not resemble each other. On the text trace this chained topic-similar posts into one
4,000-post "campaign" (community 0, measured 2026-10-11). `kma.dossier` already used hub-star families.

Rule now: every place that groups posts by text similarity uses `kma.families` (hub-star). The post with
most partners claims every unclaimed partner, so each member is a direct match of the family hub and each
post is in exactly one family. Deterministic (degree, then index).

Places changed:
- `coordination.content_clusters` (near-duplicate clusters, the text_sim action objects, tau 0.9).
- `stories.candidate_stories` (claim clusters, story tau).
- `dossier.near_duplicate_families` now imports the shared function (behaviour unchanged).
- `investigations/2026-10-11-text-only-adjudication/01_packets.py` `families()`: was union-find over
  word-Jaccard pairs; now hub-star, pairs crossing two families are not counted in either.

Does the old chaining guard in `stories` suffice? No. `_reject_chaining_blobs` only fires above
`MAX_COHERENT_AUTHORS` (150) authors AND cohesion < 0.80. Measured on a synthetic random walk on the sphere
(neighbours at cosine 0.9, tau 0.85, 64 dims, one author per post, `tests/test_families.py`):

| posts | components | cohesion | guard keeps blob | hub families (size>1) | largest family |
|---|---|---|---|---|---|
| 120 | 1 (120) | 0.361 | yes | 55 | 4 |
| 150 | 1 (150) | 0.326 | yes | 70 | 4 |
| 2000 | 1 (2000) | 0.094 | no | 890 | 5 |

A chain of up to 150 authors with cohesion about 0.33 (real claim clusters are 0.75-0.99) passed the guard
as one "story". The guard stays as a backstop for hub families, which cannot chain by construction. Real-corpus
effect (story counts, sizes before/after) was not measured: it needs R2 reads of the embeddings and a
re-run, and downstream story ids (`stable_story_id`) will shift because membership changes.

Tests: `analysis/tests/test_families.py`.
