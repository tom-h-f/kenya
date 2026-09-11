# Text-similarity threshold sensitivity (finishing-the-revamp, headline check)

Run 2026-09-11 on snapshot `2026-09-05-promotion-off`. Question: the v2 text
trace uses an absolute cosine cut of 0.85 - our choice, not the paper's - and
`v2-findings` says it "determines most of the ranking". How stable is the top
500 as that cut moves?

## Method

1. `00_export.py` (tac2, R2): the text-eligible rows and their embeddings,
   exactly as `modal_textsim.build` selects them.
2. `01_pairs.py` (mike-pc, RTX 3070): every post pair at or above 0.75 recorded
   once, then replayed through the unchanged `coord2.text_similarity_network`
   for each cut. A user edge is weighted by the mean over qualifying post pairs,
   so higher cuts are not a filter of lower ones - replay keeps the project's
   own aggregation. `--verify 0.85` recomputes that cut directly on the GPU:
   **identical, 409,085 edges**.
3. `02_score.py` (tac2, R2): the four bipartite traces built once; per cut, the
   text floor, fusion, eigenvector centrality, top 500 and Kenya share - what
   `coord2_run.run` does.

## Gating checks

- **Check 1, local 0.85 text trace against production:** 409,085 against
  409,084. The one extra edge has similarity exactly 0.85 - a boundary pair the
  A10G and the 3070 round differently; across the 409,084 shared edges the
  largest weight difference is 1.5e-6. The activity floor removes that edge, so
  both give **244,308 edges after the floor**, the figure `v2-findings` reports.
- **Check 2, local 0.85 top 500 against the persisted production run:** 500 of
  500 accounts shared.

The sweep reproduces production at the production setting.

## Result

| cut | text edges after floor | top-500 overlap with 0.85 | Jaccard | Spearman on overlap | top-50 overlap | Kenya share mean / median | accounts >= 15% |
|---|---|---|---|---|---|---|---|
| 0.75 | 5,368,556 | 363 | 0.570 | 0.492 | 24 | 0.448 / 0.493 | 401 |
| 0.80 | 1,541,356 | 420 | 0.724 | 0.739 | 39 | 0.431 / 0.486 | 382 |
| **0.85** | 244,308 | 500 | 1.000 | 1.000 | 50 | 0.404 / 0.456 | 353 |
| 0.90 | 22,109 | **1** | 0.001 | - | 0 | 0.830 / 1.000 | 500 |
| 0.95 | 4,513 | **1** | 0.001 | - | 0 | 0.830 / 1.000 | 500 |

**Below 0.85 the ranking is moderately stable; between 0.85 and 0.90 it is
replaced wholesale.** At 0.90 and 0.95 the top 500 is the same 500 accounts,
only 13 and 2 of them have any text edge at all, and their median activity is 3
posts against 18 at 0.85. Above the break, centrality falls back to the
co-retweet core - consistent with `v2-findings`' note that the text-driven
ranking shares one account with the co-retweet-only one. It shares exactly one
here too.

## What it means

- **v2's top 500 is whichever dense block owns the leading eigenvector.** At
  0.85 and below that is the text-similarity mass; above it, the co-retweet
  core. Eigenvector centrality concentrates on the dominant block, which is also
  why `v2-findings` §8 found the top 500 to be one component of density 0.491.
- **0.85 sits just below a phase change**, not on a plateau. A ranking that
  swaps its entire membership over 0.05 of a cut is a property of the method on
  this corpus, and any v2 output should carry its cut with it (it already does,
  in `text_threshold`).
- The two regimes are different findings, not better and worse versions of one:
  the text regime surfaces accounts posting near-identical text; the co-retweet
  regime surfaces low-activity accounts co-amplifying the same posts. The higher
  Kenya share above 0.90 rests on a median of 3 posts per account through a
  keyword proxy whose recall is 0.508, so it is not evidence that regime is
  more on-topic.
- The reference implementation's default cut is 0.70. It was not run at full
  scale: at 50k rows it already gives 2.2M user edges, and fusion at 0.75 took
  7.3 GB on tac2.

## Found on the way

- **Only 335,023 of 536,722 text-eligible posts (62%) have an embedding** in
  the pinned snapshot. The production trace uses the same inner join, so it
  silently omits 38% of eligible posts. The embedding model runs on the 3070.
- **Cost, measured:** export 265 s and 7.2 GB on tac2; GPU pair pass 29 s at
  2.09 GiB peak for 335k posts (23.6M pairs at 0.75); scoring all five cuts
  967 s and 7.3 GB on tac2.

## A2 size-matched re-run

Pending - appended when the adjudication completes.
