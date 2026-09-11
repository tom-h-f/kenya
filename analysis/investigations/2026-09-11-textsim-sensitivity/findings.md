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

Run 2026-09-11. The 2026-09-09 comparison read v1's seven largest clusters
against v2 groups of 4-40 members. Engagement pods are the large ones, so its
1-of-7 against 6-of-7 political could have been size. This pairs every v2 group
of 4+ members (Leiden CPM 0.5, seed 0, over the 0.85 top 500: 255, 143, 27, 19,
16, 12, 5, 5, 4, 4, 4) with the unused v1 cluster nearest its size (177, 88, 27,
17, 14, 12, 5, 5, 4, 4, 4). v1 has no cluster near 255 or 143, so the two largest
pairs are approximate; the rest match within three members.

**Reader:** headless Claude Code (`05_a2_headless.py`, claude-sonnet-5). One
fresh process per case, with the adjudicator's own system prompt and rubric, in
an empty directory with no tools, MCP servers, hooks or user settings - checked
with `--probe` - over 22 cases under shuffled ids. Verdicts were persisted before
unblinding: `coordination/platform=x/kind=verdicts/dt=2026-09-11/run=20260911T161407Z.parquet`.
22 cases in 34 s, 0 failures, reported cost $0.30.

| | cases | political | Kenya-relevant | verdicts |
|---|---|---|---|---|
| v1 | 11 | 0 | 0 | engagement_pod 8, fandom_or_interest 3 |
| v2 | 11 | 3 | 9 | engagement_pod 4, unclear 4, political_campaign 3 |

By size: v1 is engagement pods and fandoms at every size. v2's two large blobs
(255, 143) are engagement pods, both Kenya-relevant; its mid-sized groups are 2
of 4 political; its small groups are 4 of 5 unclear. Nothing in either method was
political at high confidence, and nothing was called an influence operation: all
three political verdicts are open pro-Sifuna campaign amplification.

**The four unclear v2 verdicts are an instrument gap, not a finding.** They are
exactly v2's four cases with no shared objects. `kma.dossier.build` fills "what
they jointly amplified" from co-retweets, which is v1's signal; a group v2 linked
by text similarity arrives with that section empty, and the rubric tells the
reader that members' own posts are "context, not evidence". All four rationales
say the co-action was unavailable. Where v2's dossier did carry evidence, 3 of 7
groups were political and 6 of 7 Kenya-relevant.

What this does to the 2026-09-09 result:

- **Does not survive:** "v2's output reads as political coordination (6 of 7)".
  Size-matched it is 3 of 11 - 3 of 7 where there was evidence - and the largest
  v2 groups are Kenyan engagement pods.
- **Survives, and sharper:** v2 surfaces Kenyan activity and v1 does not - 9 of
  11 against 0 of 11 Kenya-relevant, with v1 non-political at every size. For
  monitoring the Kenyan election conversation, v2 is still the instrument.
- **New:** adjudicating v2 needs a dossier that shows text-similarity co-action
  (the groups' near-duplicate posts) before its small groups can be judged at all.

Caveats kept attached: model verdicts, not human ones; n = 11 per method; the
largest pairs are size-approximate; the Leiden split is a fixed seed, not
2026-09-09's unrecorded one.

## What the 0.85 text trace links

Found 2026-09-11 while building the dossier fix above. The rebuilt dossiers
still showed nothing for v2's four empty cases, although every edge inside them
is a text-similarity edge (so is every edge but 5 inside all 11 v2 groups). Two
causes:

- The dossier reads member posts through `_latest_posts_cte` with the default
  14-day lookback. The pairs behind these groups date 2026-07-06 to 2026-08-15.
  Fixable.
- The pairs are not near-duplicates. They are unrelated short Swahili and Sheng
  replies to different accounts - "@KeKirwa Wakifumble hapa watakuwa kama
  wakamba..." against "@AokoOtieno_ Vitu zingine kama kuwa goons..." at 0.899.

`06_pair_audit.py` measures this over the 0.85 top 500 (snapshot
`2026-09-05-promotion-off`, 34,082 embedded posts):

- 279,060 cross-author post pairs at or above 0.85.
- In a sample of 3,000, median token Jaccard is 0. 84.6% share under a tenth of
  their words; 3.0% are near-copies (Jaccard >= 0.5).
- Random pairs average cosine 0.261 and essentially none reach 0.85 (0.04%
  among the top 500's own posts). This is a dense region, not a collapsed
  space.
- A blind headless reader (claude-sonnet-5) on 100 of the 3,000: 26 same
  message, 15 same topic, 59 unrelated. None of the 100 was a near-copy.

The same-message pairs are what the trace is for. Most of those read were one
paraphrased Tanzanian Saba Saba campaign (#TumekataaHarakatiZaVurugu,
#AibuKwaWanaharakati) whose posts share almost no words. So the trace does catch
paraphrased messaging, but at 0.85 most of its links behind the top 500 join
posts that say different things. v2-findings section 4 says 0.85 was chosen
"because that is where near-duplicates sit in our encoder's space"; near-copies
are 3% of what it admits.

What this changes:

- **The four unclear A2 verdicts were correct.** Those groups have no co-action
  to show.
- **"v2 surfaces Kenyan activity, v1 does not" is weaker than stated above.** A
  trace that links Swahili and Sheng replies whatever they say will surface
  Kenyan accounts by language. That needs testing before the claim is quoted.
- **The dossier exhibit on `feat/dossier-textsim` is not merged and A2 was not
  rerun on it.** It labels these pairs "near-identical words", which is false
  for most of them.

Caveats: model labels, n = 100; one snapshot; one encoder.
