# Why v2's top 500 is always one block, and what to do about it

Follows `2026-09-11-textsim-sensitivity/findings.md`, which found that every
text-trace variant hands the whole top 500 to one dense block - the Sheng-reply
mass, one co-retweet block, or a greeting farm.

## 1. It is not disconnection

`01_structure.py`, snapshot `2026-09-05-promotion-off`, bipartite traces built
once and cached (318 s).

| text trace | accounts | edges | components | largest | eigenvector mass: 90% on |
|---|---|---|---|---|---|
| current (mpnet 0.85) | 22,063 | 617,850 | 893 | 19,472 | 716 accounts |
| word-overlap floor | 20,480 | 423,125 | 939 | 17,551 | 433 accounts |

The largest component holds nearly every account and every trace: with the
floor, 370,221 co-retweet and 46,456 text edges. The next components are 102
accounts and smaller. So `coord2.centrality`'s documented hazard - the leading
eigenvector sitting on one component of a disconnected graph - is not what
happens here. The concentration is inside one component: the eigenvector
localises on its densest core.

All 31 authors of the reader's same-message pairs are in that component too.
With the floor, none of them is in the top 500 under any of the three
rankings below except PageRank (15).

| floored trace | components in top 500 | text-linked | campaign authors | Kenya share mean |
|---|---|---|---|---|
| global eigenvector (current) | 1 | 15 | 0 | 0.830 |
| per-component eigenvector x eigenvalue | 4 | 25 | 0 | 0.757 |
| PageRank | 2 | 217 | 15 | 0.608 |

## 2. Replacing the score fails the reproduction gate

`02_gate.py`: the IOHunter protocol with only the centrality swapped. Iran is
judged against the reference code's 71.31 +/- 0.76 (2026-09-11 Iran finding).

| country | target | global | per-component | PageRank |
|---|---|---|---|---|
| UAE | 84.66 | 84.64 | 84.44 | 84.79 |
| cuba | 57.92 | 57.92 | 57.92 | 60.06 |
| russia | 87.65 | 87.83 | 84.42 | 71.15 |
| venezuela | 95.05 | 95.05 | 89.85 | 55.43 |
| iran | 71.31 | 71.43 | 68.64 | 63.32 |
| china | 63.66 | 63.67 | 52.55 | 64.87 |
| within 2 SD | | **6/6** | 3/6 | 3/6 |

On the benchmarks the operations' accounts sit in the dense core, and the
global score is what finds them. So the score stays the paper's. What changes
is the report: `03_communities.py` splits the fused graph into communities and
ranks within each, keeping the global score on every row.

## 3. The production text trace, and what the community report lists

`coord2.clean_text` now strips @mentions and `text_similarity_network` takes a
word-overlap floor (`TEXT_MIN_OVERLAP` 0.10, char-4gram Jaccard on cleaned
text). Built on Modal over the snapshot: 309,391 eligible posts (25,632 fewer
than before - mentions no longer count toward four words), 50,085 user edges
over 12,737 users, at
`coord2/platform=x/kind=textsim/snapshot=2026-09-05-promotion-off/threshold=0.85/overlap=0.10/edges.parquet`.
Checked against an independent replay of the recorded 0.85 pairs with the new
cleaner and floor (`09_lexical_floor.py --texts`): 50,085 edges shared, weights
within 1.4e-6, and one replay-only edge at exactly 0.85 - the GPU rounding
boundary seen in the original check.

`03_communities.py` on that trace: 17,655 accounts, 398,694 edges, 872 Leiden
communities (modularity, resolution 1, seed 0), 125 of four or more accounts.
Listing each community's top 25 by its own leading eigenvector, communities in
order of their leading eigenvalue, fills the 500-account budget from 21
communities. 55 of the 500 are in the global top 500.

| | global top 500 | community listing |
|---|---|---|
| communities represented | 1 block | 21 |
| text-linked accounts | 15 | 154 |
| campaign authors (of 31) | 0 | 7 |
| Kenya share mean | 0.830 | 0.505 |

The 21 are three kinds: eleven co-retweet communities, most with listed Kenya
share 0.77-0.91 (two at 0.116 and 0.355); one large text community (3,008
accounts, 84% text edges, Kenya share 0.597); and five text communities with
Kenya share 0.000-0.015. What each is, is A2's question.

## 4. A2 on the community report

`04_a2_sample.py`: each listed community is one case (its listed accounts),
paired largest first with the unused v1 cluster nearest its size. 42 blinded
cases, 905 member rows, in `analysis/out/a2_communities/`. The sizes do not
match well: every v2 case is 25 accounts or fewer by construction, and v1's
remaining clusters run 40 down to 6, so later pairs set 25 against 6-13. v1's
largest clusters (177, 88) were not drawn, so v1 is not handicapped by pod
size this time; results are reported by size band regardless.

Read blind by the isolated headless reader (claude-sonnet-5, rubric
`2026-09-12-communities`, dossier text exhibit floored like the trace): 42
cases in 90 s, 0 failures. Verdicts persisted before unblinding at
`coordination/platform=x/kind=verdicts/dt=2026-09-12/run=20260912T100555Z.parquet`;
unblinded in `analysis/out/a2_communities/a2_unblinded.csv`.

| | cases | political | Kenya-relevant | verdicts |
|---|---|---|---|---|
| v1 | 21 | 2 | 3 | engagement_pod 17, political_campaign 2, news 1, fandom 1 |
| v2 | 21 | 6 | 12 | political_campaign 6, engagement_pod 5, fandom 5, unclear 3, news 2 |

- Nothing on either side was called an influence operation, or political at
  high confidence.
- v2's political verdicts: four Kenyan - the large text community (open
  pro-Sifuna campaigning with matching slogans), election and opposition
  amplification, vote-protection messaging, a partisan exchange - and two not:
  Tanzanian opposition, and Mexican anti-Morena hashtag copypasta.
- v2's five fandom verdicts are the off-domain text communities: K-pop and
  anime fan hashtag drives, a Mexican reality show, football transfer news.
  That is real matching-text coordination - the trace now finds what it is
  for - about other things.
- v1's cases are 17 engagement pods (greetings, motivational posts, buy-sell
  prompts) plus one pro-Sifuna cluster, one DCP cluster and one Kenyan-news one.
- The empty-evidence cases of 2026-09-11 are gone. v2's three unclears now
  carry evidence - overlaps of two to four members per object - and the reader
  found it too thin to call.

What survives, now on a text trace whose pairs share words and a report no
longer confined to one block: v2 surfaces Kenyan political activity where v1
surfaces engagement pods - 12 of 21 Kenya-relevant against 3, 6 political
against 2. What is new: a third of v2's listed communities are off-domain, so
the listing should order communities by Kenya share (or by the relevance
classifier being trained) rather than by eigenvalue alone.

Caveats kept attached: model verdicts; 21 cases per method; sizes mismatched
(v2 10-25 accounts, v1 6-40); one Leiden seed and resolution.

One local download of the UAE pickle from the `iohunter-bench` volume came back
with 86 zero-filled 64 KiB blocks and would not unpickle; a second download
loaded cleanly (sha256 d6ddec3c...). Check the hash before trusting a copy.

## 5. Ordering the listing by relevance

A2 found a third of the listed communities were off-domain. `03_communities.py`
now takes `--relevance-model` and `--min-kenya`: each account's share of posts
the classifier calls Kenyan (`db.relevance_source`, same shape as
`coord2_run.attach_relevance` uses the gate), averaged over a community's
members, and communities below the floor are dropped. A filter, not a re-sort:
the eigenvalue order is the coherence the detector fired on, and relevance only
decides whether it is our business.

The two separate cleanly. Kenyan communities score 0.92-0.98; the off-domain
ones 0.00-0.45. At a 0.5 floor, 104 of 125 communities drop, including the
823-account devotional group and the K-pop, anime and Mexican hashtag drives
A2 read.

| listing | communities | accounts | gate Kenya share (listed) mean / median | text-linked |
|---|---|---|---|---|
| eigenvalue only | 21 | 500 | 0.505 / 0.500 | 154 |
| + relevance floor 0.2 | 28 | 472 | 0.635 / 0.744 | 123 |
| + relevance floor 0.5 | 21 | 381 | 0.730 / 0.878 | 66 |

Two of the kept communities are exactly what the classifier was built for: the
gate puts them at 0.116 and 0.355 Kenya share, the model at 0.976 and 0.936.

At the 0.5 floor the budget is not spent - 381 of 500 accounts - because only
21 communities qualify and each contributes at most `--per-group` 25. Raising
the cap or lowering the floor spends the rest; both are triage-budget choices,
not detection ones.
