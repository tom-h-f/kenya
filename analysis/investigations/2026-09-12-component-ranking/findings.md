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

One local download of the UAE pickle from the `iohunter-bench` volume came back
with 86 zero-filled 64 KiB blocks and would not unpickle; a second download
loaded cleanly (sha256 d6ddec3c...). Check the hash before trusting a copy.
