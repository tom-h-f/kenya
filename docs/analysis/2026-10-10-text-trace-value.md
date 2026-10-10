# How valuable is the text trace in v2?

Measured 2026-10-10 on window `daily-2026-10-09-30d` (run `20261010T170630Z`)
and the six IOHunter release datasets. Scripts and notes:
`analysis/investigations/2026-10-10-text-trace-value/` (outputs in untracked
`out/`; no account ids or handles appear here, per
`2026-09-13-publishable-statistics.md`). Nothing was written to R2.

Two coverage states are reported throughout. **Manifest coverage** is what the
live run saw: 139,995 of 583,982 text-eligible posts embedded (24.0%). **Full
coverage** is the embeddings on R2 at 20:00 BST after the backfill: 583,982 of
583,982 (100%). The window itself is the 30 d one, not the new 21 d default.

Reproduction check: the manifest-coverage pair search reproduces the live
trace exactly (10,451 day-edges, 9,594 account pairs, max weight difference
1.5e-6, no edge in either that is not in the other).

## Q1 Contribution to the listing

Method: same four behavioural networks, `coord2.fuse` -> `centrality` ->
`coord2_communities.report` at production defaults, with and without the text
network (`03_arms.py`). Noise floor: `04_null.py` re-ranks with other Leiden
seeds and with random co-retweet edges removed (`rank.py`).

| | manifest cov. | full cov. |
|---|---|---|
| text edges after floor | 4,701 | 48,767 |
| accounts with a text edge | 2,825 | 11,265 |
| accounts whose ONLY edges are text | 2,097 | 8,997 |
| listed, no text / with text | 294 / 357 | 294 / 254 |
| listed communities, no text / with text | 16 / 21 | 16 / 16 |
| listing Jaccard, no text vs with text | 0.318 | 0.277 |
| listed accounts with any text edge | 60 (16.8%) | 67 (26.4%) |
| listed accounts with ONLY text edges | 48 (13.4%) | 41 (16.1%) |
| listed in communities that fall apart without text-only edges | 50 (14.0%) | 37 (14.6%) |
| such communities | 6 of 21 | 6 of 16 |

- The overlap between the with-text and without-text listings is low (0.28-0.32)
  but so is the listing's overlap with itself under no text change: other Leiden
  seeds give 0.27-0.44 (no text) and 0.30-0.53 (with text); removing the same
  number of random co-retweet edges gives 0.20-0.37 (full-coverage run: 0.20-0.37,
  manifest run: 0.23-0.33). The change in listing overlap cannot be attributed to
  text.
- What can be attributed is the set of communities that exist only through text
  edges: 6 communities, 37-50 listed accounts (about 14.5% of the listing), 4-28
  members each. They are absent without the text trace. The other 10-15 listed
  communities have a behavioural core (largest component without text-only edges
  is at least 4 and usually most of the community).
- The global top-500 by centrality is the same with and without text at both
  coverages (500 of 500): text never reaches the paper's own ranking, only the
  community report.
- At full coverage one community of 7,242 accounts is 78% text edges and holds
  25 listed accounts; it still has a 1,880-account behavioural component, so it
  is not text-only, but text now glues it together.

## Q2 Benchmark

Method: `10_benchmark.py`, the repo's own `iohunter_gate.score_nodepruning`
(unsupervised Macro-F1, the metric `coord2.detect` is gated on). The fused graph
is rebuilt as the union of the release's trace networks; all-five reproduces
the released graph edge for edge in all six. Text is `tweetSim`. Release traces
are at cosine 0.7, so this is the release's text trace, not ours.

| dataset | published | all five | no text | delta | text pair edges |
|---|---|---|---|---|---|
| UAE | 84.66 | 84.64 | 76.70 | -7.94 | 1,998,495 |
| cuba | 57.92 | 57.92 | 57.45 | -0.47 | 265,390 |
| russia | 87.65 | 87.83 | 87.83 | 0.00 | 0 |
| venezuela | 95.05 | 95.05 | 95.05 | 0.00 | 0 |
| iran | 60.83 | 71.43 | 71.43 | 0.00 | 0 |
| china | 63.66 | 63.67 | 63.67 | 0.00 | 0 |

- Without text, 5 of 6 still reproduce (within 2 published SD); UAE falls out
  (-7.96 against published). Iran failed before (known) and is unchanged.
- Only UAE and cuba have text. In the other four the release's `tweetSim`
  network holds one self-loop per positive account and no pair edge: it labels
  positives and does nothing in the centrality (dropping the loops changes
  nothing). Those four say nothing about text value.
- Coverage of the benchmark is mostly about co-URL: leaving out coURL moves
  russia -23.5, venezuela -45.2, china -14.6; text alone moves only UAE.
- Caveat: Kenya has almost no URL signal (v2-findings section 3), so benchmark
  reliance on coURL does not transfer. UAE is the one dataset where text is the
  dominant trace (72.8% of edges).

## Q3 Precision of text edges

Method: every post pair behind the live-method trace (cosine >= 0.85, char
4-gram >= 0.10, mentions stripped, cross-author), `05_precision.py`. A stratified
blind read of 60 pairs (15 per cosine stratum, one pair per account pair), by
the 2026-09-11 reader and prompt unchanged (`06_read.py` wrapping
`06_pair_audit.py`, headless claude-sonnet-5, isolated). The read used the
manifest-coverage trace; the 80-pair budget was then spent, so full coverage is
measured by overlap metrics and the read is reweighted by stratum.

| | 2026-09-11 (no floor) | 2026-10-10 manifest cov. | full cov. |
|---|---|---|---|
| post pairs | 279,060 (top 500) | 14,526 | 131,241 |
| median token Jaccard | 0 | 0.375 | 0.323 |
| share under 0.1 token overlap | 84.6% | 6.6% | 9.0% |
| share of near-copies (>= 0.5) | 3.0% | 38.8% | 33.4% |
| share identical after cleaning | - | 14.7% | 10.1% |
| account pairs resting on a single post pair | - | 91% | 87% |

Blind read, 60 pairs: **45 same message, 14 same topic, 1 unrelated**, against
26 / 15 / 59 on 2026-09-11. By cosine stratum, same message / same topic /
unrelated: 0.85-0.88: 9/5/1; 0.88-0.92: 11/4/0; 0.92-0.96: 11/4/0; 0.96-1.00:
14/1/0. Reweighted by each coverage's stratum mix, same-message is about 72%
at manifest coverage and 69% at full coverage. Lower strata are weaker, as
expected: 49% of full-coverage pairs sit in 0.85-0.88 where 13% are
near-copies and 14% share under a tenth of their words.

Caveats: one reader (model), 60 pairs, 15 per stratum; the reader may lean on
shared words; the same-message share at full coverage is a reweighting, not a
read. The 2026-09-11 figures were a random sample behind the top 500 with no
floor, so the comparison shows the floor and mention stripping worked, not that
the encoder improved.

## Q4 Unique value

"Text-only" means in the text network and in none of the four thresholded
behavioural networks (`07_unique.py`).

- Manifest: 2,097 of 2,825 text-edge accounts (74%) are text-only; full: 8,997
  of 11,265 (80%). Median 5-8 posts in window (text-only), against 30-48 for
  accounts that also have behavioural edges. Largest text-only component
  inside the text network: 4,741 accounts at full coverage (28 members at
  manifest coverage); 53 and 136 components of 4+.
- Listed: 41-48 listed accounts are text-only (13-16% of listing); 37-50 sit
  in the six communities with no behavioural core.
- Kenya relevance (relevance model, share of the account's scored posts at the
  model threshold): listed text-only 0.85-0.88 mean, 93-96% at 0.5 or above;
  the whole listing 0.90-0.92; but ALL text-only accounts in the graph only
  0.46-0.48 (54% at or above 0.5). The relevance gate in the community report
  is what makes the listed ones Kenyan; the raw text network is mostly not.
- Blind read of 20 account pairs from the text-only communities (manifest
  coverage): 17 same message, 2 same topic, 1 unrelated. Their post pairs have
  median token Jaccard 0.60-0.85 and 17-27% are identical after cleaning, and
  those communities are 4-28 accounts each posting few distinct texts (50
  distinct cleaned texts across 82 edges at full coverage): template or copy-
  paste behaviour. Whether it is campaigning, pods or greeting farms is not
  judged here (A2-style adjudication was not rerun; budget spent).
- Caveat: the 20-pair read sits inside the same 80-pair budget as Q3, which
  makes each cell small.

## Q5 Cost

- Cloud CPU, 30 d, manifest coverage: textsim 587 s of 1,576 s total (37%),
  rank 710 s, peak 11.85 GiB (docs/plans/2026-10-10-cloud-v2-daily.md).
- Mac CPU, this study, full coverage (`02_pairs.py`, 584k vectors, nothing
  persisted): 2,427 s wall, of which 965 s loading embeddings and 1,196 s pair
  search (CPU, shared with other jobs for part of it), peak RSS 8.77 GiB. At
  manifest coverage the same search took 14 s. The search is quadratic in
  vectors. The `cloud` box is not measured at full coverage and has a 12 GiB
  cap; whether 30 d or 21 d fits there needs to be measured.
- Modal `kma-textsim`: $0.035-0.149 per run (docs/plans/2026-10-10-modal-budget.md),
  measured on snapshots at partial coverage; full-coverage cost not measured.
  Month-to-date Modal spend was $1.46 on 2026-10-10 (`kma.modal_spend`); this
  study spent none.
- Value per cost: text is 37% of cloud wall time and most of its memory growth
  for 14.5% of the listing; at full coverage it is the step that no longer fits
  the cloud host cheaply.

## Q6 Short text window

Method: from the same full-coverage post pairs, rebuild the text network for
(a) edges whose later post is in the last N days (the `bucket=day` axis, after
a whole-window search), (b) pairs with both posts created in the last N days,
(c) both posts collected in the last N days (what an N-day window would pin).
Then rank (`03_arms.py`, `08_window.py`). The full-coverage 21 d window was not
rebuilt; c-arms on the 30 d window are the proxy.

| arm (full coverage) | share of text edges | text-dependent listed kept (of 37) | full-added listed kept (of 135) |
|---|---|---|---|
| a7 | 23.9% | 10 | 35 |
| c7 | 14.1% | 6 | 7 |
| a14 | 52.1% | 18 | 22 |
| c14 | 34.2% | 8 | 35 |

A real 7 d text window holds about 14% of the 30 d text edges (a-axis 24%) and
keeps 6-10 of the 37 text-dependent listed accounts (16-27%); 14 d keeps 8-18.
At manifest coverage the picture is the same or worse (c7 keeps 0 of 50; only
a14 keeps any). The retained shares are no better than the noise floor in Q1
for the "added" accounts, so the short window mostly loses the six text-only
communities. Memory: the text step scales with vectors in the text window
(roughly 13 KB per vector plus the quadratic search), so a 7 d window holds
roughly a quarter of the vectors, not measured here.

## Recommendation

| option | verdict |
|---|---|
| Keep as is | Not on the cloud box without a measurement at full coverage. |
| Shorten the text window to 7 d | No. Keeps 16-27% of text-dependent accounts for about 14-24% of the edges, and the 7 d set is not a clean subset. |
| Drop the text trace | No, but it is a smaller component than assumed. 85% of the listing does not need it; the benchmark needs it on one dataset of six (UAE -7.9). |
| Make it conditional / offload | Yes: keep the trace, same 21 d window, but run the text trace off the cloud host (Modal A10G or mike-pc, $0.04-0.15 per run measured at partial coverage) and let the cloud job reuse it; skip it when the daily is memory-bound. |

What justifies it: precision is now high (same message 69-72%, 1 of 60
unrelated), the six text-only communities (37-50 accounts, 14.5% of the
listing, 92-96% Kenyan after the relevance gate) are found by no other trace,
and the text edges are 10x more numerous at full coverage (9.6k to 93k account
pairs), which makes the step the expensive one.

What would change the call: drop if a human or A2 adjudication of the text-only
communities calls them engagement pods or greeting farms rather than
campaigns; keep on the cloud host if a measured full-coverage 21 d run fits
12 GiB; shorten only if per-day text-only communities (a windowed pass) prove
to be the product.

## Not measured

- `cloud` host at full coverage; Modal cost at full coverage; the 21 d window
  at full coverage.
- Per-account verdicts: nothing here says a text-linked account is inauthentic.
- Blind read is 60 + 20 pairs of one model; none at full coverage.
- Listing churn: the community report's listing is unstable run to run
  (Jaccard about 0.3 under an unchanged text trace), so overlap alone cannot
  rank a trace's value.
