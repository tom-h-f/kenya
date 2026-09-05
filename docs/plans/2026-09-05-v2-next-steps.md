# v2 next steps

Execution order for [2026-09-05-v2-methodology.md](2026-09-05-v2-methodology.md).
Written 2026-09-05, after the six decisions were taken and the deployed-code
recovery landed on master.

Ordering principle: **the benchmark is the product of the next few weeks**, not
the Kenya detector. Nothing touches the Kenya corpus until the reproduction gate
passes, because a detector we cannot score is what we already have.

---

## Step 0 - done 2026-09-05

- Deployed code recovered into git. tf1's adjudication layer and pi0's two
  unpushed commits (twscrape 0.20.1, SearchTimeline POST) are on master, along
  with `fix/collector-seed-oom`, which is what pi0 was actually running.
- Cluster promotion switched off and deployed to pi0
  (`CLUSTER_PROMOTION_ENABLED=0`).
- Control-set request sent to the authors.

## Step 1 - the frozen snapshot (do first, blocks nothing else)

Cheap, and every later comparison is meaningless without it.

- `kma.bench.snapshot(name)` writes `bench/snapshot=<name>/manifest.parquet`:
  every object path, size, etag, row count, per prefix, plus the collector
  `code_version` range it spans.
- A snapshot is a manifest over immutable per-run objects, not a copy.
- **Take the first snapshot immediately and name it after the promotion
  switch-off**, because that date splits the corpus into two populations
  (§11 of the methodology doc). Anything spanning it must split on it.
- Acceptance: two `latest_posts` reads pinned to the same snapshot id return
  identical row counts a week apart.

## Step 2 - acquire and slice the IO archive (Modal)

- Pull `ioa_tweets.csv` (105.9 GB, all 21 countries) and `ioa_users.csv`
  (15.7 MB) to a Modal volume.
- **Verify before building anything:** that user ids of accounts under 5,000
  followers are stable pseudonyms rather than per-row hashes (networks die
  otherwise), and that tweet text survives anonymisation (the text-similarity
  trace dies otherwise). If either fails, the plan changes and it is better to
  know in week one.
- Slice to the paper's six countries and write per-country Parquet. Report row
  counts against the paper's Table 1 (China 5,191 drivers / 13.8M tweets, Cuba
  503 / 4.8M, Egypt & UAE 240 / 1.5M, Iran 209 / 9.9M, Russia 3,487 / 9.8M,
  Venezuela 33 / 9.5M). A large mismatch means the mirror is not what the paper
  used, and that has to be resolved before the gate means anything.

## Step 3 - the control group (the actual critical path)

Route A is sent and waiting. Route B starts now regardless, because a null
result on A costs weeks.

- Build the Stream Grab path: fetch the monthly archives covering the driver
  windows, filter to accounts using the drivers' hashtags in the same periods,
  cap at 100 tweets per account to match the paper's construction.
- Record the sampling difference in the harness itself, not just in prose: the
  benchmark should report which control provenance produced each number.
- **Stop rule:** if neither route yields a usable control set, the supervised
  half of the method is unavailable and the plan falls back to unsupervised
  centrality only, scored by rank rather than by AUC. Decide that explicitly
  rather than drifting into it.

## Step 4 - implement `kma.coord2` and pass the gate

Build order is deliberate: cheapest trace first, hardest last.

1. `co_url`, `hashtag_sequence`, `co_retweet` - the three that share the
   TF-IDF bipartite recipe. One builder, three entity extractors.
2. `fast_retweet` - needs the 60s threshold and retweet timestamps.
3. `text_similarity` - the odd one out, and the one whose parameter we are
   re-deriving per embedding space.
4. Fusion by edge union, unweighted eigenvector centrality, prune at 10^-2.
5. node2vec (128d, 16 walks, 16 steps) + Random Forest, 10-fold CV.

**Validate at small scale first:** Egypt & UAE only (240 drivers, 370 controls,
1.9M tweets). It is the smallest campaign and the cheapest way to be wrong.
Only after it reproduces do the other five run.

**Gate:** fused unsupervised AUC ~0.83 / F1 ~0.76; supervised AUC ~0.94 /
F1 ~0.82 / precision ~0.96, within a tolerance stated before the run, not after.

## Step 5 - v1 on the same benchmark

Per the decision, a full comparison rather than an overlap anchor. Run
`kma.coordination` against the same six campaigns and the same control set.

State the caveat with the result, both directions: v1's degree-corrected null
assumes near-total census incidence and the archive is not a census, so part of
any gap is structural rather than methodological.

## Step 6 - transfer to Kenya

- Run v2 over the frozen snapshot. Report: accounts above threshold, their
  overlap with v1's last published clusters, and their Kenya share.
- The `measure` relevance gate is still unvalidated (OBJECTIVES A1) and it is
  the cheapest remaining measurement in the project: hand-label a few hundred
  posts and report precision and recall. Do it here, because v2's output is
  filtered through it.

## Step 7 - re-point collection, then republish

- Promotion comes back on, reading v2 centrality ranks rather than clusters.
  `adaptive.py` consumes handles, so this is a source swap, not a rewrite.
- Dashboard series restarts on v2 output. Publish standardised or
  within-partition rates only.

---

## What is deliberately not being done

- No live parallel collection arms and no random control arm. Replay on the
  frozen corpus is the only collection test for now, with its incumbent bias
  stated on every result.
- No Elmas-style trend forensics yet, despite it matching Kenya's documented
  attack pattern more closely than IO detection does. It is a separate track and
  merging it into v2 would defeat the point of copying one methodology.
- No IOHunter. It becomes a drop-in once the fused network exists.

## The risk that should be re-read at each step

The north star is recall against state-backed operations. Kenya's documented
problem is domestic disinformation-for-hire. Every step above optimises for the
former. That is a deliberate choice made for the sake of having ground truth at
all, and it should be restated in anything published, not quietly carried.
