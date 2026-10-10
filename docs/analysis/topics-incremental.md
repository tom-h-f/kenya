# Incremental topic assignment (`kma.topics_assign`)

Posts embedded after the topic fit get the topic of their nearest fitted posts.
The fit itself (UMAP + HDBSCAN, Modal A100, `modal_backfill.py --topic`) is
unchanged and is still the only thing that creates or reshapes topics.

## What it does

Reference: every post in the newest fit run (`params` without `method=knn`), 663,294
posts for the 2026-09-04 fit, with its fit label (`-1` included). A fit post is never
reassigned. Pending: embedded posts with no row in the fit and no knn row against the
same fit. Cosine kNN over the 768-d embeddings on GPU (fp16 reference, 0.95 GiB),
k=10, weight `sim ** 16`, label = heaviest topic, and `-1` when the winner's share of
the total weight is under 0.40.

Output goes to the same prefix and schema:
`topics/platform=x/model=<slug>/dt=<day>/run=<runid>-cNNNN.parquet` (50k rows per
chunk), `params` = `method=knn,k=10,threshold=0.40,fit_run=<fit>,heldout_agreement=0.8017,...`.
A JSON completion marker per run sits under `topics/_assign/` (outside every reader's
glob). `semantic.load_topics` adds `assignment_method` (`fit` or `knn`), derived from
`params`; `topic_summary` is unchanged and works on the combined frame. Newest
`assigned_at` wins, so the next full fit supersedes every knn row.

## Calibration (measured 2026-10-10)

5% of the fit held out (stratified by topic, 33,136 posts, seed 42), predicted from the
other 95%, grid over k in {1,3,5,10,20,30,50}, weight power {0,1,4,16}, three confidence
features, thresholds 0.30 to 0.95. Chosen by balanced accuracy (mean of topic recall and
noise recall).

| reference | k | power | feature | thr | topic recall | noise recall | balanced |
|---|---|---|---|---|---|---|---|
| topics only | 10 | 16 | share | 0.84 | 0.7206 | 0.8329 | 0.7767 |
| topics + fit noise as `-1` voters | 10 | 16 | share | 0.40 | 0.8017 | 0.9229 | 0.8623 |

Chosen: the second. Topic recall by fit topic size: under 100 posts 0.7317, 100-500
0.7965, over 500 0.8493. Precision of an assigned topic 0.8652. fp16 vs fp32 reference
gives the same winning label for 98.95% of queries. The grid is in
`calibration-<fit_run>.json` in the job cache.

## Not the fit

It cannot make a new topic, split or merge one, or move a boundary. A narrative that
did not exist on 2026-09-04 lands in `-1` or in the nearest old topic. Readers that need
the fit's own counts filter `assignment_method = 'fit'`.

## When a full refit is needed

Measure, do not guess. Every run's marker records `topic_ge0` and `noise`. Baseline:

- held-out fit posts: 40.3% got a topic (fit itself: 43.6% in a topic)
- first real run, 1,803,207 posts embedded since 09-04: 33.3% got a topic (599,749)

The knn topic share on new posts is already 7 points below the held-out figure after
5 weeks. Refit when that share keeps falling run over run, or when the share of new posts
whose best neighbour is below the calibrated floor rises, or when a reader finds a
live narrative sitting in `-1`. Where to set the number is open: it needs a few more
runs to see the trend. The refit is the Modal job in `modal_backfill.py` (about $5).
After a refit the next run calibrates again on the new fit and starts clean.

## Running it

mike-wsl holds no R2 credentials. pi0 signs short-lived urls (`deploy/pi0/kenya_topics_r2.py`)
and the job reads and writes through them. Layout, cron line and retry behaviour:
`deploy/pi0/README.md`.

Measured 2026-10-10 on the RTX 3070: 1,803,207 posts in 224 s once the reference was
cached, peak 1.47 GiB VRAM (limit 7 GiB), 37 chunks.
