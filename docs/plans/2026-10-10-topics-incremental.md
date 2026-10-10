# Incremental topic assignment, twice a week, on mike-pc

2026-10-10. `topics/` holds one fit (2026-09-04, 663,294 posts). Every post
embedded since has no topic, and a full UMAP+HDBSCAN refit over the whole
corpus does not fit the 3070's 8 GiB. This adds a cheap job that gives each
unassigned embedded post the topic of its nearest fitted posts.

## Shape

`kma.topics_assign` (`python -m kma.topics_assign`):

- Reference: posts with `topic >= 0` from the newest fit run (a run whose
  `params` has no `method=knn`). Fit posts are never reassigned.
- Pending: embedded posts with no row in the fit and no row from a knn run
  against the same fit. Knn rows made against an older fit count as unassigned.
- Cosine kNN on GPU, fp16 reference, `k` neighbours, similarity-weighted vote,
  `-1` when the winner's support is below a threshold. `k`, vote and threshold
  are chosen by a held-out test on the fit itself (below), not by guess.
- Output: same `topics/platform=x/model=<slug>/dt=<day>/run=<id>-cNNNN.parquet`
  schema, same five columns. `params` starts `method=knn,k=..,threshold=..,
  fit_run=..,heldout_agreement=..`. `load_topics` gains a derived
  `assignment_method` (`fit`|`knn`) column. Newest `assigned_at` wins, so the
  next full refit supersedes every knn row.
- Idempotent: pending is recomputed from R2's listing each attempt, chunks are
  separate objects, a crash leaves complete chunks and the rerun does the rest.
  Completion is a JSON marker under `topics/_assign/` (outside the readers'
  glob) carrying counts and the job's own read-back count.

## Credentials: none on mike-wsl

mike-wsl has no R2 credentials, and the embeddings backfill there was built to
need none (`02_encode.py`: "Needs no R2 credentials"). Putting long-lived keys
on a machine shared with two other Windows users is Tom's call, so it is not
done here. pi0 already holds the collector's R2 keys. Each attempt, pi0 signs
short-lived (6 h) GET urls for the embeddings and topics objects and PUT urls
for this run's chunk and marker keys, and pipes that manifest to mike-wsl over
ssh (0600 file, deleted after the attempt). The job needs nothing else.

## Scheduler (pi0 cron -> mike-pc)

`deploy/pi0/kenya-topics-cron.py`, one crontab line, `flock -n`. Per attempt:
wake mike-pc if needed (never sleeps it), sync the clone on mike-wsl, check the
GPU is idle, `awake on` with margin, run the job detached in tmux under its own
flock, poll, verify against R2 independently, then release the hold only if we
set it. Failure of any step backs off exponentially (60 s doubling, cap 60 min)
until 20 h after the scheduled start, then gives up loudly. Gauge file
`kma_topics.prom` on pi0.

## As built (2026-10-10)

Calibrated k=10, weight sim**16, share >= 0.40, topics+noise reference, held-out topic
agreement 0.8017. First run: 1,803,207 posts in 224 s, 1.47 GiB peak. pi0 reaches WSL
through its existing Windows-side key (`wsl -e bash -s`), because it has no key for WSL's
sshd. Schedule Tue/Fri 02:10 Europe/London. Details: `docs/analysis/topics-incremental.md`,
`deploy/pi0/README.md`.

## Not here

- The periodic full refit (Modal A100, about $5). See "When to refit" in
  `docs/analysis/topics-incremental.md` for the drift measure.
- Anything in `coord2.py`, `textsim_run.py`, `v2_daily.py`, `window.py`.

## Open

- R2 keys on mike-wsl instead of presigned manifests: Tom's decision. It would
  remove the manifest step and nothing else.
