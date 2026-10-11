# Topics job: pi0 cron to mike-pc

Twice a week, Tuesday and Friday 02:10 (pi0 timezone Europe/London), assign topics to
newly embedded posts on mike-pc. Design: `docs/plans/2026-10-10-topics-incremental.md`.
Method: `docs/analysis/topics-incremental.md`.

Files on pi0 (`/home/tom/`): `kenya-topics-cron.sh`, `kenya_topics_r2.py`. Crontab:

    10 2 * * 2,5 KMA_TOPICS_REF=feat/topics-incremental /usr/bin/flock -n /tmp/kenya-topics.lock /home/tom/kenya-topics-cron.sh

Change `KMA_TOPICS_REF` to `master` once the branch is merged. Log:
`/home/tom/cron-logs/kenya-topics-YYYYMMDD.log`. Gauges:
`/var/lib/node_exporter/textfile/kma_topics.prom`.

On mike-wsl (`~/projects/kenya-topics/`): `repo/` (clone, the wrapper fetches the ref each
attempt), `pyproject.toml` + `.venv` (from `deploy/mike/pyproject.toml`), `cache/` (7.7 GB
mirror of embeddings and topics, the reference cache, per-fit calibration), `logs/`,
`job.lock`.

pi0 reaches WSL through its existing key for the Windows side (`ssh mike-pc "wsl -d Debian
-u tom -e bash -s"`, script on stdin); it has no key for WSL's sshd.

Retries: any failed step (wake, ssh, GPU busy, job non-zero, R2 check) backs off 60 s
doubling to 60 min, until 20 h after the scheduled start, then gives up and leaves
`kma_topics_last_attempt_ok 0`. Exit 4 (GPU OOM at the smallest block) retries once with
query block 256. A run still going at 110 min is left running and adopted by the next
attempt. Success needs the job's marker (`status=complete`, read-back rows equal assigned)
and pi0's own R2 listing to agree on every chunk size. mike-pc is never slept or shut down
here; `awake off` only when the flag is the one this job wrote and no run is active.

Alert: `KenyaTopicsStale` in the iot repo, 5.5 days without a success.

Test hooks (env): `KMA_TOPICS_FAIL_FIRST`, `KMA_TOPICS_DEADLINE_S`,
`KMA_TOPICS_BACKOFF_BASE_S`, `KMA_TOPICS_TEXTFILE_DIR`, `KMA_TOPICS_LIMIT`.

# Incitement job: pi0 cron to mike-pc

Twice a week, Wednesday and Saturday 02:10 (pi0 timezone Europe/London), score unscored posts
with the incitement NLI pass on mike-pc. Method and measurements:
`docs/analysis/incitement-backlog.md`. Same shape as the topics job above.

File on pi0 (`/home/tom/`): `kenya-incitement-cron.sh` only. It reuses the signing functions of
`kenya_topics_r2.py` (already there), so that file must stay. The manifest and verify code is
embedded in the wrapper. Crontab (one line, added after backing up the crontab to
`/home/tom/backups/crontab-20261010-pre-incitement.txt`):

    10 2 * * 3,6 KMA_INCITEMENT_REF=feat/incitement-backfill /usr/bin/flock -n /tmp/kenya-incitement.lock /home/tom/kenya-incitement-cron.sh

Change `KMA_INCITEMENT_REF` to `master` once the branch is merged. Log:
`/home/tom/cron-logs/kenya-incitement-YYYYMMDD.log`. Gauges:
`/var/lib/node_exporter/textfile/kma_incitement.prom`.

On mike-wsl (`~/projects/kenya-incitement/`): `repo/` (its own clone, so it never races the topics
job's checkout), `pyproject.toml` + `.venv` (from `deploy/mike/incitement/pyproject.toml`: torch
cu130, transformers 5), `cache/` (about 1 GB mirror of posts, incitement and relevance, plus
`runs/<run_id>/` with the plan and chunk files), `logs/`, `job.lock`.

Exclusion from the topics job: an attempt does not start while a `kt-*` tmux session exists or
the GPU holds more than 2000 MiB. The topics job has the same GPU check against this one.

Retries: as the topics job (60 s doubling to 60 min, 20 h from the scheduled start; a run still
going at 110 min is adopted by the next attempt). Exit 4 (GPU out of memory at the smallest batch)
retries once with batch 16. Success needs the marker (`status=complete`, read-back rows equal
distinct equal scored) and pi0's R2 listing to agree on every chunk size. `awake on` is taken
only when no hold exists, and `awake off` only when the flag is the one this job wrote.

Alert: `KenyaIncitementStale` in the iot repo, 5.5 days without a success.

Test hooks (env): `KMA_INCITEMENT_FAIL_FIRST`, `KMA_INCITEMENT_DEADLINE_S`,
`KMA_INCITEMENT_BACKOFF_BASE_S`, `KMA_INCITEMENT_TEXTFILE_DIR`, `KMA_INCITEMENT_LIMIT`,
`KMA_INCITEMENT_ATTEMPT_TIMEOUT_S`, `KMA_INCITEMENT_AWAKE_FOR`.
