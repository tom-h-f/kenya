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
