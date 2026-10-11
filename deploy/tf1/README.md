# tf1: enrich worker deploy

tf1 (Hetzner, 2 vCPU, 3.7 GiB RAM) runs one service of this repo: compose project `analysis`,
service `enrich`, container `analysis-enrich-1`. Other containers on the host (gatus,
garmin-postgres, mealie, cadvisor, node-exporter, x-egress-proxy) are not ours: never touch them.

## Where the code comes from

- `~/kenya-monitor-2027-git`: clone of https://github.com/tom-h-f/kenya.git (https, read only).
  Created and fetched by `sync.sh`. Never edit it.
- `~/kenya-monitor-2027/analysis`: the build context, not a git checkout. `sync.sh` rsyncs only
  `Dockerfile .dockerignore .python-version pyproject.toml uv.lock README.md compose.yaml` and
  `src/ tests/ notebooks/` (the last three with `--delete`) from the clone.
- `~/kenya-monitor-2027/analysis/docker-compose.override.yaml`: host-local, comes from
  `deploy/tf1/docker-compose.override.yaml`. Holds `command: ["--loop", "--no-incitement"]` and
  `mem_limit: 2g`, so `compose.yaml` itself equals the repo. Compose loads it automatically.
- `~/kenya-monitor-2027/.env`: host-local, never synced. `R2_*` keys plus the now-unread
  `COORD/SCORECARD/ADJUDICATE_REFRESH_HOURS=0`. Backup `.env.pre-v1-retire-20261010` stays.
- Not synced, left alone: `collector/` (stale, not running on tf1), `*.bak*` files.

## Sync

From a checkout that contains `deploy/tf1/` (the ref being deployed must contain it too):

    ssh tf1 'bash -s -- --dry-run' < deploy/tf1/sync.sh
    ssh tf1 'bash -s -- --ref origin/master' < deploy/tf1/sync.sh

The dry run fetches the clone and prints the rsync itemised diff; it writes nothing else.
The real run: tgz backup of the tree to `~/backups/kma-tf1-<stamp>.tgz` (mode 600, holds `.env`)
plus a tag `analysis-enrich:pre-sync-<stamp>` on the running image, rsync, `docker compose build
enrich` with `GIT_SHA` stamped, wait for a gap between enrich passes (no `kma.enrich --once`
child, up to 90 min), `up -d --force-recreate --no-build enrich`, then verify (running, 0
restarts for 180 s, `GIT_SHA`, `--no-incitement`, `mem_limit`, cgroup `oom_kill` 0, no
Traceback/ERROR in the log). A failed verify rolls back automatically.

## Rollback

    ssh tf1 'bash -s -- --rollback ~/backups/kma-tf1-<stamp>.tgz' < deploy/tf1/sync.sh

Restores `src/ tests/ notebooks/` and the root files from the tgz, retags the pre-sync image as
`latest` and recreates `enrich` without a build. Without the image tag it rebuilds from the
restored tree. Backups and `.bak*` files are never pruned by the script.

## Checks after a sync

- `docker logs analysis-enrich-1`: `cycle N done`, `embedded`, `labelled`, `hate-scored`; no
  `incitement-scored`, no `coordination refreshed`, no `scorecards refreshed`.
- `docker stats --no-stream analysis-enrich-1` against the 2 GiB cap, and
  `/sys/fs/cgroup/system.slice/docker-<id>.scope/memory.events` `oom_kill 0`.
- `coded_suspect` while the incitement pass is off: see `docs/plans/2026-10-11-tf1-incitement-disable.md`.

## Build cost

The image uses `uv sync --frozen` from `uv.lock`; the layer is cached unless `pyproject.toml` or
`uv.lock` changed, which they have in the 2026-10-11 sync. Measured figures are in
`docs/plans/2026-10-11-tf1-sync.md`. `docker builder prune` reclaims the build cache; the host
disk was 80% used before the first sync.
