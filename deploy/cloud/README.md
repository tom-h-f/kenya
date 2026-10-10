# v2 daily detector on `cloud`

A systemd oneshot, 03:30 UTC, that runs `kma.v2_daily_run`: window manifest,
text trace, rank, `coord2/kind=daily_*`, `coord2/kind=scores`, then the
freshness gauge. No Modal. The design and the decisions behind it are in
`docs/plans/2026-10-10-cloud-v2-daily.md`.

Not deployed. Nothing here has been run on `cloud`.

## Host facts (read-only, 2026-10-10)

| | |
|---|---|
| OS | Debian 13, x86_64, systemd 257 |
| CPU / RAM / swap | 4 vCPU, 15 GiB, 4 GiB swap; the iot stack holds about 1.8 GiB |
| Disk | 31 GB free on `/` |
| Python | system 3.13.5; `uv` is not installed |
| Login user | `tom` (uid 1001, groups `tom`, `sudo`; not in `docker`) |
| Textfile dir | `/opt/iot-cloud/state/textfile`, `debian:debian`, mode 775 |

The textfile directory is the iot stack's node-exporter textfile directory
(`docker-compose.yml` mounts `./state/textfile` read-only at `/textfile`).
`tom` is not in group `debian`, so the unit sets `SupplementaryGroups=debian`:
the group write bit is enough to create `kma_v2_daily.prom`, and the iot
secrets directory (`debian` only, mode 700) stays out of reach.

## Install

```sh
# 1. code: the repo is public, clone over https
git clone https://github.com/tom-h-f/kenya.git ~/kenya-monitor-2027

# 2. uv, once
curl -LsSf https://astral.sh/uv/install.sh | sh        # installs to ~/.local/bin/uv

# 3. dependencies, once (and after a uv.lock change)
cd ~/kenya-monitor-2027/analysis && ~/.local/bin/uv sync --frozen --no-dev

# 4. secrets, by name only; see the table below
sudo install -d -m 0750 -o root -g tom /etc/kma
sudo install -m 0640 -o root -g tom /dev/null /etc/kma/v2-daily.env   # then fill it in

# 5. units
sudo cp deploy/cloud/kma-v2-daily.service deploy/cloud/kma-v2-daily.timer /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now kma-v2-daily.timer
```

The `uv sync` pulls the CPU torch wheel (the `uv.lock` Linux source) and every
other analysis dependency. `INSTALL httpfs` makes DuckDB fetch its httpfs
extension from duckdb.org on the first run, and `uv` reaches PyPI and the
PyTorch CPU index at install time. Those are install-time fetches; at run time
the only contact is R2.

## Env file contract

`/etc/kma/v2-daily.env`, loaded by systemd. Names only; values are not kept in
this repo.

| name | required | meaning |
|---|---|---|
| `R2_ACCOUNT_ID` | yes | R2 account |
| `R2_ACCESS_KEY_ID` | yes | S3 key id; read and write on the bucket |
| `R2_SECRET_ACCESS_KEY` | yes | S3 secret |
| `R2_BUCKET` | no | defaults to `kenya-monitor-2027` |
| `KMA_TEXTFILE_DIR` | set by the unit | directory for `kma_v2_daily.prom` |

The job writes `bench/`, `coord2/kind=textsim`, `coord2/kind=daily_*` and
`coord2/kind=scores`, so the key needs write on those prefixes. Nothing here
needs the X credentials or `ANTHROPIC_API_KEY`.

## Run, logs

```sh
sudo systemctl start kma-v2-daily.service        # a real run now
journalctl -u kma-v2-daily -f                    # logs; the last line is a JSON summary
systemctl list-timers kma-v2-daily.timer
cd ~/kenya-monitor-2027/analysis && uv run --frozen --no-dev python -m kma.v2_daily_run \
    --days 3 --out-dir /tmp/v2-scratch           # a trial: nothing is written to R2
```

The summary carries window id, post rows, per-stage seconds, `peak_rss_mb`,
and the keys written. A re-run on the same day reuses the manifest and text
trace for that window id and writes a new rank.

## Flags

| flag | default | |
|---|---|---|
| `--days` | 21 | window length; `--end` defaults to yesterday UTC |
| `--min-entities` | `coord2.MIN_ENTITIES_PER_USER` (2) | the rolling rank's activity floor |
| `--canary` | off | union the active planted canary into the window |
| `--windowed day\|week` | off | add the windowed pass at `windowed.DETECTION` / `SERIES` |
| `--out-dir` | none | write everything here, nothing to R2 |

To change a flag in production, add it to `ExecStart` in the unit.

## Sizing

Measured 2026-10-10 on a Mac (14 cores, 24 GB, CPU torch, home network to R2),
window ending 2026-10-09. The `cloud` row below is the 4 vCPU x86 box itself.

| window | post rows | wall | peak RSS | text vectors (coverage) | trace edges |
|---|---|---|---|---|---|
| 1 d | 50,031 | 239 s | 4.0 GB | 2,812 (14%) | 84 |
| 3 d | 195,794 | 225 s | 4.1 GB | 7,331 (12%) | 302 |
| 7 d | 455,792 | 283 s | 6.6 GB | 14,396 (12%) | 506 |
| 30 d | 2,469,983 | 802 s | 6.3 GB | 139,995 (24%) | 10,451 |
| 90 d | 5,295,770 | 2,166 s | 13.8 GB | 678,506 (60%) | 137,656 |
| 30 d, on `cloud` | n/a | 1,576 s (textsim 587, rank 710) | 11.85 GiB | n/a | 10,451 |

- `MemoryMax=12G` caps the job below the iot stack's needs. On `cloud` the
  30 d window peaked at 11.85 GiB, 0.15 GiB under the cap, at 24% embedding
  coverage, so the default is 21 d. Its peak on `cloud` is not yet measured.
- Memory grows with embedded vectors (about 13 KB each between the 30 d and
  90 d runs) and the pair search is quadratic in them. Embedding coverage was
  12 to 60% when measured; when the enrich backlog clears, a 30 d window holds
  about 584,000 vectors and will exceed 12 GB. A 21 d window needs re-measuring then.
- On this Mac the 90 d pair search ran at 678,506 vectors in 555 s. A 4 vCPU
  x86 host will be slower by a factor that has to be measured on the box.
- `--windowed` added 3.3 GB peak and 36 s on the 7 d window.

## The gauge

After the whole run succeeds the job writes, atomically (temp file in the same
directory, `chmod 644`, rename), `$KMA_TEXTFILE_DIR/kma_v2_daily.prom`:

```
# TYPE kma_v2_daily_last_success_timestamp_seconds gauge
kma_v2_daily_last_success_timestamp_seconds <epoch seconds>
```

A failed or killed run leaves the previous file in place, so staleness is the
alert signal. An empty listing is a failure, not a healthy run.

## Floors

The rolling rank keeps floor 2. `windowed.DETECTION` (day, 3) and
`windowed.SERIES` (week, 5) are the decided floors for per-day and per-week
windows and apply under `--windowed`. Whether the rolling rank's floor should
move is undecided; see the plan.
