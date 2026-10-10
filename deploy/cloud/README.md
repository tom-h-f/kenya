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
| `--text-trace` | `modal` | `modal`: reuse the trace the Modal job persisted, rank without text if absent; `local`: embed and build here; `none`: no text |
| `--no-embed` | off | with `--text-trace local`, skip the embed stage |
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

## The text trace

Default `--text-trace modal`. The text trace (embed the window's missing
posts, pairwise cosine search) is built by the Modal app `kma-text-trace`
(`analysis/modal_text_trace.py`, 02:30 UTC once its schedule is on) and left in
R2 under `coord2/kind=textsim`, keyed by window id. This host reads it back and
ranks. It holds no Modal credentials and imports no `modal`; R2 is the only
link. The pair search at full embedding coverage does not fit in `MemoryMax`
here, which is why it is not built on this host.

If the trace is not there at 03:30 the run does NOT build one (that is the
out-of-memory). It ranks without text similarity and says so:

- journal: a WARNING, `TEXT TRACE MISSING for window <id>`, with the key
  expected;
- `kind=daily_runs`: `text_trace` is `missing`, `text_threshold` is null (and
  null on the `kind=scores` rows too);
- the gauge file: `kma_v2_daily_text_trace_present 0`;
- `deploy/cloud/verify.sh postrun`: `text trace in the latest run` fails.

The unit still finishes successfully and the freshness gauge still advances:
the accounts are listed and the collector keeps its targets. The next day's
run is unaffected. To fill the day in, run the Modal job by hand
(`modal run modal_text_trace.py`), then re-run the unit: it reuses the
manifest and the trace and writes a new rank.

The Modal job's window must match this host's: it reads `DEFAULT_DAYS` from the
code deployed there, so redeploy `kma-text-trace` when `DEFAULT_DAYS` changes.

Deploy order: turn the Modal schedule on (or run the job by hand each day)
BEFORE this change reaches `cloud`, otherwise every run until then is a
`missing` run. `--text-trace local` in `ExecStart` keeps today's behaviour in
the meantime.

## The gauge

After the whole run succeeds the job writes, atomically (temp file in the same
directory, `chmod 644`, rename), `$KMA_TEXTFILE_DIR/kma_v2_daily.prom`:

```
# TYPE kma_v2_daily_last_success_timestamp_seconds gauge
kma_v2_daily_last_success_timestamp_seconds <epoch seconds>
```

A second sample sits in the same file, written in the same atomic rename:

```
# TYPE kma_v2_daily_text_trace_present gauge
kma_v2_daily_text_trace_present <1 if the rank used a text trace, else 0>
```

A failed or killed run leaves the previous file in place, so staleness is the
alert signal. An empty listing is a failure, not a healthy run. A rank without
a text trace is not a failure, so the freshness sample advances and the second
sample is what shows it; an alert on `kma_v2_daily_text_trace_present == 0` is
the iot stack's to add.

## Floors

The rolling rank keeps floor 2. `windowed.DETECTION` (day, 3) and
`windowed.SERIES` (week, 5) are the decided floors for per-day and per-week
windows and apply under `--windowed`. Whether the rolling rank's floor should
move is undecided; see the plan.

## Public dashboard summary

`kma.dashboard` reads the daily's persisted output and writes one public JSON
blob, `public/summary.json`, to the separate bucket `kenya-monitor-dashboard`.
The kenya.tomfr.net Worker binds that bucket only. The design is in
`docs/plans/2026-10-10-dashboard-v2.md`.

Not deployed. Units: `kma-dashboard.service`, `kma-dashboard.timer`,
`kma-v2-daily.service.d/dashboard.conf`.

| name | file | meaning |
|---|---|---|
| `R2_DASHBOARD_ACCESS_KEY_ID` | `/etc/kma/dashboard.env` | token scoped to `kenya-monitor-dashboard`, Object Read & Write |
| `R2_DASHBOARD_SECRET_ACCESS_KEY` | `/etc/kma/dashboard.env` | its secret |
| `R2_DASHBOARD_BUCKET` | `/etc/kma/dashboard.env` | defaults to `kenya-monitor-dashboard`; the producer refuses `kenya-monitor-2027` |

The read side reuses `/etc/kma/v2-daily.env`.

```sh
sudo install -m 0640 -o root -g tom /dev/null /etc/kma/dashboard.env      # then fill it in
sudo cp deploy/cloud/kma-dashboard.service deploy/cloud/kma-dashboard.timer /etc/systemd/system/
sudo install -d /etc/systemd/system/kma-v2-daily.service.d
sudo cp deploy/cloud/kma-v2-daily.service.d/dashboard.conf /etc/systemd/system/kma-v2-daily.service.d/
sudo systemctl daemon-reload && sudo systemctl enable --now kma-dashboard.timer
sudo systemctl start kma-dashboard.service && journalctl -u kma-dashboard -n 20
cd ~/kenya-monitor-2027/analysis && uv run --frozen --no-dev python -m kma.dashboard --out-dir /tmp/dash   # trial, publishes nothing
```

Runs after every successful daily (`OnSuccess=`) and once at 12:00 UTC. Measured
2026-10-10 on a Mac: 92 s, 344 MB peak, almost all of it footer reads over the
`posts/` prefix. The `cloud` box is not measured; `MemoryMax=1G`.

## Explorer bundle

`kma.explorer_export` reads the daily's persisted tables (`daily_runs`,
`daily_communities`, `daily_members`, and the new `daily_edges` and
`daily_traces`) and the window's posts, and writes the explorer bundle to
`kenya-monitor-dashboard` under `explorer/v1/`. Contract:
`docs/plans/explorer-contract/schema.json`. Design:
`docs/plans/2026-10-10-explorer.md`.

Not deployed. Units: `kma-explorer-export.service`,
`kma-v2-daily.service.d/explorer-export.conf` (adds to the dashboard drop-in; `OnSuccess=` is a list).
The unit is `After=kma-dashboard.service`, `Nice=15`, `IOSchedulingClass=idle`,
`MemoryMax=4G`, `TimeoutStartSec=45min`, retried at most 3 times in an hour. There is no timer: if the daily
fails the previous bundle stays and the manifest's stamp ages.

The daily now writes two more kinds, `daily_edges` (every fused edge, trace mask, largest weight,
shared retweets) and `daily_traces` (trace rows of kept members). A run made before this change has
no edges and the exporter refuses it.

Env: the same two files as the dashboard (`/etc/kma/v2-daily.env`, `/etc/kma/dashboard.env`). No new secret.

```sh
sudo cp deploy/cloud/kma-explorer-export.service /etc/systemd/system/
sudo cp deploy/cloud/kma-v2-daily.service.d/explorer-export.conf /etc/systemd/system/kma-v2-daily.service.d/
sudo systemctl daemon-reload
cd ~/kenya-monitor-2027/analysis && uv run --frozen --no-dev python -m kma.explorer_export --out-dir /tmp/explorer   # trial, publishes nothing
sudo systemctl start kma-explorer-export.service && journalctl -u kma-explorer-export -n 30
```

Safety: the bundle is staged, then checked (contract validator, language lint, id checks) before any
upload. Files under `runs/{run}/` go first, `manifest.json` last, then runs past the 30 kept are pruned.
A failure before the manifest leaves the previous run live. Remove an account by adding its id to
`explorer/denylist.txt`; it is gone from every file on the next run.

### Measured (Mac, 14 cores, home link, real R2 reads, `--out-dir`, nothing written to R2; not yet measured on `cloud`)

Window `daily-2026-10-09-21d`, daily tables from a local `rank` run of the same window.

| | |
|---|---|
| Exporter wall | 334 s (window posts 103, collection listing 55, profiles 93, assemble 138, safety 34) |
| Exporter peak RSS | 3.1 GiB (DuckDB `memory_limit` 2 GB); target under 4 GiB, so the margin is thin on a box that also counts page cache |
| Bundle | 154 files, 60.5 MB raw, 11.1 MB gzip. accounts 25.2 / 6.6, evidence 25.8 / 2.1, graph 7.1 / 1.6, hashtags 1.2 / 0.29, search 0.52 / 0.25 (raw / gzip MB) |
| Retention | 30 runs is about 1.8 GB raw (330 MB gzip) at this size |
| Daily `rank` with the edge persistence | 366 s, peak 2.79 GiB; the same rank without it 353 s, peak 2.72 GiB (+63 MB). Edge frame 0.7 s, 11 MB of trace rows held |
| Daily extra writes | `daily_edges` 459,744 rows, `daily_traces` for kept members |

The counts match the inventory: 10,941 accounts, 459,744 fused edges, 46 communities, 16 kept, 10,179
members, 256,691 co-retweet edges inside kept communities (19,095 with shared >= 2). The combined graph
holds 7,013 nodes and 41,846 edges: the default rule over the whole fused graph gives 7,547 and 42,445,
and the difference is edges touching accounts outside kept communities, which the contract keeps out.

`profiles` reads only `authors/` objects dated inside the window (1,894 objects): scanning all of them took
over half an hour and the signed requests expired. All 10,179 members had a snapshot in that range.
