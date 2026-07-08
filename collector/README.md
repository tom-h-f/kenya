# kenya-monitor collector

Scrapes X (Twitter) activity around the 2027 Kenyan general election into
Cloudflare R2 as partitioned Parquet. Runs on **pi0** via Docker (`monitor run`).

**Full collection docs:** [docs/collection/README.md](../docs/collection/README.md)
(commands, R2 layout, state files, env vars, CIB rationale).

## Setup

```bash
uv sync
cp ../.env.example ../.env   # R2 credentials
cp config/accounts.example.yaml config/accounts.yaml
uv run monitor check
uv run monitor accounts add
```

## Commands (summary)

| Command | What it does |
|---------|--------------|
| `monitor run` | Always-on worker: continuous cycles of posts + snowball + metrics + follow-crawl, throttled only by per-account pacing + twscrape rate limits |
| `monitor run --once` | Single posts pass |
| `monitor collect x --keywords` | One-off keyword search |
| `monitor backfill` | Deep historical search sweep |
| `monitor snowball` | Retweeter census + reply threads + hydration |
| `monitor metrics` | Refresh engagement on top posts |
| `monitor adapt` | Promote bursting hashtags / cluster accounts |
| `monitor follows` | One-shot follower/following fetch |
| `monitor crawl-follows` | Recursive BFS follow-graph crawl |
| `monitor stats` | Volume summary from R2 |
| `monitor accounts sync` | Refresh scraping account pool |

## Storage

Immutable per-run Parquet under `r2://kenya-monitor-2027/`. See
[docs/collection/README.md](../docs/collection/README.md#r2-layout) for all
prefixes.

## Config

- `config/targets.yaml` - curated keywords and timeline accounts
- `config/accounts.yaml` - twscrape account pool (gitignored in prod)
- `state/` - crawl ledgers, dynamic targets, snowball TTLs (gitignored)
