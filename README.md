# kenya-monitor-2027

Monitors social media activity around the 2027 Kenyan general election. Scraped posts
land in Cloudflare R2 as partitioned Parquet; analysis is done with DuckDB.

## Disinformation-analysis roadmap

Phased build in `docs/analysis/`:

- Phase 0 - collector completeness (done): structured post fields + full capture.
- Phase 1 - account authenticity / bot scoring.
- Phase 2 - semantic / narrative layer (embeddings, sentiment, stance, topics).
- Phase 3 - coordination networks (coordinated inauthentic behaviour).

References: [what_are_embeddings](https://github.com/veekaybee/what_are_embeddings),
[DuckDB VSS](https://duckdb.org/docs/current/core_extensions/vss),
[sentence-transformers](https://sbert.net/docs/sentence_transformer/pretrained_models.html),
[embedding-atlas](https://apple.github.io/embedding-atlas/overview.html).

## Layout

```
kenya-monitor-2027/
  .env              # shared R2 credentials (gitignored) - used by both projects
  .env.example
  collector/        # scraper app + CLI (twscrape -> R2). Runs on pi0 (residential IP) via Docker.
  server/           # DuckDB quack server on tf1: serves R2 over the tailnet (token auth).
  analysis/         # DuckDB/Polars/marimo workspace. Query R2 directly, or via the tf1 server.
  docs/collection/  # collector methods, R2 layout, env reference
```

Two independent `uv` projects, one shared `.env`. Collector docs:
[docs/collection/README.md](docs/collection/README.md).

## Quick start

```bash
cp .env.example .env        # fill in R2 S3 credentials

# collect
cd collector && uv sync
uv run monitor check                                    # verify R2 round-trip
uv run monitor collect x --query "#KenyaDecides2027" --limit 20

# analyse
cd ../analysis && uv sync
uv run marimo edit notebooks/explore.py
```

## Storage model

Immutable per-run Parquet files in R2, Hive-partitioned; no database. Dedup and
engagement-over-time are reconstructed in DuckDB at read time.

```
r2://kenya-monitor-2027/posts/platform=x/type=search/dt=YYYY-MM-DD/run=<utc-ts>.parquet
r2://kenya-monitor-2027/posts/platform=x/type=timeline/dt=YYYY-MM-DD/run=<utc-ts>.parquet
r2://kenya-monitor-2027/metrics/platform=x/dt=YYYY-MM-DD/run=<utc-ts>.parquet
```

Full plan: `~/.claude/plans/in-this-dir-i-swift-hollerith.md`.

## TODO

Implement facebook scraping too: https://github.com/kevinzg/facebook-scraper
Implement facebook ad tracking/scraping: https://apify.com/curious_coder/facebook-ads-library-scraper
