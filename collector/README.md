# kenya-monitor-2027

Scrapes social media activity around the 2027 Kenyan general election into Cloudflare R2
as partitioned Parquet, for analysis with DuckDB. Runs on the `tf1` Hetzner box.

See the full plan: `~/.claude/plans/in-this-dir-i-swift-hollerith.md`.

## TODO

- Pull a big list of accounts via purchasing them from https://accsmarket.com/
- Setup a way to visualise and track this data and how it changes over time
- Develop a media strategy to put this information out into kenyan news and politics.


## Storage model

Immutable per-run Parquet files in R2, Hive-partitioned:

```
r2://kenya-monitor-2027/
  posts/platform=x/type=search/dt=YYYY-MM-DD/run=<utc-ts>.parquet
  posts/platform=x/type=timeline/dt=YYYY-MM-DD/run=<utc-ts>.parquet
  metrics/platform=x/dt=YYYY-MM-DD/run=<utc-ts>.parquet
```

No database. Dedup + engagement-over-time are reconstructed in DuckDB at read time.

## Setup

1. `uv sync`
2. Copy `.env.example` to `.env`, fill in R2 S3 credentials.
3. Verify the R2 round-trip: `uv run monitor check`

## Commands

- `monitor check` - round-trip a probe record through R2 (auth + write/read check).
- `monitor stats` - total and recent collection volume from R2.
- `monitor targets` - list configured accounts/keywords.
- `monitor query "SELECT ... FROM {posts} ..."` - DuckDB SQL over R2 (`{posts}` expands
  to the posts glob).
- `monitor collect x --keywords` - one-off collection (Phase 1).
- `monitor run` - scheduled worker loop (Phase 1).

## Analysis example

Latest engagement state per post:

```sql
SELECT * FROM {posts}
QUALIFY row_number() OVER (
  PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
) = 1;
```
