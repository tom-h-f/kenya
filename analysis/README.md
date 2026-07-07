# analysis

DuckDB/Polars workspace for querying the Kenya-2027 dataset stored in Cloudflare R2.
Reads R2 credentials from the shared `../.env`.

## Setup

```bash
cd analysis
uv sync
```

This creates `analysis/.venv` with an installed, editable `kma` package.

### DataSpell / PyCharm

Open the `analysis/` directory as the project. Point the interpreter at
`analysis/.venv/bin/python` (uv's venv). `kma` is importable everywhere, and Jupyter
notebooks run against this kernel (`ipykernel` is included).

### marimo (reactive notebooks)

```bash
uv run marimo edit notebooks/explore.py
uv run marimo edit notebooks/coordination.py
```

## Usage

```python
from kma.db import connect, posts, latest_posts

con = connect()

# every collected snapshot (good for engagement-over-time)
posts(con, platform="x").pl()

# one row per post, latest state (deduped)
latest_posts(con, platform="x").limit(20).pl()

# drop to raw SQL any time
from kma.db import posts_source
con.sql(f"SELECT author_handle, count(*) FROM {posts_source('x')} GROUP BY 1 ORDER BY 2 DESC")
```

`.pl()` -> polars, `.df()` -> pandas, `.arrow()` -> Arrow.

## Two ways to connect

- **Direct to R2** (`connect()`): your local DuckDB reads Parquet straight from R2. Needs
  the R2 credentials locally; each query pulls data over the network to you.
- **Via the tf1 quack server** (`connect_quack()`): queries run *on tf1* against R2 and
  only results stream back. Needs just `QUACK_TOKEN` + `QUACK_HOST` (no R2 creds), works
  over the tailnet, and the server pre-exposes the `posts` / `latest_posts` / `metrics`
  views.

```python
from kma import connect_quack

con = connect_quack()                       # attaches tf1 as `kenya`
con.sql("FROM kenya.query('SELECT count(*) FROM latest_posts')").pl()
con.sql(\"\"\"FROM kenya.query('
    SELECT author_handle, count(*) n FROM latest_posts GROUP BY 1 ORDER BY n DESC LIMIT 10
')\"\"\").pl()
```

The tf1 server is defined in `../server/` (DuckDB + quack in Docker, bound to tf1's
tailscale IP). Requires `pytz` client-side (already a dependency).
