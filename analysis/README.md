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
uv run marimo edit notebooks/desk_brief.py
uv run marimo edit notebooks/coordination.py
```

`desk_brief.py` is the investigator-facing composition (claims, corroboration,
amplifiers, framing, claim-scoped coordination, region/community aggregates).
See `../docs/plans/2026-07-16-misinfo-desk-brief/`.

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

## Connecting

`connect()` gives you local DuckDB with `httpfs` loaded and an R2 secret configured;
it reads Parquet straight from R2 over the network. Needs the R2 credentials in `.env`
(no other setup).
