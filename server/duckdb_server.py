"""DuckDB quack server for tf1.

Loads the R2 secret, exposes convenience views over the R2 Parquet dataset, and serves
the quack protocol (plain HTTP, bound inside the container to 0.0.0.0:9494 - Docker
publishes it only on tf1's tailscale IP, so it is reachable solely over the tailnet).

Clients attach with just the token (no R2 credentials needed client-side):

    INSTALL quack; LOAD quack;
    ATTACH 'quack:tf1.meerkat-decibel.ts.net:9494' AS kenya (TOKEN '...', DISABLE_SSL true);
    FROM kenya.query('SELECT count(*) FROM latest_posts');
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

BUCKET = os.getenv("R2_BUCKET", "kenya-monitor-2027")
TOKEN = os.environ["QUACK_TOKEN"]
LISTEN = os.getenv("QUACK_LISTEN", "quack:0.0.0.0:9494")


def _posts(platform: str = "*", type: str = "*") -> str:
    glob = f"r2://{BUCKET}/posts/platform={platform}/type={type}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def _metrics(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/metrics/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def _authors(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/authors/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


con = duckdb.connect()
con.execute("INSTALL quack; LOAD quack; INSTALL httpfs; LOAD httpfs;")
con.execute(
    "CREATE OR REPLACE SECRET r2 (TYPE r2, KEY_ID ?, SECRET ?, ACCOUNT_ID ?)",
    [os.environ["R2_ACCESS_KEY_ID"], os.environ["R2_SECRET_ACCESS_KEY"], os.environ["R2_ACCOUNT_ID"]],
)

def make_view(name: str, select: str) -> None:
    """Create a view, skipping it if its R2 glob currently has no files."""
    try:
        con.execute(f"CREATE OR REPLACE VIEW {name} AS {select}")
        print(f"view ready: {name}", flush=True)
    except duckdb.IOException:
        print(f"view skipped (no files yet): {name}", flush=True)


make_view("posts", f"SELECT * FROM {_posts()}")
make_view(
    "latest_posts",
    f"""SELECT * FROM {_posts()}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
        ) = 1""",
)
make_view("metrics", f"SELECT * FROM {_metrics()}")
make_view("authors", f"SELECT * FROM {_authors()}")
make_view(
    "latest_authors",
    f"""SELECT * FROM {_authors()}
        QUALIFY row_number() OVER (
            PARTITION BY platform_user_id ORDER BY collected_at DESC
        ) = 1""",
)

info = con.execute(
    f"CALL quack_serve('{LISTEN}', token := ?, allow_other_hostname := true, disable_ssl := true)",
    [TOKEN],
).fetchall()
print(f"quack serving: {info}", flush=True)

threading.Event().wait()  # serve runs in a background thread; keep the process alive
