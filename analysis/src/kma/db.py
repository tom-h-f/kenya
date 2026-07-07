"""Connect DuckDB to the R2 election dataset and expose convenience relations.

    from kma.db import connect, latest_posts
    con = connect()
    df = latest_posts(con).pl()      # polars DataFrame
    pdf = latest_posts(con).df()     # pandas DataFrame
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

MONOREPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(MONOREPO_ROOT / ".env")

BUCKET = os.getenv("R2_BUCKET", "kenya-monitor-2027")


def connect() -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with httpfs loaded and an R2 secret configured."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(
        "CREATE OR REPLACE SECRET r2 (TYPE r2, KEY_ID ?, SECRET ?, ACCOUNT_ID ?)",
        [
            os.environ["R2_ACCESS_KEY_ID"],
            os.environ["R2_SECRET_ACCESS_KEY"],
            os.environ["R2_ACCOUNT_ID"],
        ],
    )
    return con


def posts_source(platform: str = "*", type: str = "*") -> str:
    """A read_parquet(...) expression usable directly in SQL FROM clauses."""
    glob = f"r2://{BUCKET}/posts/platform={platform}/type={type}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def posts(con: duckdb.DuckDBPyConnection, platform: str = "*", type: str = "*"):
    """All collected post rows (every engagement snapshot, not deduped)."""
    return con.sql(f"SELECT * FROM {posts_source(platform, type)}")


def latest_posts(con: duckdb.DuckDBPyConnection, platform: str = "*", type: str = "*"):
    """One row per post: its most recently collected state."""
    return con.sql(
        f"""
        SELECT * FROM {posts_source(platform, type)}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
        ) = 1
        """
    )


def metrics_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/metrics/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def authors_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/authors/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_authors(con: duckdb.DuckDBPyConnection, platform: str = "*"):
    """One row per author: their most recently collected profile snapshot."""
    return con.sql(
        f"""
        SELECT * FROM {authors_source(platform)}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_user_id ORDER BY collected_at DESC
        ) = 1
        """
    )


def embeddings_source(platform: str = "*", model: str = "*") -> str:
    glob = f"r2://{BUCKET}/embeddings/platform={platform}/model={model}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_embeddings(con: duckdb.DuckDBPyConnection, platform: str = "*", model: str = "*"):
    """One embedding row per post (latest), for a given model."""
    return con.sql(
        f"""
        SELECT * FROM {embeddings_source(platform, model)}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id, model ORDER BY embedded_at DESC
        ) = 1
        """
    )


def labels_source(platform: str = "*") -> str:
    glob = f"r2://{BUCKET}/labels/platform={platform}/dt=*/run=*.parquet"
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_labels(con: duckdb.DuckDBPyConnection, platform: str = "*"):
    """One sentiment/emotion label row per post (latest)."""
    return con.sql(
        f"""
        SELECT * FROM {labels_source(platform)}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY labeled_at DESC
        ) = 1
        """
    )


def coordination_source(
    kind: str = "edges",
    platform: str = "*",
    channel: str = "*",
    method: str = "*",
) -> str:
    """A read_parquet(...) expression for persisted coordination artifacts."""
    if kind == "edges":
        glob = (
            f"r2://{BUCKET}/coordination/platform={platform}/kind=edges"
            f"/channel={channel}/method={method}/dt=*/run=*.parquet"
        )
    elif kind == "clusters":
        glob = f"r2://{BUCKET}/coordination/platform={platform}/kind=clusters/dt=*/run=*.parquet"
    else:
        raise ValueError(f"unknown coordination kind {kind!r}")
    return f"read_parquet('{glob}', union_by_name=true, hive_partitioning=true)"


def latest_coordination_edges(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    channel: str = "*",
    method: str = "*",
):
    """Latest validated edge row per (src, dst, channel, method) run."""
    return con.sql(
        f"""
        SELECT * FROM {coordination_source('edges', platform, channel, method)}
        QUALIFY row_number() OVER (
            PARTITION BY src, dst, channel, method ORDER BY computed_at DESC
        ) = 1
        """
    )


def latest_coordination_clusters(con: duckdb.DuckDBPyConnection, platform: str = "x"):
    """Latest cluster membership row per (cluster_id, author_id)."""
    return con.sql(
        f"""
        SELECT * FROM {coordination_source('clusters', platform)}
        QUALIFY row_number() OVER (
            PARTITION BY cluster_id, author_id ORDER BY computed_at DESC
        ) = 1
        """
    )


def connect_quack(name: str = "kenya") -> duckdb.DuckDBPyConnection:
    """Attach the tf1 DuckDB quack server. Queries run on tf1 against R2; no R2 creds
    are needed locally - only QUACK_HOST + QUACK_TOKEN (from the shared .env).

        con = connect_quack()
        con.sql("FROM kenya.query('SELECT count(*) FROM latest_posts')")

    The server exposes the views: posts, latest_posts, metrics.
    """
    host = os.environ["QUACK_HOST"]
    token = os.environ["QUACK_TOKEN"]
    con = duckdb.connect()
    con.execute("INSTALL quack; LOAD quack;")
    con.execute(f"ATTACH 'quack:{host}' AS {name} (TOKEN '{token}', DISABLE_SSL true)")
    return con
