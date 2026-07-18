"""Shared helpers for the 2026-07-17 manipulation-sweep investigation scripts.

Every script in this directory:
- runs standalone: `uv run python investigations/2026-07-17-manipulation-sweep/NN_x.py`
- defaults to a sub-minute --sample run; pass --full for the real sweep
- writes artifacts to out/ and prints a top-N table plus the standard caveat block
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from kma import db
from kma.coordination import SAMPLING_CAVEAT as COORDINATION_CAVEAT
from kma.stories import SAMPLING_CAVEAT, STORY_CAVEAT

OUT = Path(__file__).resolve().parent / "out"

EXTRA_CAVEATS = (
    "Suspicion scores are triage signals, not bot labels. Coordination and "
    "cohort structure are probabilistic evidence of similarity, not proof of "
    "malice or inauthenticity. Absence of corroboration is not evidence a "
    "claim is false."
)


def parse_args(description: str, default_sample: int = 2000) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--sample",
        type=int,
        default=default_sample,
        help="row cap for the validation run (default %(default)s)",
    )
    p.add_argument(
        "--full", action="store_true", help="run against the full corpus"
    )
    args = p.parse_args()
    if args.full:
        args.sample = None
    return args


def connect() -> duckdb.DuckDBPyConnection:
    """kma.db.connect with timeouts sized for the 300MB consolidated
    embeddings run file (default 30s HTTP timeout flakes on it)."""
    con = db.connect()
    con.execute("SET http_timeout=300000; SET http_retries=5;")
    return con


def to_days(ts: pd.Series) -> pd.Series:
    """Epoch days from any datetime series regardless of unit/tz (DuckDB emits
    datetime64[us], which breaks int64-nanosecond arithmetic)."""
    return (
        pd.to_datetime(ts, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")
    ).dt.days


def coordination_clusters() -> pd.DataFrame:
    """Cluster membership cached by 00_coordination_refresh.py (R2 has no
    persisted coordination runs; this sweep avoids persist side effects)."""
    path = OUT / "00_coordination_clusters.parquet"
    if not path.exists():
        print("NOTE: no coordination cache - run 00_coordination_refresh.py first")
        return pd.DataFrame(columns=["author_id", "cluster_id", "size", "channels"])
    return pd.read_parquet(path)


def save(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    if name.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"wrote {path.relative_to(OUT.parent)} ({len(df)} rows)")
    return path


def show(df: pd.DataFrame, title: str, n: int = 10) -> None:
    print(f"\n== {title} (top {min(n, len(df))} of {len(df)}) ==")
    if df.empty:
        print("(empty)")
        return
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df.head(n).to_string(index=False))


def print_caveats() -> None:
    print("\n-- caveats --")
    for c in (SAMPLING_CAVEAT, STORY_CAVEAT, COORDINATION_CAVEAT, EXTRA_CAVEATS):
        print(f"* {c}")
