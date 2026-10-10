"""Shared paths and loaders for the text-trace-value investigation."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
WINDOW = "daily-2026-10-09-30d"
BIPARTITE = ("co_retweet", "co_url", "hashtag_sequence", "fast_retweet")


def wdir(window: str = WINDOW) -> Path:
    path = OUT / window
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_bipartite(window: str = WINDOW) -> dict[str, pd.DataFrame]:
    nets = {}
    for name in BIPARTITE:
        path = wdir(window) / "nets" / f"{name}.parquet"
        if path.exists():
            nets[name] = pd.read_parquet(path).astype({"source": str, "target": str})
    return nets
