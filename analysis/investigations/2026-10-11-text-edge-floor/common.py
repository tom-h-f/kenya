"""Overlap-floor sweep on daily-2026-10-10-21d. Run from analysis/, in order, outputs in untracked out/.

01_export.py, 01b_nets.py   window rows, manifest, bipartite networks (read-only on R2)
02_pairs.py                 every cross-author post pair at cosine >= 0.85, with char-4gram overlap
04_repro.py                 gate: rebuilt trace equals the persisted production trace
03_sweep.py --bundle DIR    floor sweep; DIR holds the public bundle's communities.json and g<cid>.json
05_labelled_recall.py DIR   same-message recall per floor on the 2026-09-11 labelled pairs
06_communities.py, 08_template_eval.py   template detector per community
07_read_new.py              blind headless read of the text-only communities a floor leaves (BUNDLE=DIR)
40_tz_scan.py DIR, 41_tz_label.py, 42_tz_guard.py   Kenya relevance model on Tanzanian posts
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
WINDOW = "daily-2026-10-10-21d"
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
