"""Kenya relevance model on Tanzanian politics: persisted scores of Tanzanian posts in the 21d window.

Read-only on R2. Writes out/tz_scan.parquet (untracked) and prints the summary.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

from kma import bench, coord2, measure, relevance
from kma.db import connect

from common import WINDOW, wdir

BUNDLE = Path(sys.argv[1])
TZ = (r"tanzaniamassacre|tundu\s*lissu|\blissu\b|samia\s*suluhu|samiamuuaji|chadema|\bmo29\b|"
      r"updates_kesi_ya_uhaini|katibampya|freetundulissu|magufuli|\bdodoma\b|oktoba\s*29|bunge\s+la\s+tanzania")
TZ_RE = re.compile(TZ, re.I)

con = connect()
con.execute("SET enable_progress_bar = false")
manifest = pd.read_parquet(wdir(WINDOW) / "manifest.parquet")
view = coord2.posts_view(bench.pinned_source(manifest, "posts"))
hits = con.sql(
    f"SELECT post_id, user_id, created_at, is_retweet, text FROM ({view}) "
    f"WHERE text IS NOT NULL AND regexp_matches(text, '(?i){TZ}')"
).df()
hits["user_id"] = hits["user_id"].astype(str)
hits["kenya_anchor"] = hits["text"].map(lambda t: bool(measure._KENYA_RE.search(t)))
hits["p"] = hits["post_id"].astype(str).map(relevance.scores(con, hits["post_id"]))
hits["is_tz_member"] = False

g = json.loads((BUNDLE / "g8.json").read_text())
members = set(map(str, g["nodes"]["id"]))
mem = con.sql(
    f"SELECT post_id, user_id, created_at, is_retweet, text FROM ({view}) WHERE text IS NOT NULL"
).df()
mem["user_id"] = mem["user_id"].astype(str)
mem = mem[mem["user_id"].isin(members)].copy()
mem["p"] = mem["post_id"].astype(str).map(relevance.scores(con, mem["post_id"]))
mem["tz_text"] = mem["text"].map(lambda t: bool(TZ_RE.search(t)))
mem["kenya_anchor"] = mem["text"].map(lambda t: bool(measure._KENYA_RE.search(t)))
hits.to_parquet(wdir(WINDOW).parent.parent / "out" / "tz_hits.parquet") if False else None
out = Path(__file__).parent / "out"
hits.to_parquet(out / "tz_hits.parquet")
mem.to_parquet(out / "c8_member_posts.parquet")


def rate(df: pd.DataFrame, label: str) -> dict:
    s = df.dropna(subset=["p"])
    return {"set": label, "posts": len(df), "scored": len(s),
            "p>=0.5": int((s["p"] >= 0.5).sum()),
            "share>=0.5": round(float((s["p"] >= 0.5).mean()), 3) if len(s) else None,
            "mean_p": round(float(s["p"].mean()), 3) if len(s) else None}


rows = [
    rate(hits, "tz keyword posts, window"),
    rate(hits[~hits["kenya_anchor"]], "tz keyword, no Kenya anchor"),
    rate(hits[~hits["kenya_anchor"] & ~hits["is_retweet"].astype(bool)], "  ... originals/replies only"),
    rate(mem, "c8 member posts, all"),
    rate(mem[mem["tz_text"] & ~mem["kenya_anchor"]], "c8 member posts, tz kw no anchor"),
    rate(mem[~mem["tz_text"] & ~mem["kenya_anchor"]], "c8 member posts, no tz kw no anchor"),
]
print(pd.DataFrame(rows).to_string(index=False))
