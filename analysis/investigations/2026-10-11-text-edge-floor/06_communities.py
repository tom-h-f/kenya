"""Kept communities per arm, with their template statistics and own posts.

Needs 03_sweep.py outputs. Own posts: non-retweet text posts by the community's accounts in the window.
Writes out/community_table.parquet and out/community_posts.parquet.
"""
import sys

import numpy as np
import pandas as pd

from kma import bench, coord2, template_spam as ts
from kma.db import connect

from common import WINDOW, wdir
import rank as rk
from pathlib import Path

OUT = Path(__file__).parent / "out"
ARMS = sys.argv[1:] or ["char40.10", "char40.30", "char40.45", "char40.60"]
base = wdir(WINDOW)
con = connect()
con.execute("SET enable_progress_bar = false")
manifest = pd.read_parquet(base / "manifest.parquet")
view = coord2.posts_view(bench.pinned_source(manifest, "posts"))

frames = {}
users = set()
for a in ARMS:
    m = pd.read_parquet(OUT / f"members_{a}.parquet")
    c = pd.read_parquet(OUT / f"communities_{a}.parquet")
    l = pd.read_parquet(OUT / f"listed_{a}.parquet")
    keep = c.loc[c["kept"], "community"]
    frames[a] = (m[m["community"].isin(keep)], c, l)
    users |= set(frames[a][0]["user_id"].astype(str))
con.register("_u", pd.DataFrame({"user_id": sorted(users)}))
posts = con.sql(
    f"SELECT CAST(user_id AS VARCHAR) AS user_id, post_id, created_at, text FROM ({view}) "
    f"WHERE NOT coalesce(is_retweet, false) AND text IS NOT NULL "
    f"AND CAST(user_id AS VARCHAR) IN (SELECT user_id FROM _u)"
).df()
posts.to_parquet(OUT / "community_posts.parquet")
print("posts", len(posts), "users", len(users))

rows = []
for a, (m, c, l) in frames.items():
    arm_nets = None
    for cid, g in m.groupby("community"):
        ids = set(g["user_id"].astype(str))
        own = posts[posts["user_id"].isin(ids)]
        st = ts.template_stats(own)
        row = c[c["community"] == cid].iloc[0]
        rows.append({"arm": a, "community": int(cid), "size": len(ids),
                     "listed": int((l["community"] == cid).sum()),
                     "share_text": float(row.get("share_text_similarity", 0.0) or 0.0),
                     "kenya_model": float(row["kenya_model"]) if pd.notna(row["kenya_model"]) else None,
                     **st.as_dict(), "label": ts.classify(st)})
tab = pd.DataFrame(rows)
tab.to_parquet(OUT / "community_table.parquet")
pd.set_option("display.width", 250)
print(tab.sort_values(["arm", "size"], ascending=[True, False]).to_string(index=False))
