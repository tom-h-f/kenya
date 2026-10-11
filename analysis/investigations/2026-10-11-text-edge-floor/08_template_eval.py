"""Template detector on every kept community of two arms, over (a) own posts and (b) text-edge posts.

(b) = posts in cross-author text pairs (cosine >= 0.85, char-4gram overlap >= the arm's floor) whose two
authors are both in the community. Writes out/template_eval.csv.
"""
import pandas as pd

from kma import template_spam as ts
from common import WINDOW, wdir
from pathlib import Path

OUT = Path(__file__).parent / "out"
base = wdir(WINDOW)
pairs = pd.read_parquet(base / "pairs_manifest" / "pairs_wj.parquet")
rows = pd.read_parquet(base / "rows.parquet", columns=["post_id", "user_id", "text"]).set_index("post_id")
own = pd.read_parquet(OUT / "community_posts.parquet")
out = []
for arm, floor in (("char40.10", 0.10), ("char40.45", 0.45)):
    m = pd.read_parquet(OUT / f"members_{arm}.parquet")
    c = pd.read_parquet(OUT / f"communities_{arm}.parquet")
    l = pd.read_parquet(OUT / f"listed_{arm}.parquet")
    comm_of = dict(zip(m["user_id"].astype(str), m["community"]))
    kept = pairs[pairs["overlap"] >= floor].copy()
    kept["ca"] = kept["user_a"].map(comm_of)
    kept["cb"] = kept["user_b"].map(comm_of)
    inside = kept[(kept["ca"] == kept["cb"]) & kept["ca"].notna()]
    for cid in c.loc[c["kept"], "community"]:
        members = set(m.loc[m["community"] == cid, "user_id"].astype(str))
        sub = inside[inside["ca"] == cid]
        ids = pd.unique(pd.concat([sub["post_a"], sub["post_b"]]))
        edge_posts = rows.loc[ids].reset_index()[["post_id", "user_id", "text"]] if len(ids) else rows.iloc[:0]
        s_own = ts.template_stats(own[own["user_id"].isin(members)])
        s_edge = ts.template_stats(edge_posts) if len(edge_posts) else None
        share = c.loc[c["community"] == cid, "share_text_similarity"].iloc[0]
        out.append({"arm": arm, "community": int(cid), "size": len(members), "share_text": share,
                    "edge_posts": len(edge_posts),
                    "own_rate": s_own.template_rate, "own_promo": s_own.promo_rate, "own_label": ts.classify(s_own),
                    "edge_rate": s_edge.template_rate if s_edge else None,
                    "edge_promo": s_edge.promo_rate if s_edge else None,
                    "edge_eligible": s_edge.eligible if s_edge else 0,
                    "edge_label": ts.classify(s_edge) if s_edge else None})
df = pd.DataFrame(out)
df.to_csv(OUT / "template_eval.csv", index=False)
pd.set_option("display.width", 250)
print(df[df["edge_posts"] > 0].sort_values(["arm", "edge_rate"], ascending=[True, False]).to_string(index=False))
