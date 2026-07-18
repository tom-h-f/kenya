"""Copypasta lag forensics: who seeds near-duplicate text, who echoes it, and
how fast. Plus entity-swapped template detection.

Near-dup components come from coordination.content_clusters (cosine >= tau over
post embeddings, originals only) - the same objects as the text_sim channel,
but here each component is ORDERED by created_at, which the symmetric SVN
edges throw away. Seed = earliest captured member (capture caveat applies).

Templates: posts sharing a token skeleton (urls/mentions/hashtags/digits
masked) but with >=3 distinct raw texts across >=3 authors - the
fill-in-the-blank pattern of paste campaigns.
"""

from __future__ import annotations

import re

import pandas as pd

import _lib
from kma import db
from kma.coordination import content_clusters

MIN_ECHO_AUTHORS = 3
BURST_MINUTES = 60
SKELETON_MIN_LEN = 40


def skeleton(text: str) -> str:
    t = text.lower()
    t = re.sub(r"https?://\S+", "<u>", t)
    t = re.sub(r"@\w+", "<m>", t)
    t = re.sub(r"#\w+", "<h>", t)
    t = re.sub(r"\d+([.,:]\d+)*", "<n>", t)
    return re.sub(r"\s+", " ", t).strip()


def main() -> None:
    args = _lib.parse_args("Copypasta lag forensics", default_sample=300)
    con = db.connect()

    cc = content_clusters(con)
    print(f"{cc['cluster_id'].nunique()} near-dup components, {len(cc)} posts")
    if args.sample:
        keep = cc["cluster_id"].drop_duplicates().head(args.sample)
        cc = cc[cc["cluster_id"].isin(keep)]

    posts = con.sql(
        f"""
        SELECT p.platform_post_id, p.author_id, p.author_handle, p.created_at,
               p.text, a.followers_count
        FROM (
            SELECT * FROM {db.posts_source('x')}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY collected_at DESC
            ) = 1
        ) p
        LEFT JOIN (
            SELECT * FROM {db.authors_source('x')}
            QUALIFY row_number() OVER (
                PARTITION BY platform_user_id ORDER BY collected_at DESC
            ) = 1
        ) a ON p.author_id = a.platform_user_id
        WHERE p.created_at IS NOT NULL
        """
    ).df()
    m = cc.merge(posts, on="platform_post_id").sort_values("created_at")

    comp_rows, seed_of = [], {}
    for cid, grp in m.groupby("cluster_id"):
        authors = grp.drop_duplicates("author_id")
        if authors["author_id"].nunique() < 2:
            continue
        seed = authors.iloc[0]
        span_min = (
            grp["created_at"].max() - grp["created_at"].min()
        ).total_seconds() / 60
        lags = (
            authors["created_at"].iloc[1:] - seed["created_at"]
        ).dt.total_seconds() / 60
        seed_of[cid] = seed["author_id"]
        comp_rows.append(
            {
                "cluster_id": cid,
                "n_posts": len(grp),
                "n_authors": authors["author_id"].nunique(),
                "span_min": round(span_min, 1),
                "median_echo_lag_min": round(float(lags.median()), 1),
                "echoes_within_60min": int((lags <= BURST_MINUTES).sum()),
                "seed_handle": seed["author_handle"],
                "seed_followers": seed["followers_count"],
                "text": seed["text"][:120],
            }
        )
    comps = pd.DataFrame(comp_rows)
    bursts = comps[
        (comps["n_authors"] >= MIN_ECHO_AUTHORS)
        & (comps["span_min"] <= BURST_MINUTES)
    ].sort_values("n_authors", ascending=False)
    _lib.show(bursts, "paste bursts (>=3 authors inside 60 min)")
    _lib.save(comps, "03_components.csv")
    _lib.save(bursts, "03_paste_bursts.csv")

    clusters_cache = _lib.coordination_clusters()
    in_cluster = set(clusters_cache["author_id"])
    seeded = comps[comps["n_authors"] >= MIN_ECHO_AUTHORS]
    seed_counts = seeded.groupby("seed_handle").agg(
        seeds=("cluster_id", "size"),
        echo_authors=("n_authors", "sum"),
        seed_followers=("seed_followers", "first"),
    )
    echo_counts = (
        m[m["cluster_id"].isin(seeded["cluster_id"])]
        .loc[lambda d: d["author_id"] != d["cluster_id"].map(seed_of)]
        .groupby("author_handle")["cluster_id"]
        .nunique()
        .rename("echoed_components")
    )
    seeders = seed_counts.join(echo_counts, how="outer").fillna(0)
    seeders = seeders.astype(
        {"seeds": int, "echo_authors": int, "echoed_components": int}
    ).sort_values(["seeds", "echoed_components"], ascending=False)
    handle_ids = m.drop_duplicates("author_handle").set_index("author_handle")[
        "author_id"
    ]
    seeders["in_coord_cluster"] = (
        seeders.index.map(handle_ids).isin(in_cluster)
    )
    _lib.show(
        seeders.reset_index(), "serial seeders / echoers of multi-author copypasta"
    )
    _lib.save(seeders.reset_index(), "03_seeders.csv")

    elig = m.drop_duplicates("platform_post_id").copy()
    elig["skeleton"] = elig["text"].map(skeleton)
    elig = elig[elig["skeleton"].str.len() >= SKELETON_MIN_LEN]
    trows = []
    for skel, grp in elig.groupby("skeleton"):
        if grp["text"].nunique() < 3 or grp["author_id"].nunique() < 3:
            continue
        trows.append(
            {
                "n_texts": grp["text"].nunique(),
                "n_authors": grp["author_id"].nunique(),
                "handles": ", ".join(grp["author_handle"].unique()[:6]),
                "skeleton": skel[:110],
            }
        )
    templates = pd.DataFrame(
        trows, columns=["n_texts", "n_authors", "handles", "skeleton"]
    ).sort_values("n_authors", ascending=False)
    _lib.show(templates, "entity-swapped templates (>=3 texts, >=3 authors)")
    _lib.save(templates, "03_templates.csv")

    _lib.print_caveats()


if __name__ == "__main__":
    main()
