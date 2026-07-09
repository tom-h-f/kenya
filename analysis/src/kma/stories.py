"""Story layer: surface discrete claims, triage them against trusted media,
and trace origin + spread (Phase 4).

Where the coordination layer asks *which accounts act together*, this layer asks
*which claims are circulating right now, and are they corroborated by trusted
outlets?* A claim-level "story" is a near-duplicate content cluster over a recent
window; it is scored on how much it looks amplified/coordinated AND how little the
trusted Kenyan outlets (already collected + embedded in the same 768d space) echo
it. A viral claim with no semantically-similar trusted-outlet post is a strong
*"unverified / likely-fabricated"* triage signal.

    from kma.db import connect
    from kma import stories as st
    con = connect()
    s = st.candidate_stories(con, days=7)
    corrob = st.corroboration(con, s)
    cards = st.story_scorecard(con, s, corrob)
    st.persist_stories(con, cards)          # collector handoff

Reuses the coordination primitives (content-cluster components, c-TF-IDF naming,
burstiness) and the authenticity score; it adds only the trusted-media
corroboration signal on top.

Triage, never a verdict: absence of trusted coverage is NOT proof of falsity -
outlets lag and often tweet only headlines. Always read the nearest trusted post
(surfaced by `corroboration`) before judging a story. Capture is a sample, so the
earliest *collected* post is not necessarily patient-zero and spread recall is
bounded.
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from kma.coordination import _burstiness_days, _ctfidf_top_terms
from kma.db import (
    BUCKET,
    embeddings_source,
    engagements_source,
    posts_source,
)
from kma.semantic import MODEL, _slug

# Handles whose posts count as trusted corroboration. The five media outlets are
# already collected as timeline accounts (targets.yaml); the two fact-checkers are
# the strongest truth signal (they publish debunks directly) but are added to the
# collector later, so their corroboration strengthens only as their timelines
# backfill. Matched case-insensitively against posts.author_handle.
TRUSTED_SOURCES = [
    "StandardKenya",
    "citizentvkenya",
    "NationAfrica",
    "KTNNewsKE",
    "ntvkenya",
    "PesaCheck",
    "AfricaCheck",
]

STORY_CAVEAT = (
    "A corroboration gap is a triage flag, NOT proof of falsity. Trusted outlets "
    "lag breaking news and often tweet only headlines, so a real story can show a "
    "gap for hours. Always read the nearest trusted post before judging - and note "
    "that fact-checker coverage (PesaCheck / AfricaCheck) is weak until their "
    "timelines backfill."
)

SAMPLING_CAVEAT = (
    "Capture is a sample, not a census: the earliest collected post is not "
    "necessarily patient-zero, and spread (retweeters / repliers) is bounded by "
    "what the snowball census reached."
)

# Story-level defaults. tau is lower than coordination's 0.9 verbatim-copypasta
# threshold: a story groups paraphrases of one claim, not identical text. Re-sweep
# in the notebook as data grows.
DEFAULT_TAU = 0.80
DEFAULT_DAYS = 7
DEFAULT_MIN_SIZE = 4

# Transparent triage weights over percentile-ranked components (mirror
# coordination.INAUTHENTICITY_WEIGHTS). The corroboration gap carries the most
# weight - it is the signal this layer adds - but a gap alone never flags a story;
# amplification + coordination must stack with it. Not a verdict.
STORY_WEIGHTS = {
    "corroboration_gap": 0.30,
    "amplifier_botness": 0.25,
    "coordination_overlap": 0.20,
    "burst_recency": 0.15,
    "source_concentration": 0.10,
}

METRIC_GLOSSARY = {
    "story_id": "Stable index of a near-duplicate content cluster (connected "
    "component of the cosine >= tau graph) over the recent window. One story = one "
    "claim circulating, paraphrases included.",
    "size": "Distinct authors posting the story - reach in accounts, not posts.",
    "n_posts": "Member posts in the story (>= size; reposts and paraphrases inflate "
    "this above the author count).",
    "keywords": "Distinctive terms of the story via c-TF-IDF over member text (same "
    "recipe as topic naming) - what the story is about, in a few words.",
    "hashtags": "Most frequent hashtags among member posts.",
    "representative_text": "The member post nearest the story centroid - the single "
    "post that best summarises the cluster.",
    "corrob_sim": "Max cosine of the story centroid to any trusted-outlet post "
    "within the window (0-1). High = a trusted outlet is saying something very "
    "similar; low = no mainstream echo.",
    "nearest_handle": "Trusted outlet whose post is closest to the story - read its "
    "text before trusting the gap.",
    "corroboration_gap": "1 - corrob_sim. High = the corroboration gap: the claim is "
    "spreading with no close trusted-media match. A triage flag, never a verdict "
    "(outlets lag; see STORY_CAVEAT).",
    "amplifier_botness": "Mean Phase-1 authenticity suspicion of the story's authors "
    "(0-1). High = the accounts pushing it look bot-like.",
    "coordination_overlap": "Share of story authors that sit in a persisted "
    "Phase-3 coordination cluster. High = a known coordinated group is pushing it.",
    "burst_recency": "Concentration of member posts in time (tightest window holding "
    "half of them, inverted). High = a sharp burst rather than a slow simmer.",
    "source_concentration": "Posts per distinct author. High = a handful of accounts "
    "generating the volume (manufactured), rather than broad organic spread.",
    "story_suspicion_index": "Transparent 0-1 triage score = weighted sum of the five "
    "percentile-ranked components (see STORY_WEIGHTS). NOT a verdict - the component "
    "breakdown, plus the nearest trusted post, is what a human acts on.",
}


def glossary_md() -> str:
    """Markdown rendering of METRIC_GLOSSARY + the caveats, for notebook display."""
    mt = "\n".join(f"- **`{k}`** - {v}" for k, v in METRIC_GLOSSARY.items())
    return (
        f"## Metric glossary\n\n{mt}\n\n"
        f"## Read before triaging\n\n- {STORY_CAVEAT}\n- {SAMPLING_CAVEAT}"
    )


def _latest_posts_cte(platform: str) -> str:
    return f"""
        SELECT * FROM {posts_source(platform)}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
        ) = 1
    """


def _latest_embeddings_cte(platform: str, model: str) -> str:
    return f"""
        SELECT platform_post_id, embedding FROM {embeddings_source(platform, _slug(model))}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY embedded_at DESC
        ) = 1
    """


def _renorm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def candidate_stories(
    con: duckdb.DuckDBPyConnection,
    days: int = DEFAULT_DAYS,
    tau: float = DEFAULT_TAU,
    min_size: int = DEFAULT_MIN_SIZE,
    platform: str = "x",
    model: str = MODEL,
) -> pd.DataFrame:
    """Claim-level stories: connected components of the cosine >= tau graph over
    recent post embeddings (same primitive as coordination.content_clusters, but
    joined to latest_posts and filtered to the last `days`, at a lower story-level
    tau). Keeps components with >= min_size distinct authors.

    Returns one row per member post: story_id, author_id, author_handle, text,
    created_at, is_repost, hashtags, conversation_id, embedding. story_id is a
    contiguous 0..k index."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import radius_neighbors_graph

    cols = [
        "platform_post_id", "story_id", "author_id", "author_handle", "text",
        "created_at", "is_repost", "hashtags", "conversation_id", "embedding",
    ]
    df = con.sql(
        f"""
        WITH e AS ({_latest_embeddings_cte(platform, model)}),
             lp AS ({_latest_posts_cte(platform)})
        SELECT lp.platform_post_id, lp.author_id, lp.author_handle, lp.text,
               lp.created_at, lp.is_repost, lp.hashtags, lp.conversation_id,
               e.embedding
        FROM e JOIN lp USING (platform_post_id)
        WHERE lp.created_at > now() - INTERVAL '{int(days)}' DAY
        """
    ).df()
    if df.empty:
        return pd.DataFrame(columns=cols)
    x = np.asarray(df["embedding"].tolist(), dtype="float32")
    # embeddings are L2-normalised: cosine >= tau <=> euclidean <= sqrt(2 - 2 tau)
    g = radius_neighbors_graph(x, radius=float(np.sqrt(2 - 2 * tau)), mode="connectivity")
    _, labels = connected_components(g, directed=False)
    df["_comp"] = labels
    # keep components with >= min_size distinct authors
    author_counts = df.groupby("_comp")["author_id"].nunique()
    keep = author_counts[author_counts >= min_size].index
    df = df[df["_comp"].isin(keep)].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    # relabel surviving components to a contiguous 0..k, largest first
    order = df["_comp"].value_counts().index.tolist()
    remap = {c: i for i, c in enumerate(order)}
    df["story_id"] = df["_comp"].map(remap)
    return df[cols].reset_index(drop=True)


def story_centroids(stories: pd.DataFrame) -> dict[int, np.ndarray]:
    """Per-story L2-renormalised mean member embedding."""
    out: dict[int, np.ndarray] = {}
    for sid, grp in stories.groupby("story_id"):
        v = np.asarray(grp["embedding"].tolist(), dtype="float32").mean(axis=0)
        out[int(sid)] = _renorm(v)
    return out


def _trusted_posts(
    con: duckdb.DuckDBPyConnection, days: int, platform: str, model: str
) -> pd.DataFrame:
    """Embeddings + text of trusted-source posts within the window."""
    handles = ", ".join(f"'{h.lower()}'" for h in TRUSTED_SOURCES)
    return con.sql(
        f"""
        WITH e AS ({_latest_embeddings_cte(platform, model)}),
             lp AS ({_latest_posts_cte(platform)})
        SELECT lp.platform_post_id, lp.author_handle, lp.text, lp.created_at,
               e.embedding
        FROM e JOIN lp USING (platform_post_id)
        WHERE lower(lp.author_handle) IN ({handles})
          AND lp.created_at > now() - INTERVAL '{int(days)}' DAY
        """
    ).df()


def corroboration(
    con: duckdb.DuckDBPyConnection,
    stories: pd.DataFrame,
    days: int = DEFAULT_DAYS,
    platform: str = "x",
    model: str = MODEL,
) -> pd.DataFrame:
    """Per story: max cosine of its centroid to any trusted-source post in the
    window, plus that nearest trusted post (for the human to judge the gap).

    Returns story_id, corrob_sim, nearest_handle, nearest_text, nearest_post_id.
    When a story's own members include a trusted handle, corrob_sim is ~1 by
    construction (that story is corroborated). No trusted posts in range ->
    corrob_sim 0.0 and null nearest fields (a maximal gap)."""
    cols = ["story_id", "corrob_sim", "nearest_handle", "nearest_text", "nearest_post_id"]
    if stories.empty:
        return pd.DataFrame(columns=cols)
    centroids = story_centroids(stories)
    trusted = _trusted_posts(con, days, platform, model)
    rows = []
    if trusted.empty:
        for sid in centroids:
            rows.append({"story_id": sid, "corrob_sim": 0.0, "nearest_handle": None,
                         "nearest_text": None, "nearest_post_id": None})
        return pd.DataFrame(rows, columns=cols)
    tv = np.asarray(trusted["embedding"].tolist(), dtype="float32")
    for sid, c in centroids.items():
        sims = tv @ c
        j = int(np.argmax(sims))
        rows.append(
            {
                "story_id": sid,
                "corrob_sim": float(sims[j]),
                "nearest_handle": trusted["author_handle"].iloc[j],
                "nearest_text": trusted["text"].iloc[j],
                "nearest_post_id": trusted["platform_post_id"].iloc[j],
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _coordination_author_ids(con: duckdb.DuckDBPyConnection, platform: str) -> set[str]:
    """author_ids in the latest persisted coordination clusters (empty if none)."""
    from kma.db import latest_coordination_clusters

    try:
        df = latest_coordination_clusters(con, platform).df()
    except duckdb.Error:
        return set()
    return set(df["author_id"].tolist()) if "author_id" in df.columns else set()


def _story_keywords(stories: pd.DataFrame) -> dict[int, list[str]]:
    sids = sorted(stories["story_id"].unique())
    docs = [
        " ".join(stories.loc[stories["story_id"] == s, "text"].dropna().tolist())
        for s in sids
    ]
    terms = _ctfidf_top_terms(docs)
    return {int(s): t for s, t in zip(sids, terms)}


def _story_hashtags(stories: pd.DataFrame, top: int = 5) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for sid, grp in stories.groupby("story_id"):
        counts: dict[str, int] = {}
        for tags in grp["hashtags"]:
            for t in tags or []:
                t = f"#{str(t).lstrip('#').lower()}"
                counts[t] = counts.get(t, 0) + 1
        out[int(sid)] = [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]
    return out


def _representative_post(grp: pd.DataFrame, centroid: np.ndarray) -> tuple[str, str]:
    v = np.asarray(grp["embedding"].tolist(), dtype="float32")
    j = int(np.argmax(v @ centroid))
    return grp["platform_post_id"].iloc[j], grp["text"].iloc[j]


def story_scorecard(
    con: duckdb.DuckDBPyConnection,
    stories: pd.DataFrame,
    corrob: pd.DataFrame | None = None,
    platform: str = "x",
    model: str = MODEL,
    days: int = DEFAULT_DAYS,
) -> pd.DataFrame:
    """Per-story scorecard ranked by a transparent story_suspicion_index (weighted
    sum of percentile-ranked components; see STORY_WEIGHTS + METRIC_GLOSSARY).

    Stacks the corroboration gap (this layer's signal) with amplifier botness,
    coordination overlap, burstiness and source concentration. Not an auto-label:
    triage for human review, always read alongside the nearest trusted post."""
    from kma.authenticity import authenticity_score

    cols = ["story_id", "size", "n_posts", "keywords", "hashtags", "representative_text",
            "representative_post_id"]
    if stories.empty:
        return pd.DataFrame(columns=cols + ["story_suspicion_index"])
    if corrob is None:
        corrob = corroboration(con, stories, days=days, platform=platform, model=model)

    auth = authenticity_score(con, platform=platform).set_index("platform_user_id")
    coord_ids = _coordination_author_ids(con, platform)
    centroids = story_centroids(stories)
    keywords = _story_keywords(stories)
    hashtags = _story_hashtags(stories)

    rows = []
    for sid, grp in stories.groupby("story_id"):
        sid = int(sid)
        author_ids = grp["author_id"].unique().tolist()
        susp = auth["suspicion"].reindex(author_ids).dropna()
        rep_id, rep_text = _representative_post(grp, centroids[sid])
        rows.append(
            {
                "story_id": sid,
                "size": len(author_ids),
                "n_posts": len(grp),
                "keywords": keywords.get(sid, []),
                "hashtags": hashtags.get(sid, []),
                "representative_text": rep_text,
                "representative_post_id": rep_id,
                "member_post_ids": grp["platform_post_id"].tolist(),
                "amplifier_botness": float(susp.mean()) if len(susp) else float("nan"),
                "coordination_overlap": (
                    np.mean([a in coord_ids for a in author_ids]) if author_ids else 0.0
                ),
                "burst_days": _burstiness_days(grp["created_at"]),
                "source_concentration": len(grp) / max(len(author_ids), 1),
            }
        )
    df = pd.DataFrame(rows)
    df = df.merge(corrob, on="story_id", how="left")
    df["corrob_sim"] = df["corrob_sim"].fillna(0.0)
    df["corroboration_gap"] = 1.0 - df["corrob_sim"]

    def rank(s: pd.Series, invert: bool = False) -> pd.Series:
        r = s.rank(pct=True)
        return (1 - r) if invert else r

    components = {
        "corroboration_gap": rank(df["corroboration_gap"]),
        "amplifier_botness": rank(df["amplifier_botness"]),
        "coordination_overlap": rank(df["coordination_overlap"]),
        # tighter burst window (fewer days) => burstier => higher rank
        "burst_recency": rank(df["burst_days"], invert=True),
        "source_concentration": rank(df["source_concentration"]),
    }
    for name, comp in components.items():
        df[f"ix_{name}"] = comp.fillna(0.0)
    df["story_suspicion_index"] = sum(
        w * df[f"ix_{k}"] for k, w in STORY_WEIGHTS.items()
    )
    return df.sort_values("story_suspicion_index", ascending=False, ignore_index=True)


def origin(
    con: duckdb.DuckDBPyConnection,
    story: pd.DataFrame,
    platform: str = "x",
    top: int = 15,
) -> pd.DataFrame:
    """Earliest-seen member posts of one story with author authenticity and
    coordination-cluster membership - a first-mover view.

    `story` is the candidate_stories rows for a single story_id. Caveat: bounded
    by capture; the earliest *collected* post is not necessarily patient-zero."""
    from kma.authenticity import authenticity_score

    cols = ["created_at", "author_handle", "author_id", "text", "is_repost",
            "suspicion", "in_coordination_cluster"]
    if story.empty:
        return pd.DataFrame(columns=cols)
    auth = authenticity_score(con, platform=platform).set_index("platform_user_id")
    coord_ids = _coordination_author_ids(con, platform)
    out = story.sort_values("created_at").head(top).copy()
    out["suspicion"] = auth["suspicion"].reindex(out["author_id"]).to_numpy()
    out["in_coordination_cluster"] = out["author_id"].isin(coord_ids)
    return out[cols].reset_index(drop=True)


def spread(
    con: duckdb.DuckDBPyConnection,
    story: pd.DataFrame,
    platform: str = "x",
) -> dict[str, pd.DataFrame]:
    """Diffusion of one story: amplifier accounts (retweeters of member posts +
    repliers in member conversations) and a post-volume timeline.

    Returns {"amplifiers": df(handle/id, kind, n), "timeline": df(hour, n_posts)}.
    Retweeters come from the snowball engagement census; repliers from posts whose
    conversation_id matches a member post. Recall is bounded by capture."""
    empty = {
        "amplifiers": pd.DataFrame(columns=["platform_user_id", "kind", "n"]),
        "timeline": pd.DataFrame(columns=["hour", "n_posts"]),
    }
    if story.empty:
        return empty
    post_ids = story["platform_post_id"].tolist()
    convo_ids = [c for c in story["conversation_id"].dropna().unique().tolist()]
    con.register("_story_posts", pd.DataFrame({"platform_post_id": post_ids}))
    con.register("_story_convos", pd.DataFrame({"conversation_id": convo_ids or [None]}))
    try:
        try:
            retweeters = con.sql(
                f"""
                SELECT platform_user_id, 'retweet' AS kind, count(*) AS n
                FROM (
                    SELECT * FROM {engagements_source(platform)}
                    QUALIFY row_number() OVER (
                        PARTITION BY platform, platform_post_id, platform_user_id, kind
                        ORDER BY collected_at DESC
                    ) = 1
                )
                WHERE kind = 'retweet'
                  AND platform_post_id IN (SELECT platform_post_id FROM _story_posts)
                GROUP BY 1
                """
            ).df()
        except duckdb.Error:
            retweeters = pd.DataFrame(columns=["platform_user_id", "kind", "n"])
        repliers = con.sql(
            f"""
            WITH lp AS ({_latest_posts_cte(platform)})
            SELECT author_id AS platform_user_id, 'reply' AS kind, count(*) AS n
            FROM lp
            WHERE in_reply_to_id IS NOT NULL
              AND conversation_id IN (SELECT conversation_id FROM _story_convos)
              AND platform_post_id NOT IN (SELECT platform_post_id FROM _story_posts)
            GROUP BY 1
            """
        ).df()
    finally:
        con.unregister("_story_posts")
        con.unregister("_story_convos")
    amplifiers = pd.concat([retweeters, repliers], ignore_index=True).sort_values(
        "n", ascending=False, ignore_index=True
    )
    timeline = (
        story.assign(hour=story["created_at"].dt.floor("h"))
        .groupby("hour")
        .size()
        .reset_index(name="n_posts")
    )
    return {"amplifiers": amplifiers, "timeline": timeline}


def persist_stories(
    con: duckdb.DuckDBPyConnection,
    scorecard: pd.DataFrame,
    platform: str = "x",
    min_index: float = 0.0,
) -> str | None:
    """Write scored stories as a Parquet run under the stories/ prefix (mirror
    coordination.persist_clusters). This is the collector handoff: the collector
    reads the latest run and promotes flagged stories' keywords/hashtags as
    targeted-collection terms. `min_index` drops low-suspicion stories.

    Returns the R2 key, or None when nothing clears the cutoff."""
    if scorecard.empty:
        return None
    keep = scorecard[scorecard["story_suspicion_index"] >= min_index]
    if keep.empty:
        return None
    now = datetime.now(timezone.utc)
    cols = [
        "story_id", "size", "n_posts", "keywords", "hashtags", "representative_text",
        "representative_post_id", "member_post_ids", "corrob_sim", "corroboration_gap",
        "amplifier_botness", "coordination_overlap", "source_concentration",
        "story_suspicion_index",
    ]
    buf = keep[[c for c in cols if c in keep.columns]].copy()
    buf["computed_at"] = now
    key = (
        f"stories/platform={platform}"
        f"/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    )
    con.register("_story_buf", buf)
    try:
        con.execute(
            f"COPY _story_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_story_buf")
    return key
