"""Claim-anchored framing: topic neighborhood + mood around a story.

Topics remain corpus-wide (UMAP/HDBSCAN); this module maps each story onto that
landscape and summarises sentiment over the claim window. Does not use X `lang`
for primary slices.
"""

from __future__ import annotations

import pandas as pd

from kma.coordination import _ctfidf_top_terms


def story_topics(
    stories: pd.DataFrame,
    topics_df: pd.DataFrame,
    topic_summary: pd.DataFrame | None = None,
    top_n: int = 3,
) -> pd.DataFrame:
    """Map each story to its nearest topic(s) by member-post overlap.

    `topics_df` is `assign_topics` output (needs platform_post_id, topic).
    Optional `topic_summary` adds terms/name. Topic -1 (noise) is ignored unless
    it is the only assignment.

    Returns story_id[, stable_story_id], topic_id, overlap_n, share, topic_terms.
    """
    cols = ["story_id", "topic_id", "overlap_n", "share", "topic_terms"]
    if stories.empty or topics_df.empty:
        return pd.DataFrame(columns=cols)

    members = stories[["story_id", "platform_post_id"]].drop_duplicates()
    if "stable_story_id" in stories.columns:
        stab = stories.groupby("story_id")["stable_story_id"].first()
    else:
        stab = None

    joined = members.merge(
        topics_df[["platform_post_id", "topic"]],
        on="platform_post_id",
        how="left",
    )
    joined["topic"] = joined["topic"].fillna(-1).astype(int)

    rows = []
    for sid, grp in joined.groupby("story_id"):
        n = len(grp)
        counts = grp["topic"].value_counts()
        # prefer non-noise topics; fall back to -1 if that is all we have
        ranked = [t for t in counts.index if t != -1] or list(counts.index)
        for topic_id in ranked[:top_n]:
            overlap = int(counts.get(topic_id, 0))
            terms = ""
            if topic_summary is not None and not topic_summary.empty:
                hit = topic_summary[topic_summary["topic"] == topic_id]
                if len(hit):
                    terms = hit["terms"].iloc[0] if "terms" in hit.columns else ""
                    if isinstance(terms, list):
                        terms = ", ".join(terms)
            rows.append(
                {
                    "story_id": int(sid),
                    "topic_id": int(topic_id),
                    "overlap_n": overlap,
                    "share": overlap / n if n else 0.0,
                    "topic_terms": terms,
                }
            )
    out = pd.DataFrame(rows, columns=cols)
    if stab is not None and not out.empty:
        out["stable_story_id"] = out["story_id"].map(stab)
    return out


def story_keywords(stories: pd.DataFrame, top_terms: int = 8) -> dict[int, list[str]]:
    """c-TF-IDF keywords per story from member text (claim neighborhood)."""
    if stories.empty:
        return {}
    grouped = stories.groupby("story_id")["text"]
    sids = sorted(grouped.groups)
    docs = [" ".join(grouped.get_group(s).dropna().astype(str).tolist()) for s in sids]
    terms = _ctfidf_top_terms(docs, top_terms=top_terms)
    return {int(s): t for s, t in zip(sids, terms)}


def sentiment_timeline(
    stories: pd.DataFrame,
    labels: pd.DataFrame,
    freq: str = "D",
) -> pd.DataFrame:
    """Bucket sentiment over the claim window for story member posts.

    `labels` needs platform_post_id + sentiment (+ optional labeled_at).
    Stories need platform_post_id + created_at. Empty neighborhood -> empty frame.
    """
    cols = ["story_id", "bucket", "n_posts", "mean_sentiment", "neg_share", "pos_share"]
    if stories.empty or labels.empty:
        return pd.DataFrame(columns=cols)

    lab = labels[["platform_post_id", "sentiment"]].drop_duplicates("platform_post_id")
    m = stories.merge(lab, on="platform_post_id", how="inner")
    if m.empty:
        return pd.DataFrame(columns=cols)

    sent_num = m["sentiment"].map(
        {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    ).fillna(0.0)
    m = m.assign(_sent=sent_num, bucket=pd.to_datetime(m["created_at"]).dt.floor(freq))

    rows = []
    for (sid, bucket), grp in m.groupby(["story_id", "bucket"]):
        s = grp["_sent"]
        rows.append(
            {
                "story_id": int(sid),
                "bucket": bucket,
                "n_posts": len(grp),
                "mean_sentiment": round(float(s.mean()), 3),
                "neg_share": round(float((s == -1).mean()), 3),
                "pos_share": round(float((s == 1).mean()), 3),
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values(["story_id", "bucket"])


def story_framing(
    stories: pd.DataFrame,
    topics_df: pd.DataFrame | None = None,
    topic_summary: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Bundle framing views for a stories member frame.

    Returns dict with keys: topics, keywords, sentiment_timeline.
    Missing optional inputs yield empty structures (no exception).
    """
    topics = (
        story_topics(stories, topics_df, topic_summary)
        if topics_df is not None
        else pd.DataFrame(columns=["story_id", "topic_id", "overlap_n", "share", "topic_terms"])
    )
    keywords = story_keywords(stories)
    timeline = (
        sentiment_timeline(stories, labels)
        if labels is not None
        else pd.DataFrame(
            columns=["story_id", "bucket", "n_posts", "mean_sentiment", "neg_share", "pos_share"]
        )
    )
    return {"topics": topics, "keywords": keywords, "sentiment_timeline": timeline}
