"""Is this post about Kenya: the classifier's call, with the keyword gate behind it.

`kma.measure.domain_bucket` is two regexes. Measured against Tom's blind labels
it is precise and leaky - precision 0.971, recall 0.655 corpus-weighted - and
its misses are plain Kenyan politics that happens to avoid the anchor
vocabulary. The classifier trained on 2026-09-12 scores 0.918 / 1.000 on the
same labels, so this module is what readers should ask.

    from kma import relevance
    relevance.buckets(con, posts)        # kenya / offdomain / ambiguous per row

Two properties worth knowing before using it:

- **The gate is the fallback, not a rival.** Scores are persisted per post, so
  anything collected since the last scoring pass has none, and those rows fall
  back to `domain_bucket`. A mixed answer is the normal case.
- **The threshold is a reader's choice.** Probabilities are persisted, the cut
  is not tuned, and 0.5 is only where the evaluation was run. Near-empty posts
  ride on their @mentions ("@SomeForcePolice [emoji]" scores 0.80), so a pass
  that cares about precision should raise it.
"""

from __future__ import annotations

import logging
import os

import duckdb
import pandas as pd

from kma.db import relevance_source

log = logging.getLogger("kma")

MODEL = os.getenv("KMA_RELEVANCE_MODEL", "kenya-relevance-afroxlmr-2026-09-12")
THRESHOLD = float(os.getenv("KMA_RELEVANCE_THRESHOLD", "0.5"))


def latest_scores_cte(platform: str = "x", model: str = MODEL) -> str:
    """One score per post, the most recent, as a SELECT for a caller's CTE."""
    return f"""
        SELECT platform_post_id, p_kenya FROM {relevance_source(platform, model)}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY scored_at DESC) = 1
    """


def scores(
    con: duckdb.DuckDBPyConnection,
    post_ids,
    platform: str = "x",
    model: str = MODEL,
) -> pd.Series:
    """`p_kenya` for the posts that have one, indexed by post id.

    An empty result rather than an error when nothing is persisted yet: every
    caller has the gate to fall back to.
    """
    ids = pd.DataFrame({"platform_post_id": pd.Series(list(post_ids), dtype="object").astype(str)})
    if ids.empty:
        return pd.Series(dtype="float64", name="p_kenya")
    con.register("_rel_wanted", ids.drop_duplicates())
    try:
        got = con.sql(
            f"""
            WITH s AS ({latest_scores_cte(platform, model)})
            SELECT CAST(s.platform_post_id AS VARCHAR) AS platform_post_id, s.p_kenya
            FROM s SEMI JOIN _rel_wanted w ON w.platform_post_id = s.platform_post_id
            """
        ).df()
    except duckdb.Error:
        log.warning("relevance: no scores for model %s; falling back to the keyword gate", model)
        return pd.Series(dtype="float64", name="p_kenya")
    finally:
        con.unregister("_rel_wanted")
    return got.set_index("platform_post_id")["p_kenya"]


def buckets(
    con: duckdb.DuckDBPyConnection,
    posts: pd.DataFrame,
    *,
    post_id: str = "post_id",
    text: str = "text",
    platform: str = "x",
    model: str = MODEL,
    threshold: float = THRESHOLD,
) -> pd.Series:
    """`kenya` / `offdomain` / `ambiguous` per row: the model where it scored
    the post, `measure.domain_bucket` where it did not.

    The model answers a binary question, so it never returns `ambiguous` - that
    value only survives on posts it has not seen, which is what a reader should
    take it to mean.
    """
    from kma import measure

    gate = posts[text].map(measure.domain_bucket)
    if posts.empty:
        return gate

    p = posts[post_id].astype(str).map(scores(con, posts[post_id], platform, model))
    learned = p.map(lambda v: None if pd.isna(v) else ("kenya" if v >= threshold else "offdomain"))
    out = learned.fillna(gate)
    log.info("relevance: %d of %d posts scored by %s", int(p.notna().sum()), len(posts), model)
    return out.rename("domain")
