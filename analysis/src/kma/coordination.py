"""Coordination layer: detect and characterize coordinated behaviour (CIB).

Pipeline (docs/analysis/phase-3): behavioural traces -> bipartite account x
object projection -> statistical validation (SVN hypergeometric null +
FDR/Bonferroni, percentile baseline, Monte-Carlo time-shuffle for timed
channels) -> multiplex Leiden communities -> corroboration -> scorecards.

    from kma.db import connect
    from kma import coordination as co
    con = connect()
    edges = co.validated_edges(con, "co_retweet")
    layers = co.build_layers(con, ["co_retweet", "text_sim"])
    members, clusters = co.clusters(layers)
    cards = co.scorecards(con, members, layers)

Sampling caveat: twscrape capture is a sample, not a census - absence of a
co-action is not evidence of absence. This bounds recall, not precision, and
applies to every output of this module. Coordination alone is not malicious;
scorecards are a triage tool for human review, never an auto-label.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from kma.db import (
    BUCKET,
    embeddings_source,
    engagements_source,
    hatespeech_source,
    posts_source,
)
from kma.semantic import DIM, MODEL, _slug

log = logging.getLogger("kma")

SAMPLING_CAVEAT = (
    "Capture is a sample, not a census: absence of a co-action is not evidence "
    "of absence. Recall is bounded; precision is not affected."
)

# How each behavioural channel defines "acting together". A channel is one
# multiplex layer; two accounts share an edge in it when they act on the same
# object (a tweet, a hashtag, a near-duplicate text cluster).
CHANNELS = {
    "co_retweet": "Both retweeted the same original tweet. Amplification is the "
    "classic coordination signal; the retweeter census makes this a near-total "
    "count for the objects we snowball, not just a sample.",
    "co_reply": "Both replied under the same tweet - accounts swarming one target.",
    "text_sim": "Both posted near-duplicate text (embedding cosine >= tau). Our "
    "advantage over link/hashtag-only tools: catches paraphrased copypasta.",
    "fast_co_share": "co_retweet within `delta` seconds - scripted synchrony, the "
    "strongest classic signal. Uses the time-shuffle null, not the hypergeometric.",
    "co_hashtag": "Both used the same hashtag (Wave B; needs post-Phase-0 data).",
    "co_url": "Both shared the same normalised outbound URL (Wave B).",
    "co_mention": "Both mentioned the same account (Wave B). Inflated on "
    "reply-heavy data - replies auto-mention their target.",
}

# What every metric a scorecard / edge table exposes actually means, so an
# analyst (or a notebook) never has to reverse-engineer a column. `glossary_md`
# renders this; keep it in sync when adding a metric.
METRIC_GLOSSARY = {
    # --- edge-level (one row per account pair, per channel) ---
    "weight": "Shared action count for the pair: distinct objects both acted on "
    "(untimed channels) or co-actions within `delta` (timed). >= min_repetition "
    "to be tested at all - one shared action is never coordination.",
    "p_value": "Probability the pair's overlap is this large or larger under the "
    "null model. Small = surprising = hard to explain by chance. Default null is "
    "degree-corrected (conditions on how popular each object is), so sharing one "
    "viral tweet is not surprising.",
    "p_uniform": "The same p under the classic uniform hypergeometric null, kept "
    "for comparison. It over-flags on hub objects - that gap is why we switched.",
    "sig_fdr": "Survived Benjamini-Hochberg FDR control at q=0.01 - the sensitive "
    "view. Expect a few false positives among many edges; read as a candidate set.",
    "sig_bonferroni": "Survived Bonferroni control - the high-precision core. "
    "Near-empty on random data by design; treat these edges as load-bearing.",
    "sig_percentile": "In the top weight percentile (CooRnet-style baseline, no "
    "null model). Divergence from the SVN sets flags popular-object noise.",
    "min_gap": "Tightest inter-arrival gap (seconds) between the pair's co-actions. "
    "Small = scripted/synchronised; large = plausibly independent.",
    # --- cluster-level (one row per detected community) ---
    "size": "Number of accounts in the cluster (Leiden community, singletons dropped).",
    "n_channels": "How many independent channels support the cluster. n>=2 is the "
    "strongest evidence short of ground truth - organic co-activity rarely lines "
    "up across retweets AND text AND replies at once.",
    "channels": "Which specific layers support it, e.g. {co_retweet, text_sim}.",
    "internal_edge_share": "Fraction of possible within-cluster pairs that are "
    "actually validated edges. 1.0 = a clique; low = a loose group.",
    # --- characterization (Phase 1 + Phase 2 joined onto clusters) ---
    "suspicion_mean": "Mean Phase-1 bot-likeness of members (0-1). Blends account "
    "age, follower/following ratio, posting rate, duplicate-text ratio, "
    "default-image / empty-bio / digit-handle flags.",
    "anomaly_rank_mean": "Mean isolation-forest anomaly percentile. A second, "
    "unsupervised lens - flags outliers including genuine mega-influencers, so "
    "read alongside suspicion, never alone.",
    "creation_burst_days": "Tightest window (days) holding half the members' "
    "account-creation dates. A narrow burst is a strong CIB signal (a batch of "
    "accounts registered together).",
    "near_dup_rate": "Mean pairwise embedding cosine of member posts (0-1). High = "
    "the cluster is pushing near-identical content.",
    "topic_entropy": "Shannon entropy of the members' topic mix. Low = they "
    "concentrate on one narrative (homogeneous); high = varied.",
    "median_min_gap_s": "Median synchrony of member co-actions (seconds). Tighter "
    "= more scripted.",
    "engagement_per_follower": "Aggregate engagement over combined followers - "
    "amplification efficiency. Unusually high can mean manufactured engagement.",
    # --- the composite ---
    "inauthenticity_index": "Transparent 0-1 triage score = weighted sum of five "
    "percentile-ranked components (see INAUTHENTICITY_WEIGHTS): bot_likeness, "
    "synchrony, homogeneity, concealment, corroboration. NOT a verdict - "
    "legitimate coordination scores non-zero; the component breakdown is what an "
    "analyst acts on, not the scalar.",
    # --- evaluation ---
    "precision/recall/f1": "Synthetic-injection recovery: plant a known cluster, "
    "measure how cleanly the pipeline recovers exactly those accounts.",
    "weighted_precision": "Recovery precision penalised for splitting the planted "
    "group across several clusters (survey metric).",
    "suspicion_effect": "How many null standard deviations the cluster's mean "
    "suspicion sits above random same-size groups. Positive + large = the cluster "
    "is genuinely more bot-like than chance (a falsification test, not a vanity "
    "metric - if it's ~0, detection captured nothing).",
    "homogeneity_effect": "Same, for narrative homogeneity.",
}


def glossary_md() -> str:
    """Markdown rendering of METRIC_GLOSSARY + CHANNELS, for notebook display."""
    ch = "\n".join(f"- **{k}** - {v}" for k, v in CHANNELS.items())
    mt = "\n".join(f"- **`{k}`** - {v}" for k, v in METRIC_GLOSSARY.items())
    return f"## Channels\n\n{ch}\n\n## Metric glossary\n\n{mt}"

# channel -> (trace SQL object expression, wave). text_sim is built in Python
# from embeddings. Wave B channels need post-Phase-0 rows (hashtags/urls/
# mentions arrays); coverage() reports when they become usable.
WAVE_A = ["co_retweet", "co_reply", "text_sim", "fast_co_share"]
WAVE_B = ["co_hashtag", "co_url", "co_mention"]

# Measured on live 18k corpus (2026-07-07); re-sweep when data grows.
DEFAULT_TAU = 0.9
DEFAULT_RESOLUTION = 0.05
DEFAULT_DELTAS: dict[str, int] = {"fast_co_share": 300, "text_sim": 3600}

_SIMPLE_TRACES = {
    "co_retweet": "SELECT author_id, repost_of_id AS action_object, created_at"
    " FROM lp WHERE repost_of_id IS NOT NULL",
    "co_reply": "SELECT author_id, in_reply_to_id AS action_object, created_at"
    " FROM lp WHERE in_reply_to_id IS NOT NULL",
    "co_hashtag": "SELECT author_id, lower(unnest(hashtags)) AS action_object,"
    " created_at FROM lp WHERE len(hashtags) > 0",
    "co_mention": "SELECT author_id, lower(unnest(mentions)) AS action_object,"
    " created_at FROM lp WHERE len(mentions) > 0",
    # strip tracking params + trailing slash; shortener expansion is upstream
    "co_url": """
        SELECT author_id,
               regexp_replace(regexp_replace(regexp_replace(
                   lower(unnest(urls)),
                   '[?&](utm_[^&#]*|fbclid=[^&#]*|gclid=[^&#]*)', '', 'g'),
                   '\\?$', ''), '/+$', '') AS action_object,
               created_at
        FROM lp WHERE len(urls) > 0
    """,
}


# Coordination is scoped to a rolling window, not the whole corpus.
#
# Two reasons. Practically, the projection is a self-join whose cost grows with
# the corpus - unbounded, it OOMed a 3.7Gi host at 195,964 posts and would keep
# getting worse. Substantively, coordination is a time-bounded phenomenon: a ring
# that was active months ago is not the current network, and the collector reads
# the LATEST persisted run to pick targets.
#
# 14 days matches the rest of the system - `collectors/x.MAX_AGE_DAYS` (how far
# search can reach) and `hate_signal.HATE_LOOKBACK_DAYS` (seed scoring). Set 0 to
# restore the old unbounded behaviour.
COORD_LOOKBACK_DAYS = int(os.getenv("COORD_LOOKBACK_DAYS", "14"))

# Upper bound on the hub cap. At cap 50 the projection keeps 98.1% of objects and
# 0.7% of the pairs; the discarded 1.9% are mega-viral posts whose co-amplifiers
# are organic, not coordinated.
HUB_CAP_MAX = int(os.getenv("HUB_CAP_MAX", "100"))

# Which edge filter feeds community detection. Bonferroni, not FDR, and this is
# measured rather than a taste for conservatism.
#
# FDR was the default until 2026-08-02, when the retweeter census scaled
# co_retweet from 3,751 to 140,518 edges in 33 hours and corroborated clusters
# fell 14 -> 1. Three measurements on the live corpus at that scale:
#
#   1. Shuffle control (`null_baseline`): on degree-preserving shuffled traces
#      Bonferroni admits 0 edges (co_retweet) and 1 (co_reply); FDR admits
#      1,817 and 256 - 4% and 28% of pure noise. FDR is behaving as specified
#      (BH allows alpha of DISCOVERIES to be false); the "0/0" recorded on
#      2026-07-08 was a small-corpus artefact that set a false expectation.
#   2. Planted-cluster recovery (`inject_synthetic` + `evaluate_recovery`):
#      identical F1 = 1.0 for both at a 20-account/10-object plant, so FDR's
#      extra ~139,000 edges buy no recall.
#   3. At the detection limit they diverge, and NOT in FDR's favour. With 8
#      accounts sharing 5 objects, both filters contain all 28 planted pairs,
#      but Leiden places 0 of 8 planted accounts in any cluster under FDR
#      versus 8 of 8 under Bonferroni. The marginal edges wire the group to
#      unrelated accounts until CPM can no longer hold it together.
#
# So the failure is in community detection, not the null: the same dilution the
# hub cap exists to prevent, one layer down. Both filters share a sensitivity
# floor of 3 shared objects, so nothing is given up.
#
# `percentile` is disqualified outright - it missed even the 20x10 plant on
# co_retweet. It is still computed and persisted for comparison; do not cluster
# on it.
DEFAULT_EDGE_METHOD = os.getenv("COORD_EDGE_METHOD", "bonferroni")


def _latest_posts_cte(platform: str, lookback_days: int | None = None) -> str:
    days = COORD_LOOKBACK_DAYS if lookback_days is None else lookback_days
    window = (
        f"WHERE created_at > now() - INTERVAL {int(days)} DAY" if days and days > 0 else ""
    )
    return f"""
        SELECT * FROM {posts_source(platform)}
        {window}
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
        ) = 1
    """


_MANUAL_RT = re.compile(r"^\s*RT\s+@", re.IGNORECASE)


def _is_manual_retweet(text: object) -> bool:
    """Old-style `RT @handle: …` posts when is_repost metadata is missing."""
    return isinstance(text, str) and bool(_MANUAL_RT.match(text))


def _filter_clustering_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop retweets and duplicate same-author reposts before similarity clustering.

    Retweets (native or manual RT @…) amplify someone else's claim; they are not
    independent voices repeating it. Same author posting identical text twice is
    spam/repost behaviour, not cross-account coordination."""
    if df.empty:
        return df
    repost = df["is_repost"].fillna(False).astype(bool)
    manual_rt = df["text"].map(_is_manual_retweet)
    out = df[~(repost | manual_rt)].copy()
    if out.empty:
        return out
    out["_text_key"] = out["text"].str.lower().str.strip()
    out = out.sort_values("created_at").drop_duplicates(
        subset=["author_id", "_text_key"], keep="first"
    )
    return out.drop(columns=["_text_key"])


def _eligible_clustering_posts_cte(platform: str) -> str:
    """Latest posts kept for text-sim / story clustering (not co_retweet traces)."""
    return f"""
        SELECT * FROM ({_latest_posts_cte(platform)})
        WHERE NOT COALESCE(is_repost, false)
          AND NOT regexp_matches(coalesce(text, ''), '^\\s*RT\\s+@', 'i')
        QUALIFY row_number() OVER (
            PARTITION BY author_id, lower(trim(text)) ORDER BY created_at
        ) = 1
    """


def _original_posts_sql(prefix: str = "lp") -> str:
    """SQL predicates excluding retweets (for trusted corroboration, etc.)."""
    return (
        f"NOT COALESCE({prefix}.is_repost, false) AND "
        f"NOT regexp_matches(coalesce({prefix}.text, ''), '^\\\\s*RT\\\\s+@', 'i')"
    )


def content_clusters(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    model: str = MODEL,
    tau: float = 0.9,
) -> pd.DataFrame:
    """Near-duplicate content clusters: connected components of the cosine >= tau
    graph over post embeddings. Returns (platform_post_id, cluster_id) for posts
    in components of size >= 2; these are the text_sim action objects."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import radius_neighbors_graph

    df = con.sql(
        f"""
        WITH eligible AS ({_eligible_clustering_posts_cte(platform)}),
             e AS (
                 SELECT platform_post_id, embedding
                 FROM {embeddings_source(platform, _slug(model))}
                 QUALIFY row_number() OVER (
                     PARTITION BY platform_post_id ORDER BY embedded_at DESC
                 ) = 1
             )
        SELECT e.platform_post_id, e.embedding
        FROM e JOIN eligible USING (platform_post_id)
        """
    ).df()
    if df.empty:
        return pd.DataFrame(columns=["platform_post_id", "cluster_id"])
    x = np.asarray(df["embedding"].tolist(), dtype="float32")
    # embeddings are L2-normalised: cosine >= tau <=> euclidean <= sqrt(2 - 2 tau)
    g = radius_neighbors_graph(x, radius=float(np.sqrt(2 - 2 * tau)), mode="connectivity")
    _, labels = connected_components(g, directed=False)
    df["cluster_id"] = labels
    sizes = df["cluster_id"].value_counts()
    df = df[df["cluster_id"].isin(sizes[sizes >= 2].index)]
    return df[["platform_post_id", "cluster_id"]].reset_index(drop=True)


def _engagement_traces(con: duckdb.DuckDBPyConnection, platform: str) -> str | None:
    """Snowballed retweeter incidence as trace rows (untimed - the platform does
    not expose retweet times, so created_at is NULL and timed variants skip
    these rows automatically). None when no engagement data exists yet."""
    src = engagements_source(platform)
    try:
        con.sql(f"SELECT 1 FROM {src} LIMIT 1").fetchall()
    except duckdb.Error:
        return None
    return f"""
        SELECT platform_user_id AS author_id,
               platform_post_id AS action_object,
               CAST(NULL AS TIMESTAMPTZ) AS created_at
        FROM (
            SELECT * FROM {src}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id, platform_user_id, kind
                ORDER BY collected_at DESC
            ) = 1
        )
        WHERE kind = 'retweet'
    """


def traces(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    platform: str = "x",
    model: str = MODEL,
    tau: float = 0.9,
    include_engagements: bool = True,
):
    """Behavioural trace relation (author_id, action_object, created_at) for one
    channel, deduped to one row per (author, object, time).

    co_retweet unions the snowballed retweeter lists (engagements/) with
    post-derived retweets: censused objects then test real incidence under the
    hypergeometric null instead of sampling luck. Timed channels (fast_co_share)
    stay post-only - engagement rows carry no event time."""
    if channel == "co_retweet":
        inner = _SIMPLE_TRACES["co_retweet"]
        if include_engagements:
            eng = _engagement_traces(con, platform)
            if eng is not None:
                inner = f"{inner} UNION ALL {eng}"
    elif channel == "fast_co_share":
        inner = _SIMPLE_TRACES["co_retweet"]
    elif channel in _SIMPLE_TRACES:
        inner = _SIMPLE_TRACES[channel]
    elif channel == "text_sim":
        cc = content_clusters(con, platform, model, tau)
        con.register("_content_clusters", cc)
        inner = """
            SELECT lp.author_id, 'c' || _content_clusters.cluster_id AS action_object,
                   lp.created_at
            FROM lp JOIN _content_clusters USING (platform_post_id)
        """
    else:
        raise ValueError(f"unknown channel {channel!r}")
    return con.sql(
        f"WITH lp AS ({_latest_posts_cte(platform)}) SELECT DISTINCT * FROM ({inner})"
    )


def coverage(con: duckdb.DuckDBPyConnection, platform: str = "x") -> pd.DataFrame:
    """Share of posts carrying each Wave B field - tracks when those channels
    unlock as post-Phase-0 data accrues."""
    return con.sql(
        f"""
        WITH lp AS ({_latest_posts_cte(platform)})
        SELECT count(*) AS posts,
               count(*) FILTER (len(hashtags) > 0) / count(*) AS hashtag_share,
               count(*) FILTER (len(urls) > 0) / count(*) AS url_share,
               count(*) FILTER (len(mentions) > 0) / count(*) AS mention_share
        FROM lp
        """
    ).df()


def _register_traces(con, channel, platform, model, tau) -> str:
    name = f"_tr_{channel}"
    con.register(name, traces(con, channel, platform, model, tau).df())
    return name


def projected_edges(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    platform: str = "x",
    delta: int | None = None,
    min_repetition: int = 2,
    weighting: str = "count",
    model: str = MODEL,
    tau: float = 0.9,
    trace_table: str | None = None,
) -> pd.DataFrame:
    """Account-account projection of one channel's bipartite graph, via a DuckDB
    self-join on action_object (never materialises the account x account matrix).

    weight = shared distinct objects (untimed) or co-action events within
    `delta` seconds (timed). weighting="tfidf" adds a `weight_tfidf` column
    (Pacheco: rare shared objects weigh more than popular ones).
    `trace_table` overrides the trace source (used by the evaluation harness).
    """
    t = trace_table or _register_traces(con, channel, platform, model, tau)
    timed = f"AND abs(epoch(a.created_at) - epoch(b.created_at)) <= {int(delta)}" if delta else ""
    metric = "count(*)" if delta else "count(DISTINCT a.action_object)"
    edges = con.sql(
        f"""
        SELECT a.author_id AS src, b.author_id AS dst,
               {metric} AS weight,
               count(DISTINCT a.action_object) AS n_objects_shared,
               count(*) AS n_coactions,
               min(abs(epoch(a.created_at) - epoch(b.created_at))) AS min_gap
        FROM {t} a JOIN {t} b
          ON a.action_object = b.action_object
         AND a.author_id < b.author_id
         {timed}
        GROUP BY 1, 2
        HAVING {metric} >= {int(min_repetition)}
        """
    ).df()
    if weighting == "tfidf" and not edges.empty:
        tfidf = con.sql(
            f"""
            WITH t AS (SELECT DISTINCT author_id, action_object FROM {t}),
            obj AS (SELECT action_object, count(*) AS df FROM t GROUP BY 1),
            n AS (SELECT count(DISTINCT author_id) AS a FROM t),
            v AS (
                SELECT t.author_id, t.action_object, ln((SELECT a FROM n) / obj.df) AS w
                FROM t JOIN obj USING (action_object)
            ),
            norms AS (SELECT author_id, sqrt(sum(w * w)) AS nrm FROM v GROUP BY 1)
            SELECT a.author_id AS src, b.author_id AS dst,
                   sum(a.w * b.w) / (any_value(na.nrm) * any_value(nb.nrm)) AS weight_tfidf
            FROM v a
            JOIN v b ON a.action_object = b.action_object AND a.author_id < b.author_id
            JOIN norms na ON na.author_id = a.author_id
            JOIN norms nb ON nb.author_id = b.author_id
            GROUP BY 1, 2
            """
        ).df()
        edges = edges.merge(tfidf, on=["src", "dst"], how="left")
    return edges


def activity(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    platform: str = "x",
    model: str = MODEL,
    tau: float = 0.9,
    trace_table: str | None = None,
) -> tuple[pd.DataFrame, int]:
    """Per-account distinct-object degree and the channel's total distinct
    object count M - the inputs of the hypergeometric null (03)."""
    t = trace_table or _register_traces(con, channel, platform, model, tau)
    degrees = con.sql(
        f"SELECT author_id, count(DISTINCT action_object) AS n_objects FROM {t} GROUP BY 1"
    ).df()
    m = con.sql(f"SELECT count(DISTINCT action_object) FROM {t}").fetchone()[0]
    return degrees, int(m)


# --- statistical validation (03) -----------------------------------------


def validate_svn(
    edges: pd.DataFrame,
    degrees: pd.DataFrame,
    n_objects: int,
    method: str = "fdr_bh",
    alpha: float = 0.01,
    object_degrees: pd.Series | None = None,
) -> pd.DataFrame:
    """Statistically Validated Network: per-pair p-value + multiple-testing
    correction over the pairs actually tested.

    Two nulls:
    - `object_degrees` given (default in validated_edges): degree-corrected
      configuration-model null (Chung-Lu / BiCM Poisson tail). Conditions on
      object popularity as well as account activity - the uniform hypergeometric
      is wildly anti-conservative once censused objects carry dozens of
      retweeters (falsified by the 06.2 shuffle check on live data, 2026-07-08).
    - `object_degrees` None: the classic uniform hypergeometric (Tumminello
      2011), kept for comparison / homogeneous-degree data.

    The correction runs over the pairs in `edges` - pass the candidate set you
    would report (weight >= min_repetition). Including every x=1 pair sounds
    rigorous but the hub-driven flood of chance pairs sinks the BH threshold
    below any real cluster's p (observed live 2026-07-08); a single shared
    object is never reportable coordination anyway. method in
    {"fdr_bh", "bonferroni"}.
    """
    from scipy.stats import hypergeom, poisson
    from statsmodels.stats.multitest import multipletests

    out = edges.copy()
    if out.empty:
        out["p_value"], out["validated"] = [], []
        return out
    deg = degrees.set_index("author_id")["n_objects"]
    n_a = deg.reindex(out["src"]).to_numpy()
    n_b = deg.reindex(out["dst"]).to_numpy()
    x = out["n_objects_shared"].to_numpy()
    if object_degrees is not None:
        e_total = float(object_degrees.sum())
        s_sq = float((object_degrees.astype(float) ** 2).sum())
        lam = n_a * n_b * s_sq / (e_total**2)
        out["p_value"] = poisson.sf(x - 1, lam)
    else:
        out["p_value"] = hypergeom.sf(x - 1, n_objects, n_a, n_b)
    if method == "bonferroni":
        out["validated"] = out["p_value"] < alpha / len(out)
    elif method == "fdr_bh":
        out["validated"] = multipletests(out["p_value"], alpha=alpha, method="fdr_bh")[0]
    else:
        raise ValueError(f"unknown method {method!r}")
    return out


def object_degrees(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    platform: str = "x",
    model: str = MODEL,
    tau: float = 0.9,
    trace_table: str | None = None,
) -> pd.Series:
    """Distinct-account degree per action object - the popularity marginal the
    degree-corrected null conditions on."""
    t = trace_table or _register_traces(con, channel, platform, model, tau)
    df = con.sql(
        f"SELECT action_object, count(DISTINCT author_id) AS n FROM {t} GROUP BY 1"
    ).df()
    return df.set_index("action_object")["n"]


def validate_curveball(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    n_iter: int = 200,
    alpha: float = 0.01,
    min_repetition: int = 2,
    method: str = "fdr_bh",
    seed: int = 0,
    platform: str = "x",
    model: str = MODEL,
    tau: float = 0.9,
    trace_table: str | None = None,
) -> pd.DataFrame:
    """Exact degree-preserving Monte-Carlo null (curveball, Strona 2014):
    randomise the bipartite incidence keeping BOTH account and object degrees,
    recount shared objects per pair. Cross-check for the analytic Poisson null
    in validate_svn - empirical p floors at 1/(n_iter+1), so use FDR, not
    Bonferroni, on its output."""
    from statsmodels.stats.multitest import multipletests

    t = trace_table or _register_traces(con, channel, platform, model, tau)
    tr = con.sql(f"SELECT DISTINCT author_id, action_object FROM {t}").df()
    if tr.empty:
        return pd.DataFrame(columns=["src", "dst", "weight", "p_value", "validated"])
    authors = sorted(tr["author_id"].unique())
    sets = {a: set() for a in authors}
    for a, o in zip(tr["author_id"], tr["action_object"]):
        sets[a].add(o)

    def pair_counts(user_sets: dict) -> dict:
        by_obj: dict[str, list] = {}
        for a, objs in user_sets.items():
            for o in objs:
                by_obj.setdefault(o, []).append(a)
        counts: dict[tuple, int] = {}
        for users in by_obj.values():
            users.sort()
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    key = (users[i], users[j])
                    counts[key] = counts.get(key, 0) + 1
        return counts

    obs = pair_counts(sets)
    if not obs:
        return pd.DataFrame(columns=["src", "dst", "weight", "p_value", "validated"])
    rng = np.random.default_rng(seed)

    def trade(user_sets: dict, n_trades: int) -> None:
        keys = list(user_sets)
        for _ in range(n_trades):
            a, b = rng.choice(len(keys), 2, replace=False)
            sa, sb = user_sets[keys[a]], user_sets[keys[b]]
            only_a, only_b = list(sa - sb), list(sb - sa)
            if not only_a or not only_b:
                continue
            pool = only_a + only_b
            rng.shuffle(pool)
            take_a = set(pool[: len(only_a)])
            user_sets[keys[a]] = (sa & sb) | take_a
            user_sets[keys[b]] = (sa & sb) | (set(pool) - take_a)

    work = {a: set(s) for a, s in sets.items()}
    trade(work, 5 * len(authors))  # burn-in
    exceed = dict.fromkeys(obs, 0)
    for _ in range(n_iter):
        trade(work, len(authors))
        null = pair_counts(work)
        for p, c in obs.items():
            if null.get(p, 0) >= c:
                exceed[p] += 1
    pairs = list(obs)
    out = pd.DataFrame(
        {
            "src": [p[0] for p in pairs],
            "dst": [p[1] for p in pairs],
            "weight": [obs[p] for p in pairs],
            "p_value": [(1 + exceed[p]) / (1 + n_iter) for p in pairs],
        }
    )
    if method == "bonferroni":
        out["validated"] = out["p_value"] < alpha / len(out)
    else:
        out["validated"] = multipletests(out["p_value"], alpha=alpha, method="fdr_bh")[0]
    return out[out["weight"] >= min_repetition].reset_index(drop=True)


def _object_groups(tr: pd.DataFrame):
    obj_codes = tr["action_object"].astype("category").cat.codes.to_numpy()
    order = np.argsort(obj_codes, kind="stable")
    bounds = np.flatnonzero(np.diff(obj_codes[order])) + 1
    return [g for g in np.split(order, bounds) if len(g) >= 2]


def _codelta_counts(groups, authors: np.ndarray, times: np.ndarray, delta: int) -> dict:
    counts: dict[tuple, int] = {}
    for g in groups:
        t = times[g]
        srt = g[np.argsort(t, kind="stable")]
        ts = times[srt]
        j = 0
        for i in range(len(srt)):
            while ts[i] - ts[j] > delta:
                j += 1
            for k in range(j, i):
                a, b = authors[srt[k]], authors[srt[i]]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1
    return counts


def validate_montecarlo(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    delta: int,
    n_iter: int = 500,
    alpha: float = 0.01,
    min_repetition: int = 2,
    method: str = "fdr_bh",
    seed: int = 0,
    platform: str = "x",
    model: str = MODEL,
    tau: float = 0.9,
    trace_table: str | None = None,
) -> pd.DataFrame:
    """Time-shuffle Monte-Carlo null for timed channels (fast co-share,
    synchronised text_sim): permute created_at within each account, recompute
    co-within-delta counts, empirical one-sided p per pair, then the same
    multiple-testing correction as validate_svn."""
    from statsmodels.stats.multitest import multipletests

    t = trace_table or _register_traces(con, channel, platform, model, tau)
    tr = con.sql(
        f"SELECT author_id, action_object, created_at FROM {t} WHERE created_at IS NOT NULL"
    ).df()
    if tr.empty:
        return pd.DataFrame(
            columns=["src", "dst", "weight", "min_gap", "p_value", "validated"]
        )
    authors = tr["author_id"].to_numpy()
    times = (tr["created_at"].astype("int64") // 10**9).to_numpy()
    groups = _object_groups(tr)
    obs = _codelta_counts(groups, authors, times, delta)
    tested = {p: c for p, c in obs.items() if c >= min_repetition}
    if not tested:
        return pd.DataFrame(
            columns=["src", "dst", "weight", "min_gap", "p_value", "validated"]
        )
    author_groups = pd.Series(np.arange(len(tr))).groupby(authors).apply(np.asarray)
    rng = np.random.default_rng(seed)
    exceed = dict.fromkeys(tested, 0)
    shuffled = times.copy()
    for _ in range(n_iter):
        for idx in author_groups:
            shuffled[idx] = rng.permutation(times[idx])
        null = _codelta_counts(groups, authors, shuffled, delta)
        for p, c in tested.items():
            if null.get(p, 0) >= c:
                exceed[p] += 1
    pairs = list(tested)
    out = pd.DataFrame(
        {
            "src": [p[0] for p in pairs],
            "dst": [p[1] for p in pairs],
            "weight": [tested[p] for p in pairs],
            "p_value": [(1 + exceed[p]) / (1 + n_iter) for p in pairs],
        }
    )
    gaps = (
        con.sql(
            f"""
            SELECT a.author_id AS src, b.author_id AS dst,
                   min(abs(epoch(a.created_at) - epoch(b.created_at))) AS min_gap
            FROM {t} a JOIN {t} b
              ON a.action_object = b.action_object AND a.author_id < b.author_id
            GROUP BY 1, 2
            """
        ).df()
    )
    out = out.merge(gaps, on=["src", "dst"], how="left")
    if method == "bonferroni":
        out["validated"] = out["p_value"] < alpha / len(out)
    else:
        out["validated"] = multipletests(out["p_value"], alpha=alpha, method="fdr_bh")[0]
    return out


def percentile_filter(edges: pd.DataFrame, q: float = 0.995) -> pd.DataFrame:
    """CooRnet-style baseline: keep edges at/above the q weight quantile. No
    null model - cannot separate surprising from popular; comparison only."""
    if edges.empty:
        return edges.assign(validated=pd.Series(dtype=bool))
    cut = edges["weight"].quantile(q)
    return edges[edges["weight"] >= cut].assign(validated=True).reset_index(drop=True)


def validated_edges(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    platform: str = "x",
    delta: int | None = None,
    min_repetition: int = 2,
    alpha: float = 0.01,
    q: float = 0.995,
    model: str = MODEL,
    tau: float = 0.9,
    n_iter: int = 500,
    trace_table: str | None = None,
    hub_cap: int | None = None,
    *,
    stats: dict | None = None,
) -> pd.DataFrame:
    """One channel end to end: projection + all three edge filters. Returns the
    tested edges with `p_value`, `sig_bonferroni`, `sig_fdr`, `sig_percentile`.

    Untimed channels use the degree-corrected SVN (configuration-model null;
    `p_uniform` keeps the classic hypergeometric for comparison) over the
    incidence with hub objects excluded: objects acted on by more than
    `hub_cap` accounts (default max(50, 5% of accounts)) carry no coordination
    signal - pairs sharing only a mega-viral tweet are organic - and one such
    hub otherwise dilutes the aggregate null rate until real clusters vanish
    (doc 02 scaling note). Excluded hubs are logged. Pass `delta` for the
    Monte-Carlo time-shuffle null instead.

    `stats`, when given, is filled in place with this channel's counters. Every
    one of them is a by-product of the pass that already runs - nothing here
    triggers a second scan of the traces. That is not an optimisation detail:
    the alternative (a helper that recomputes them) re-runs `traces()` on the
    host that already OOMs at `projected_edges`."""
    t = trace_table or _register_traces(con, channel, platform, model, tau)
    n_accounts = con.sql(f"SELECT count(DISTINCT author_id) FROM {t}").fetchone()[0]
    if stats is not None:
        stats["channel"] = channel
        stats["timed"] = delta is not None
        stats["min_repetition"] = int(min_repetition)
        stats["accounts_all"] = int(n_accounts)
        stats["traces_all"] = int(con.sql(f"SELECT count(*) FROM {t}").fetchone()[0])
    if delta is not None:
        mc = lambda m: validate_montecarlo(  # noqa: E731
            con, channel, delta, n_iter=n_iter, alpha=alpha,
            min_repetition=min_repetition, method=m, trace_table=t,
        )
        out = mc("fdr_bh").rename(columns={"validated": "sig_fdr"})
        out["sig_bonferroni"] = mc("bonferroni")["validated"]
    else:
        obj_deg = object_degrees(con, channel, platform, trace_table=t)
        # The 5%-of-accounts term was meant to lift the cap on tiny corpora, but
        # it scales with the corpus and eventually stops filtering: at 64,002
        # amplifying accounts it gives 3,200, while the busiest object has 1,171
        # amplifiers - so nothing was excluded. Measured 2026-08-01 on the 14-day
        # window: 1.9% of objects (the viral tail) generated 99.3% of all
        # projected pairs, 36.8M of them. That both OOMs a small host and does
        # exactly what this cap exists to prevent - "one such hub dilutes the
        # aggregate null rate until real clusters vanish". Bounded above so it
        # stays tight as the corpus grows.
        cap = (
            hub_cap
            if hub_cap is not None
            else max(50, min(int(0.05 * n_accounts), HUB_CAP_MAX))
        )
        hubs = obj_deg[obj_deg > cap]
        if stats is not None:
            stats["hub_cap"] = int(cap)
            stats["hub_objects"] = int(len(hubs))
            stats["hub_max_degree"] = int(hubs.max()) if len(hubs) else 0
            stats["objects_all"] = int(len(obj_deg))
        if len(hubs):
            import logging

            logging.getLogger("kma").info(
                "%s: excluding %d hub object(s) with degree > %d (max %d)",
                channel, len(hubs), cap, int(hubs.max()),
            )
            con.register("_hub_objects", hubs.reset_index()[["action_object"]])
            con.register(
                f"{t}_nohub",
                con.sql(
                    f"SELECT * FROM {t} WHERE action_object NOT IN "
                    "(SELECT action_object FROM _hub_objects)"
                ).df(),
            )
            t = f"{t}_nohub"
            obj_deg = obj_deg[obj_deg <= cap]
        edges = projected_edges(
            con, channel, platform, min_repetition=min_repetition,
            weighting="tfidf", trace_table=t,
        )
        degrees, m = activity(con, channel, platform, trace_table=t)
        if stats is not None:
            # `nohub_amp` counts accounts that survived the hub cap; `pairable`
            # counts those that can actually appear in an edge. `projected_edges`
            # requires `min_repetition` distinct shared objects, so an account
            # holding a single non-hub trace is structurally incapable of being
            # in any edge, no matter what else is collected. Measured 2026-08-01
            # on the live corpus: 148,254 accounts collected, 25,083 survivors,
            # 7,079 pairable - the survivor count overstates by 3.5x. Report
            # `pairable`; `nohub_amp` is kept only to show the gap.
            stats["objects_nohub"] = int(m)
            stats["nohub_amp"] = int(len(degrees))
            stats["pairable"] = int((degrees["n_objects"] >= min_repetition).sum())
            stats["traces_nohub"] = int(con.sql(f"SELECT count(*) FROM {t}").fetchone()[0])
        out = validate_svn(
            edges, degrees, m, "fdr_bh", alpha, object_degrees=obj_deg
        ).rename(columns={"validated": "sig_fdr"})
        out["sig_bonferroni"] = validate_svn(
            edges, degrees, m, "bonferroni", alpha, object_degrees=obj_deg
        )["validated"]
        out["p_uniform"] = validate_svn(edges, degrees, m, "fdr_bh", alpha)["p_value"]
    pct = percentile_filter(out, q)
    keys = set(zip(pct["src"], pct["dst"])) if not pct.empty else set()
    out["sig_percentile"] = [k in keys for k in zip(out["src"], out["dst"])]
    if stats is not None:
        stats["edges_tested"] = int(len(out))
        for name in ("fdr", "bonferroni", "percentile"):
            stats[f"edges_{name}"] = int(out[f"sig_{name}"].sum()) if len(out) else 0
        stats["accounts_tested"] = (
            len(set(out["src"]) | set(out["dst"])) if len(out) else 0
        )
        sig = out[out["sig_fdr"]] if len(out) else out
        stats["accounts_fdr"] = len(set(sig["src"]) | set(sig["dst"])) if len(sig) else 0
    return out


def edge_report(edges: pd.DataFrame) -> pd.DataFrame:
    """Edge counts + Jaccard overlaps of the three filters (03 reporting)."""
    sets = {
        m: set(zip(edges.loc[edges[f"sig_{m}"], "src"], edges.loc[edges[f"sig_{m}"], "dst"]))
        for m in ("bonferroni", "fdr", "percentile")
    }
    rows = []
    for m, s in sets.items():
        rows.append({"method": m, "edges": len(s)})
    for a, b in [("bonferroni", "fdr"), ("bonferroni", "percentile"), ("fdr", "percentile")]:
        u = sets[a] | sets[b]
        rows.append(
            {"method": f"jaccard({a},{b})", "edges": len(sets[a] & sets[b]) / len(u) if u else 0}
        )
    return pd.DataFrame(rows)


# --- multiplex + communities (04) -----------------------------------------


def build_layers(
    con: duckdb.DuckDBPyConnection,
    channels: list[str] = WAVE_A,
    platform: str = "x",
    method: str = DEFAULT_EDGE_METHOD,
    deltas: dict[str, int] | None = None,
    *,
    stats: dict[str, dict] | None = None,
    **params,
) -> dict[str, pd.DataFrame]:
    """dict channel -> validated edge list (one multiplex layer per channel).
    `method` picks the edge filter ("bonferroni", "fdr", "percentile");
    `deltas` maps a channel to a co-action window for the timed variant.

    `stats`, when given, is filled with channel -> counters. It is an explicit
    parameter rather than part of `**params` on purpose: forwarded through
    `params` one dict would be shared by every channel and each would overwrite
    the last."""
    merged_deltas = {**DEFAULT_DELTAS, **(deltas or {})}
    layers = {}
    for ch in channels:
        ch_stats: dict | None = {} if stats is not None else None
        e = validated_edges(
            con, ch, platform, delta=merged_deltas.get(ch), stats=ch_stats, **params
        )
        layers[ch] = e[e[f"sig_{method}"]].reset_index(drop=True)
        if ch_stats is not None:
            ch_stats["method"] = method
            ch_stats["edges_kept"] = int(len(layers[ch]))
            stats[ch] = ch_stats
    return layers


def _igraph(edges: pd.DataFrame, weight_col: str = "weight"):
    import igraph as ig

    nodes = sorted(set(edges["src"]) | set(edges["dst"]))
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(
        n=len(nodes),
        edges=[(idx[s], idx[d]) for s, d in zip(edges["src"], edges["dst"])],
    )
    g.vs["name"] = nodes
    g.es["weight"] = edges[weight_col].astype(float).tolist()
    return g


def communities(edges: pd.DataFrame, resolution: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Leiden over CPM (well-connected communities, interpretable resolution =
    intra-density threshold). Returns (author_id, cluster_id), singletons
    dropped."""
    import leidenalg as la

    if edges.empty:
        return pd.DataFrame(columns=["author_id", "cluster_id"])
    g = _igraph(edges)
    part = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed,
    )
    df = pd.DataFrame({"author_id": g.vs["name"], "cluster_id": part.membership})
    sizes = df["cluster_id"].value_counts()
    return df[df["cluster_id"].isin(sizes[sizes >= 2].index)].reset_index(drop=True)


def resolution_sweep(
    edges: pd.DataFrame, gammas: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5)
) -> pd.DataFrame:
    """Community count / size profile across CPM resolutions - robust clusters
    persist across the sweep."""
    rows = []
    for g in gammas:
        m = communities(edges, resolution=g)
        sizes = m["cluster_id"].value_counts()
        rows.append(
            {
                "gamma": g,
                "clusters": int(m["cluster_id"].nunique()),
                "accounts": len(m),
                "largest": int(sizes.max()) if len(sizes) else 0,
                "median_size": float(sizes.median()) if len(sizes) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def aggregate_layers(layers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sum max-normalised layer weights into one multiplex graph, tracking the
    per-edge supporting channels."""
    frames = []
    for ch, e in layers.items():
        if e.empty:
            continue
        f = e[["src", "dst", "weight", "min_gap"]].copy()
        f["weight"] = f["weight"] / f["weight"].max()
        f["channel"] = ch
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["src", "dst", "weight", "min_gap", "channels", "n_channels"])
    allf = pd.concat(frames, ignore_index=True)
    agg = allf.groupby(["src", "dst"], as_index=False).agg(
        weight=("weight", "sum"),
        min_gap=("min_gap", "min"),
        channels=("channel", lambda s: sorted(set(s))),
    )
    agg["n_channels"] = agg["channels"].str.len()
    return agg


def corroborate(layers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per account pair: which validated layers support it. n_channels >= 2 is
    the strongest evidence available short of ground truth."""
    agg = aggregate_layers(layers)
    return agg[["src", "dst", "channels", "n_channels", "min_gap"]]


def clusters(
    layers: dict[str, pd.DataFrame],
    resolution: float = 0.05,
    min_size: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leiden on the aggregated multiplex + per-cluster corroboration stats.

    Returns (members, summary): members = (author_id, cluster_id); summary has
    size, supporting channels, per-layer intra-cluster edge density, and the
    share of internal pairs that are SVN-validated in >= 1 layer."""
    agg = aggregate_layers(layers)
    members = communities(agg, resolution=resolution, seed=seed)
    sizes = members["cluster_id"].value_counts()
    members = members[members["cluster_id"].isin(sizes[sizes >= min_size].index)]
    rows = []
    pair_channels = {
        (r.src, r.dst): r.channels for r in agg.itertuples()
    }
    for cid, grp in members.groupby("cluster_id"):
        nodes = set(grp["author_id"])
        n = len(nodes)
        possible = n * (n - 1) / 2
        internal = {
            p: chs for p, chs in pair_channels.items() if p[0] in nodes and p[1] in nodes
        }
        chans = sorted({c for chs in internal.values() for c in chs})
        density = {
            f"density_{ch}": sum(ch in chs for chs in internal.values()) / possible
            for ch in layers
        }
        rows.append(
            {
                "cluster_id": cid,
                "size": n,
                "channels": chans,
                "n_channels": len(chans),
                "internal_edges": len(internal),
                "internal_edge_share": len(internal) / possible,
                **density,
            }
        )
    summary = (
        pd.DataFrame(rows).sort_values(
            ["n_channels", "internal_edge_share", "size"], ascending=False, ignore_index=True
        )
        if rows
        else pd.DataFrame(
            columns=["cluster_id", "size", "channels", "n_channels", "internal_edges",
                     "internal_edge_share"]
        )
    )
    return members.reset_index(drop=True), summary


# --- cluster display names --------------------------------------------------

_CHANNEL_FALLBACK: dict[str, str] = {
    "co_retweet": "retweet ring",
    "co_reply": "reply ring",
    "text_sim": "duplicate text",
    "fast_co_share": "fast sharing",
    "co_hashtag": "shared hashtags",
    "co_url": "shared links",
    "co_mention": "shared mentions",
}


def _channel_fallback(channels) -> str:
    chans = list(channels or [])
    if len(chans) >= 2:
        return "multi-signal"
    if not chans:
        return "cluster"
    return _CHANNEL_FALLBACK.get(chans[0], chans[0].replace("co_", ""))


def _ctfidf_top_terms(docs: list[str], top_terms: int = 8) -> list[list[str]]:
    """Distinctive terms per document via c-TF-IDF (same recipe as topic_summary)."""
    from sklearn.feature_extraction.text import CountVectorizer

    from kma.semantic import STOPWORDS, _CLEAN

    if not docs:
        return []
    cleaned = [_CLEAN.sub(" ", doc.lower()) for doc in docs]
    cv = CountVectorizer(stop_words=STOPWORDS, min_df=1, token_pattern=r"[a-z]{3,}")
    try:
        counts = cv.fit_transform(cleaned).toarray()
    except ValueError:
        return [[] for _ in docs]
    words_per_class = np.maximum(counts.sum(axis=1, keepdims=True), 1)
    tf = counts / words_per_class
    idf = np.log(1 + words_per_class.mean() / np.maximum(counts.sum(axis=0), 1))
    ctfidf = tf * idf
    vocab = cv.get_feature_names_out()
    return [
        [vocab[j] for j in np.argsort(ctfidf[i])[::-1][:top_terms]]
        for i in range(len(docs))
    ]


def _short_name(terms: list[str], max_words: int) -> str:
    words = [t for t in terms if t and len(t) >= 3][:max_words]
    return " ".join(words)


def _dedupe_names(names: list[str], term_lists: list[list[str]], max_words: int) -> list[str]:
    used: set[str] = set()
    out: list[str] = []
    for name, terms in zip(names, term_lists):
        candidate = name
        n = max_words + 1
        while candidate in used and n <= len(terms):
            candidate = _short_name(terms, n)
            n += 1
        if candidate in used:
            candidate = f"{candidate} alt"
        used.add(candidate)
        out.append(candidate)
    return out


def cluster_names(
    con: duckdb.DuckDBPyConnection,
    members: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    platform: str = "x",
    max_words: int = 3,
    posts_view: str | None = None,
) -> pd.DataFrame:
    """Short human-readable names (1-3 words) for coordination clusters.

    Names come from c-TF-IDF terms over member posts; clusters with too little
    text fall back to a channel descriptor from `summary`. Returns
    (cluster_id, name, label) where label is ``name (n=size)`` for charts."""
    cols = ["cluster_id", "name", "label"]
    if members.empty:
        return pd.DataFrame(columns=cols)
    sizes = members.groupby("cluster_id")["author_id"].size()
    summary_by_id = (
        summary.set_index("cluster_id") if summary is not None and len(summary) else None
    )
    con.register("_coord_members", members[["cluster_id", "author_id"]])
    try:
        posts = posts_view or f"({ _latest_posts_cte(platform) })"
        texts = con.sql(
            f"""
            SELECT m.cluster_id, p.text
            FROM _coord_members m
            JOIN {posts} p ON p.author_id = m.author_id
            WHERE p.text IS NOT NULL AND length(trim(p.text)) > 0
            """
        ).df()
    finally:
        con.unregister("_coord_members")

    cluster_ids = sorted(members["cluster_id"].unique())
    docs = [
        " ".join(texts.loc[texts["cluster_id"] == cid, "text"].tolist())
        for cid in cluster_ids
    ]
    term_lists = _ctfidf_top_terms(docs)
    raw_names = []
    for cid, terms in zip(cluster_ids, term_lists):
        name = _short_name(terms, max_words)
        if not name and summary_by_id is not None and cid in summary_by_id.index:
            name = _channel_fallback(summary_by_id.loc[cid, "channels"])
        if not name:
            name = "cluster"
        raw_names.append(name)
    names = _dedupe_names(raw_names, term_lists, max_words)
    rows = [
        {
            "cluster_id": cid,
            "name": name,
            "label": f"{name} (n={int(sizes[cid])})",
        }
        for cid, name in zip(cluster_ids, names)
    ]
    return pd.DataFrame(rows, columns=cols)


def with_cluster_names(
    df: pd.DataFrame,
    names: pd.DataFrame,
    *,
    drop_id: bool = False,
) -> pd.DataFrame:
    """Attach name + label columns; optionally drop cluster_id for display tables."""
    out = df.merge(names[["cluster_id", "name", "label"]], on="cluster_id", how="left")
    if drop_id:
        out = out.drop(columns=["cluster_id"])
    return out


# --- characterization (05) -------------------------------------------------

# Transparent triage weights over percentile-ranked components; calibrated
# against the 06 evaluation. Legitimate coordination scores non-zero - the
# component breakdown, not the scalar, is what an analyst acts on.
INAUTHENTICITY_WEIGHTS = {
    "bot_likeness": 0.30,
    "synchrony": 0.20,
    "homogeneity": 0.20,
    "concealment": 0.15,
    "corroboration": 0.15,
}


def _burstiness_days(created: pd.Series, share: float = 0.5) -> float:
    """Tightest window (days) holding `share` of member account creations."""
    t = np.sort(created.dropna().astype("int64").to_numpy()) / 86_400e9
    k = max(int(np.ceil(share * len(t))), 2)
    if len(t) < k:
        return float("nan")
    return float((t[k - 1 :] - t[: len(t) - k + 1]).min())


def _member_posts(con, platform: str) -> str:
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE _member_posts AS {_latest_posts_cte(platform)}"
    )
    return "_member_posts"


def scorecards(
    con: duckdb.DuckDBPyConnection,
    members: pd.DataFrame,
    layers: dict[str, pd.DataFrame] | None = None,
    platform: str = "x",
    model: str = MODEL,
    topics: pd.DataFrame | None = None,
    max_posts_per_cluster: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-cluster scorecard integrating Phase 1 (authenticity) + Phase 2
    (narrative) + coordination-intrinsic + impact signals, ranked by a
    transparent inauthenticity index (see INAUTHENTICITY_WEIGHTS).

    `topics` = assign_topics() output, optional (adds topic entropy).
    Not an auto-label: triage for human review."""
    from kma.authenticity import authenticity_score

    auth = authenticity_score(con, platform=platform)
    auth = auth.set_index("platform_user_id")
    p90 = auth["suspicion"].quantile(0.9)
    rng = np.random.default_rng(seed)

    lp = _member_posts(con, platform)
    posts = con.sql(
        f"""
        SELECT author_id, platform_post_id,
               like_count + reply_count + repost_count + quote_count AS engagement
        FROM {lp}
        """
    ).df()
    emb = con.sql(
        f"""
        SELECT platform_post_id, embedding
        FROM {embeddings_source(platform, _slug(model))}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY embedded_at DESC
        ) = 1
        """
    ).df()
    emb = emb.merge(posts[["platform_post_id", "author_id"]], on="platform_post_id")

    corroborated = corroborate(layers) if layers else None
    topic_by_post = (
        topics.set_index("platform_post_id")["topic"] if topics is not None else None
    )

    rows = []
    for cid, grp in members.groupby("cluster_id"):
        ids = grp["author_id"].tolist()
        a = auth.reindex(ids).dropna(subset=["suspicion"])
        mposts = posts[posts["author_id"].isin(ids)]
        row = {
            "cluster_id": cid,
            "size": len(ids),
            "n_posts": len(mposts),
            # Phase 1
            "suspicion_mean": a["suspicion"].mean(),
            "suspicion_median": a["suspicion"].median(),
            "share_suspicion_p90": (a["suspicion"] > p90).mean(),
            "anomaly_rank_mean": a["anomaly_rank"].mean(),
            "creation_burst_days": _burstiness_days(a["created_at"]),
            "share_default_image": a["default_profile_image"].mean(),
            "share_empty_bio": a["empty_bio"].mean(),
            "handle_digit_ratio_mean": a["handle_digit_ratio"].mean(),
            "shared_profile_image": 1 - a["profile_image_url"].nunique() / max(len(a), 1),
            # impact
            "followers_sum": a["followers_count"].sum(),
            "engagement_sum": mposts["engagement"].sum(),
            "engagement_per_follower": mposts["engagement"].sum()
            / max(a["followers_count"].sum(), 1),
        }
        # Phase 2: narrative homogeneity via mean pairwise cosine of member posts
        e = emb[emb["author_id"].isin(ids)]
        if len(e) > max_posts_per_cluster:
            e = e.sample(max_posts_per_cluster, random_state=rng.integers(2**31))
        if len(e) >= 2:
            v = np.asarray(e["embedding"].tolist(), dtype="float32")
            sim = v @ v.T
            row["near_dup_rate"] = float(
                sim[np.triu_indices(len(v), 1)].mean()
            )
        else:
            row["near_dup_rate"] = float("nan")
        if topic_by_post is not None and len(mposts):
            t = topic_by_post.reindex(mposts["platform_post_id"]).dropna()
            t = t[t != -1]
            if len(t):
                p = t.value_counts(normalize=True).to_numpy()
                row["topic_entropy"] = float(-(p * np.log(p)).sum())
                row["dominant_topic"] = int(t.value_counts().idxmax())
        # coordination-intrinsic
        if corroborated is not None and not corroborated.empty:
            mask = corroborated["src"].isin(ids) & corroborated["dst"].isin(ids)
            intra = corroborated[mask]
            row["n_channels"] = (
                len({c for chs in intra["channels"] for c in chs}) if len(intra) else 0
            )
            row["median_min_gap_s"] = intra["min_gap"].median() if len(intra) else float("nan")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    def rank(s: pd.Series, invert: bool = False) -> pd.Series:
        r = s.rank(pct=True)
        return (1 - r) if invert else r

    components = {
        "bot_likeness": rank(df["suspicion_mean"]),
        "synchrony": rank(df.get("median_min_gap_s", pd.Series(np.nan, index=df.index)),
                          invert=True),
        "homogeneity": rank(df["near_dup_rate"]),
        "concealment": rank(
            df[["share_default_image", "share_empty_bio", "handle_digit_ratio_mean",
                "shared_profile_image"]].mean(axis=1)
            + rank(df["creation_burst_days"], invert=True).fillna(0) * 0
        ),
        "corroboration": rank(df.get("n_channels", pd.Series(0, index=df.index))),
    }
    for name, comp in components.items():
        df[f"ix_{name}"] = comp.fillna(0.0)
    df["inauthenticity_index"] = sum(
        w * df[f"ix_{k}"] for k, w in INAUTHENTICITY_WEIGHTS.items()
    )
    df = attach_hate_columns(con, members, df, platform=platform)
    return df.sort_values("inauthenticity_index", ascending=False, ignore_index=True)


def attach_hate_columns(
    con: duckdb.DuckDBPyConnection,
    members: pd.DataFrame,
    scorecard: pd.DataFrame,
    platform: str = "x",
) -> pd.DataFrame:
    """Add per-cluster hate columns, reported as a SEPARATE `hate_index`.

    Deliberately not folded into `INAUTHENTICITY_WEIGHTS`: coordination and hate
    are different claims. A cluster can be plainly coordinated and entirely
    non-toxic (a campaign amplification ring), or toxic and uncoordinated.
    Mixing them would make the existing index uninterpretable and would silently
    reshuffle every historical ranking. Triage is 2-D: inauthenticity x hate.

    `toxic_concentration` is the Herfindahl index of flagged posts across
    members - the cluster-level restatement of "one post is not a network". A
    cluster where a single member carries all the toxicity is a loud individual
    inside an otherwise ordinary group, not a hate network, and scores near 1.
    """
    out = scorecard.copy()
    cols = ["toxic_share", "coded_share", "n_toxic_authors", "toxic_concentration",
            "hate_index"]
    try:
        con.register("_hate_members", members[["cluster_id", "author_id"]])
        try:
            posts = _member_posts(con, platform)
            stats = con.sql(
                f"""
                WITH lh AS (
                    SELECT * FROM {hatespeech_source(platform)}
                    QUALIFY row_number() OVER (
                        PARTITION BY platform_post_id ORDER BY scored_at DESC
                    ) = 1
                ), scoped AS (
                    SELECT m.cluster_id, p.author_id,
                        (h.label <> 'neither' OR COALESCE(h.hate_flag, FALSE)) AS toxic,
                        COALESCE(TRY_CAST(h.coded_suspect AS BOOLEAN), FALSE) AS coded
                    FROM _hate_members m
                    JOIN {posts} p ON p.author_id = m.author_id
                    JOIN lh h ON h.platform_post_id = p.platform_post_id
                ), per_author AS (
                    SELECT cluster_id, author_id,
                        count(*) AS n_posts,
                        count(*) FILTER (WHERE toxic) AS n_toxic,
                        count(*) FILTER (WHERE coded) AS n_coded
                    FROM scoped GROUP BY cluster_id, author_id
                ), cluster_tot AS (
                    SELECT cluster_id,
                        sum(n_posts) AS tot_posts,
                        sum(n_toxic) AS tot_toxic,
                        sum(n_coded) AS tot_coded
                    FROM per_author GROUP BY cluster_id
                ), joined AS (
                    SELECT a.*, t.tot_posts, t.tot_toxic, t.tot_coded
                    FROM per_author a JOIN cluster_tot t USING (cluster_id)
                )
                SELECT cluster_id,
                    max(tot_toxic) * 1.0 / nullif(max(tot_posts), 0) AS toxic_share,
                    max(tot_coded) * 1.0 / nullif(max(tot_posts), 0) AS coded_share,
                    count(*) FILTER (WHERE n_toxic >= 3) AS n_toxic_authors,
                    sum(pow(n_toxic * 1.0 / nullif(tot_toxic, 0), 2)) AS toxic_concentration
                FROM joined GROUP BY cluster_id
                """
            ).df()
        finally:
            con.unregister("_hate_members")
    except duckdb.Error:
        # A missing prefix is expected before the first enrich run; a malformed
        # query is not, and must never masquerade as "no data".
        log.exception("hate columns unavailable; reporting nulls")
        for c in cols:
            out[c] = np.nan
        return out

    out = out.merge(stats, on="cluster_id", how="left")
    for c in ("toxic_share", "coded_share", "n_toxic_authors", "toxic_concentration"):
        if c not in out.columns:
            out[c] = np.nan
    # Spread beats level: toxicity carried by many members is a network signal,
    # the same member carrying it all is not.
    spread = 1.0 - out["toxic_concentration"].fillna(1.0)
    out["hate_index"] = (
        out["toxic_share"].fillna(0.0) * 0.5
        + out["coded_share"].fillna(0.0) * 0.2
        + spread * 0.3
    )
    return out


def member_table(
    con: duckdb.DuckDBPyConnection,
    members: pd.DataFrame,
    platform: str = "x",
    topics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Member-level view: each clustered account's authenticity + activity, for
    drill-down from a scorecard."""
    from kma.authenticity import authenticity_score

    auth = authenticity_score(con, platform=platform)
    out = members.merge(
        auth, left_on="author_id", right_on="platform_user_id", how="left"
    )
    cols = [
        "cluster_id", "author_id", "handle", "account_age_days", "followers_count",
        "following_count", "n_posts", "duplicate_text_ratio", "suspicion", "anomaly_rank",
    ]
    if topics is not None:
        dom = (
            topics[topics["topic"] != -1]
            .groupby("author_handle")["topic"]
            .agg(lambda s: s.value_counts().idxmax())
            .rename("dominant_topic")
        )
        out = out.merge(dom, left_on="handle", right_index=True, how="left")
        cols.append("dominant_topic")
    return out[cols].sort_values(["cluster_id", "suspicion"], ascending=[True, False])


# --- persistence (07) -------------------------------------------------------


def persist_edges(
    con: duckdb.DuckDBPyConnection,
    edges: pd.DataFrame,
    channel: str,
    method: str,
    platform: str = "x",
) -> str:
    """Write one validated edge list as a Parquet run under the coordination/
    prefix. method in {svn_fdr, svn_bonf, pct, mc_fdr, mc_bonf}."""
    now = datetime.now(timezone.utc)
    cols = [c for c in ("src", "dst", "weight", "n_objects_shared", "n_coactions",
                        "min_gap", "weight_tfidf", "p_value") if c in edges.columns]
    buf = edges[cols].copy()
    buf["computed_at"] = now
    key = (
        f"coordination/platform={platform}/kind=edges/channel={channel}"
        f"/method={method}/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    )
    con.register("_coord_buf", buf)
    try:
        con.execute(
            f"COPY _coord_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_coord_buf")
    return key


def persist_clusters(
    con: duckdb.DuckDBPyConnection,
    members: pd.DataFrame,
    summary: pd.DataFrame,
    platform: str = "x",
) -> str:
    """Write cluster membership (one row per member, cluster stats repeated) as
    a Parquet run under the coordination/ prefix."""
    now = datetime.now(timezone.utc)
    buf = members.merge(
        summary[["cluster_id", "size", "channels", "n_channels", "internal_edge_share"]],
        on="cluster_id",
    )
    buf["computed_at"] = now
    key = (
        f"coordination/platform={platform}/kind=clusters"
        f"/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    )
    con.register("_coord_buf", buf)
    try:
        con.execute(
            f"COPY _coord_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_coord_buf")
    return key


# One row per (run, channel): run-level columns repeat on every row, the shape
# `persist_clusters` already uses for cluster stats.
COORD_METRIC_COLUMNS: list[str] = [
    "run_id", "computed_at", "platform", "code_version",
    "channels", "method", "resolution", "min_size",
    "lookback_days", "hub_cap_max",
    "n_clusters", "n_accounts", "n_corroborated_clusters", "n_corroborated_accounts",
    "channel", "timed", "min_repetition",
    "hub_cap", "hub_objects", "hub_max_degree",
    "objects_all", "objects_nohub", "traces_all", "traces_nohub",
    "accounts_all", "nohub_amp", "pairable",
    "edges_tested", "edges_fdr", "edges_bonferroni", "edges_percentile",
    "edges_kept", "accounts_tested", "accounts_fdr",
]

# Timed channels leave the hub fields NULL. Without an explicit nullable dtype
# pandas promotes those columns to float64, so successive runs write int64 and
# double under one glob and `union_by_name` reads become a dtype error later,
# far from the cause.
_COORD_METRIC_INTS: tuple[str, ...] = (
    "min_size", "lookback_days", "hub_cap_max",
    "n_clusters", "n_accounts", "n_corroborated_clusters", "n_corroborated_accounts",
    "min_repetition", "hub_cap", "hub_objects", "hub_max_degree",
    "objects_all", "objects_nohub", "traces_all", "traces_nohub",
    "accounts_all", "nohub_amp", "pairable",
    "edges_tested", "edges_fdr", "edges_bonferroni", "edges_percentile",
    "edges_kept", "accounts_tested", "accounts_fdr",
)


def persist_run_metrics(
    con: duckdb.DuckDBPyConnection,
    channel_stats: dict[str, dict],
    run_stats: dict,
    platform: str = "x",
) -> str:
    """Write one coordination pass's counters as a Parquet run under the
    coordination/ prefix (`kind=run_metrics`).

    This prefix exists because every number in docs/analysis/census-tuning.md
    was hand-computed prose, and the one that turned out to be wrong (Q2's
    "34.2% of censused objects above the hub cap") was wrong precisely because
    nothing recorded WHEN each measurement was taken: the window straddled the
    banding commit, so it averaged the old behaviour with the new. A persisted
    series is self-dating, and `code_version` makes "before or after commit X"
    answerable without archaeology."""
    now = datetime.now(timezone.utc)
    run_id = f"{now:%Y%m%dT%H%M%SZ}"
    base = {
        "run_id": run_id,
        "computed_at": now,
        "platform": platform,
        "code_version": os.getenv("GIT_SHA", ""),
        "lookback_days": COORD_LOOKBACK_DAYS,
        "hub_cap_max": HUB_CAP_MAX,
        **run_stats,
    }
    rows = [{**base, **cs} for cs in channel_stats.values()] or [base]
    buf = pd.DataFrame(rows).reindex(columns=COORD_METRIC_COLUMNS)
    for col in _COORD_METRIC_INTS:
        buf[col] = buf[col].astype("Int64")
    if buf["channels"].notna().any():
        buf["channels"] = buf["channels"].apply(
            lambda v: ",".join(v) if isinstance(v, (list, tuple)) else v
        )
    key = (
        f"coordination/platform={platform}/kind=run_metrics"
        f"/dt={now:%Y-%m-%d}/run={run_id}.parquet"
    )
    con.register("_coord_buf", buf)
    try:
        con.execute(
            f"COPY _coord_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_coord_buf")
    return key


# --- claim-scoped views (desk brief) ----------------------------------------


def story_account_set(
    story: pd.DataFrame,
    amplifiers: pd.DataFrame | None = None,
) -> set[str]:
    """Author ids for a story's members ∪ optional amplifiers (retweeters/repliers).

    `story` is candidate_stories rows for one story (needs author_id).
    `amplifiers` is spread()["amplifiers"] (platform_user_id) when available.
    """
    accounts: set[str] = set()
    if story is not None and not story.empty and "author_id" in story.columns:
        accounts.update(str(a) for a in story["author_id"].dropna().tolist())
    if amplifiers is not None and not amplifiers.empty:
        col = "platform_user_id" if "platform_user_id" in amplifiers.columns else "author_id"
        if col in amplifiers.columns:
            accounts.update(str(a) for a in amplifiers[col].dropna().tolist())
    return accounts


def claim_coordination(
    accounts: set[str] | list[str],
    edges: pd.DataFrame | None = None,
    clusters: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Filter validated edges / cluster membership to accounts tied to one claim.

    Empty account set or empty inputs -> empty frames, no crash. Language stays
    triage: summary counts only, no auto inauthentic label.

    Returns {"edges", "clusters", "summary"} where summary has n_accounts,
    n_edges, channels, n_clusters, cluster_ids.
    """
    empty_edges = pd.DataFrame(columns=["src", "dst", "weight", "channel"])
    empty_clusters = pd.DataFrame(columns=["author_id", "cluster_id"])
    acct = {str(a) for a in accounts}
    if not acct:
        return {
            "edges": empty_edges,
            "clusters": empty_clusters,
            "summary": {
                "n_accounts": 0,
                "n_edges": 0,
                "channels": [],
                "n_clusters": 0,
                "cluster_ids": [],
                "note": "triage slice only - coordination is probabilistic, not malice",
            },
        }

    if edges is None or edges.empty:
        claim_edges = empty_edges
    else:
        e = edges.copy()
        # accept src/dst or author_a/author_b style
        src = "src" if "src" in e.columns else "author_a"
        dst = "dst" if "dst" in e.columns else "author_b"
        mask = e[src].astype(str).isin(acct) & e[dst].astype(str).isin(acct)
        claim_edges = e.loc[mask].reset_index(drop=True)

    if clusters is None or clusters.empty:
        claim_clusters = empty_clusters
    else:
        c = clusters.copy()
        aid = "author_id" if "author_id" in c.columns else "platform_user_id"
        claim_clusters = c.loc[c[aid].astype(str).isin(acct)].reset_index(drop=True)

    channels: list[str] = []
    if not claim_edges.empty and "channel" in claim_edges.columns:
        channels = sorted(claim_edges["channel"].dropna().astype(str).unique().tolist())
    cluster_ids: list = []
    if not claim_clusters.empty and "cluster_id" in claim_clusters.columns:
        cluster_ids = sorted(claim_clusters["cluster_id"].dropna().unique().tolist())

    summary = {
        "n_accounts": len(acct),
        "n_edges": int(len(claim_edges)),
        "channels": channels,
        "n_clusters": len(cluster_ids),
        "cluster_ids": cluster_ids,
        "note": "triage slice only - coordination is probabilistic, not malice",
    }
    return {"edges": claim_edges, "clusters": claim_clusters, "summary": summary}


# --- evaluation (06) --------------------------------------------------------


def inject_synthetic(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    k: int = 20,
    n_seed_objects: int = 10,
    window: int = 60,
    platform: str = "x",
    seed: int = 0,
    model: str = MODEL,
    tau: float = 0.9,
) -> tuple[str, list[str]]:
    """Plant a known coordinated cluster: `k` synthetic accounts co-acting on
    `n_seed_objects` real objects within `window` seconds. Registers the
    augmented trace table (in-memory only, never persisted) and returns its
    name + the synthetic author ids, for evaluate_recovery."""
    rng = np.random.default_rng(seed)
    tr = traces(con, channel, platform, model, tau).df()
    objs = tr["action_object"].drop_duplicates()
    seeds = objs.sample(min(n_seed_objects, len(objs)), random_state=rng.integers(2**31))
    t0 = tr["created_at"].sample(len(seeds), random_state=rng.integers(2**31)).to_numpy()
    syn_ids = [f"synthetic_{seed}_{i:04d}" for i in range(k)]
    rows = [
        {
            "author_id": s,
            "action_object": o,
            "created_at": pd.Timestamp(base) + pd.Timedelta(seconds=float(rng.uniform(0, window))),
        }
        for s in syn_ids
        for o, base in zip(seeds, t0)
    ]
    aug = pd.concat([tr, pd.DataFrame(rows)], ignore_index=True)
    name = f"_tr_injected_{channel}"
    con.register(name, aug)
    return name, syn_ids


def evaluate_recovery(members: pd.DataFrame, synthetic_ids: list[str]) -> dict:
    """Precision/recall/F1 of the injected accounts in the detected clusters,
    plus the survey's weighted precision (penalises fragmenting the injected
    group across clusters)."""
    syn = set(synthetic_ids)
    if members.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "weighted_precision": 0.0,
                "best_cluster": None}
    by_cluster = members.groupby("cluster_id")["author_id"].agg(set)
    hits = by_cluster.apply(lambda s: len(s & syn))
    best = hits.idxmax()
    best_members = by_cluster[best]
    tp = len(best_members & syn)
    precision = tp / len(best_members)
    recall = tp / len(syn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    positive = hits[hits > 0]
    weighted_precision = (
        float(
            sum(h / len(by_cluster[c]) * h for c, h in positive.items()) / positive.sum()
        )
        if positive.sum()
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weighted_precision": weighted_precision,
        "best_cluster": best,
    }


def shuffled_traces(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    mode: str = "object",
    platform: str = "x",
    seed: int = 0,
    model: str = MODEL,
    tau: float = 0.9,
) -> str:
    """Null input for the false-positive control. mode="object" permutes
    action_object across rows (degree-preserving; the null for untimed SVN
    channels, where time-shuffling changes nothing). mode="time" permutes
    created_at within each account (the null for timed channels). Registers
    and returns the shuffled trace table name."""
    rng = np.random.default_rng(seed)
    tr = traces(con, channel, platform, model, tau).df()
    if mode == "object":
        tr["action_object"] = rng.permutation(tr["action_object"].to_numpy())
    elif mode == "time":
        tr["created_at"] = tr.groupby("author_id")["created_at"].transform(
            lambda s: s.sample(frac=1, random_state=rng.integers(2**31)).to_numpy()
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")
    name = f"_tr_shuffled_{channel}"
    con.register(name, tr)
    return name


def null_baseline(
    con: duckdb.DuckDBPyConnection,
    channel: str,
    platform: str = "x",
    delta: int | None = None,
    seed: int = 0,
    **params,
) -> pd.DataFrame:
    """Run the edge pipeline on real vs shuffled traces. The shuffled Bonferroni
    edge count must be ~0 - a non-empty result signals a bug or an inadequate
    null (06.2)."""
    mode = "time" if delta is not None else "object"
    real = validated_edges(con, channel, platform, delta=delta, **params)
    shuf = validated_edges(
        con, channel, platform, delta=delta,
        trace_table=shuffled_traces(con, channel, mode, platform, seed), **params,
    )
    return pd.DataFrame(
        [
            {"input": "real", "tested_pairs": len(real),
             "bonferroni": int(real["sig_bonferroni"].sum()),
             "fdr": int(real["sig_fdr"].sum())},
            {"input": f"shuffled({mode})", "tested_pairs": len(shuf),
             "bonferroni": int(shuf["sig_bonferroni"].sum()),
             "fdr": int(shuf["sig_fdr"].sum())},
        ]
    )


def internal_validation(
    con: duckdb.DuckDBPyConnection,
    members: pd.DataFrame,
    platform: str = "x",
    model: str = MODEL,
    n_perm: int = 1000,
    min_size: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Falsification test (06.3): detected clusters must beat random same-size
    account groups on Phase 1 suspicion and Phase 2 narrative homogeneity.
    Permutation p-values + effect sizes per cluster; if clusters are
    indistinguishable from random groups the detection is meaningless."""
    from kma.authenticity import authenticity_score

    rng = np.random.default_rng(seed)
    auth = authenticity_score(con, platform=platform).set_index("platform_user_id")
    lp = _member_posts(con, platform)
    emb = con.sql(
        f"""
        SELECT e.platform_post_id, p.author_id, e.embedding
        FROM (
            SELECT * FROM {embeddings_source(platform, _slug(model))}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY embedded_at DESC
            ) = 1
        ) e JOIN {lp} p USING (platform_post_id)
        """
    ).df()
    universe = auth.index.to_numpy()
    by_author = {a: g.index.to_numpy() for a, g in emb.groupby("author_id")}
    vecs = np.asarray(emb["embedding"].tolist(), dtype="float32")

    def homogeneity(ids, cap: int = 200) -> float:
        rows = np.concatenate([by_author.get(i, np.empty(0, dtype=int)) for i in ids]) \
            if ids else np.empty(0, dtype=int)
        if len(rows) < 2:
            return np.nan
        if len(rows) > cap:
            rows = rng.choice(rows, cap, replace=False)
        v = vecs[rows]
        return float((v @ v.T)[np.triu_indices(len(v), 1)].mean())

    rows = []
    for cid, grp in members.groupby("cluster_id"):
        ids = grp["author_id"].tolist()
        if len(ids) < min_size:
            continue
        obs_susp = auth["suspicion"].reindex(ids).mean()
        obs_hom = homogeneity(ids)
        null_susp = np.empty(n_perm)
        null_hom = np.empty(n_perm)
        for i in range(n_perm):
            sample = rng.choice(universe, len(ids), replace=False)
            null_susp[i] = auth["suspicion"].reindex(sample).mean()
            null_hom[i] = homogeneity(list(sample))
        p_susp = (1 + (null_susp >= obs_susp).sum()) / (1 + n_perm)
        ok = ~np.isnan(null_hom)
        p_hom = (
            (1 + (null_hom[ok] >= obs_hom).sum()) / (1 + ok.sum())
            if ok.any() and not np.isnan(obs_hom)
            else np.nan
        )
        rows.append(
            {
                "cluster_id": cid,
                "size": len(ids),
                "suspicion_obs": obs_susp,
                "suspicion_null_mean": null_susp.mean(),
                "suspicion_effect": (obs_susp - null_susp.mean())
                / max(null_susp.std(), 1e-9),
                "p_suspicion": p_susp,
                "homogeneity_obs": obs_hom,
                "homogeneity_null_mean": float(null_hom[ok].mean()) if ok.any() else np.nan,
                "homogeneity_effect": (obs_hom - null_hom[ok].mean())
                / max(null_hom[ok].std(), 1e-9)
                if ok.any() and not np.isnan(obs_hom)
                else np.nan,
                "p_homogeneity": p_hom,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id", ignore_index=True)
