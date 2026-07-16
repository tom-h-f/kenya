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

import hashlib
import re
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from kma.coordination import (
    _burstiness_days,
    _ctfidf_top_terms,
    _filter_clustering_posts,
    _original_posts_sql,
)
from kma.db import (
    BUCKET,
    embeddings_source,
    engagements_source,
    posts_source,
)
from kma.semantic import MODEL, STOPWORDS, _slug

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
DEFAULT_MIN_SIZE = 3
# Thin high-gap lane: small clusters (below main min_size) with near-maximal
# corroboration gap + named entities. Triage as thin_evidence, never auto-elevated
# to high suspicion without amp/coordination signals.
THIN_MIN_SIZE = 2
THIN_MIN_GAP = 0.95
TIER_MAIN = "main"
TIER_THIN = "thin_evidence"

# Posts with fewer than this many content words (after stripping URLs and mentions)
# are dropped before clustering. Bare links, one-liners and pure-emoji posts carry
# near-identical low-information embeddings that bridge unrelated claims into one
# giant connected component (single-linkage chaining); removing them is what keeps a
# distinct claim from being swallowed by that blob.
MIN_CONTENT_WORDS = 5

# A connected component that is BOTH large and dispersed is a single-linkage chaining
# artifact (a hairball of unrelated claims), not one story - it is rejected. Both
# conditions are required so a small tight cluster or a genuinely viral cohesive story
# is never dropped. Measured: real claim clusters top out ~96 authors at cohesion
# ~0.75-0.99 (mean 0.91), while the pathological blob was 3,646 authors at 0.71.
MAX_COHERENT_AUTHORS = 150
MIN_COHESION = 0.80

# A trusted-outlet post counts as corroboration only if it shares at least this many
# salient terms with the story - and, when the story names entities, at least one of
# those (see _entity_terms). Embedding proximity alone is not enough. This makes the
# signal claim-level: a fabricated "Ruto motorcade crash" sits near generic accident
# coverage (~0.65) and shares generic words (accident, injured) but no entities
# (Ruto/Embu vs Nakuru/Eldoret), so that coverage no longer masks the gap.
MIN_SHARED_TERMS = 3

# Transparent triage weights over percentile-ranked components (mirror
# coordination.INAUTHENTICITY_WEIGHTS). The corroboration gap carries the most weight -
# it is the signal this layer adds - but a gap alone never flags a story; amplification,
# coordination and reach must stack with it. Not a verdict. burst_recency was dropped:
# over the recent window every claim cluster is time-tight, so it carried no signal;
# reach (how many distinct accounts carry the claim) replaced it.
STORY_WEIGHTS = {
    "corroboration_gap": 0.25,
    "amplifier_botness": 0.25,
    "coordination_overlap": 0.20,
    "reach": 0.20,
    "source_concentration": 0.10,
}

METRIC_GLOSSARY = {
    "story_id": "Run-local index of a near-duplicate content cluster (connected "
    "component of the cosine >= tau graph) over the recent window. Remapped each "
    "run (largest first). Prefer stable_story_id for persistence / deltas.",
    "stable_story_id": "Deterministic id = sha1 of sorted member platform_post_ids. "
    "Same member set => same id across runs.",
    "tier": f"'{TIER_MAIN}' = >= {DEFAULT_MIN_SIZE} authors (amp-weighted triage); "
    f"'{TIER_THIN}' = {THIN_MIN_SIZE}..{DEFAULT_MIN_SIZE - 1} authors with near-maximal "
    "corroboration gap and named entities. Thin is evidence, not high suspicion.",
    "high_suspicion": "True only when amp/coordination signals support elevation. "
    "Thin-tier stories are False unless amplifier_botness or coordination_overlap "
    "fires. Main-tier uses story_suspicion_index >= 0.5 as a soft cut.",
    "size": "Distinct authors posting the story - reach in accounts, not posts.",
    "n_posts": "Member posts in the story after dropping retweets and same-author "
    "duplicate text (>= size; paraphrases from the same account still count).",
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
    "corroboration_gap": "1 - corrob_sim, where corroboration is claim-level (a "
    "trusted post must share the story's vocabulary, not just its topic). High = the "
    "claim is spreading with no trusted-media match. A triage flag, never a verdict "
    "(outlets lag; see STORY_CAVEAT).",
    "amplifier_botness": "Mean Phase-1 authenticity suspicion of the story's authors "
    "(0-1). High = the accounts pushing it look bot-like.",
    "coordination_overlap": "Share of story authors that sit in a persisted "
    "Phase-3 coordination cluster. High = a known coordinated group is pushing it.",
    "reach": "Distinct authors carrying the claim. High = a wide push across many "
    "accounts rather than one or two voices.",
    "source_concentration": "Posts per distinct author. High = a handful of accounts "
    "generating the volume (manufactured), rather than broad organic spread.",
    "story_suspicion_index": "Transparent 0-1 triage score = weighted sum of the five "
    "percentile-ranked components (see STORY_WEIGHTS). NOT a verdict - the component "
    "breakdown, plus the nearest trusted post, is what a human acts on.",
}


def stable_story_id(member_post_ids: list | tuple) -> str:
    """Deterministic story id from the set of member platform_post_ids."""
    blob = "\n".join(sorted(str(p) for p in member_post_ids))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def attach_stable_story_ids(stories: pd.DataFrame) -> pd.DataFrame:
    """Add stable_story_id per story_id group (hash of sorted member post ids)."""
    if stories.empty:
        out = stories.copy()
        out["stable_story_id"] = pd.Series(dtype=str)
        return out
    out = stories.copy()
    sid_to_stable = {
        sid: stable_story_id(grp["platform_post_id"].tolist())
        for sid, grp in out.groupby("story_id")
    }
    out["stable_story_id"] = out["story_id"].map(sid_to_stable)
    return out


def _story_has_entities(stories: pd.DataFrame) -> dict[int, bool]:
    return {
        int(sid): bool(set().union(*(_entity_terms(t) for t in grp["text"])))
        for sid, grp in stories.groupby("story_id")
    }


def assign_tiers(stories: pd.DataFrame, scorecard: pd.DataFrame) -> pd.DataFrame:
    """Attach tier, stable_story_id, and high_suspicion to a scorecard.

    - main: size >= DEFAULT_MIN_SIZE
    - thin_evidence: THIN_MIN_SIZE <= size < DEFAULT_MIN_SIZE, gap >= THIN_MIN_GAP,
      and member text has at least one entity term
    - other small/low-gap rows get tier dropped (filtered out)

    Thin rows are never high_suspicion unless amp botness or coordination fires.
    """
    if scorecard.empty:
        out = scorecard.copy()
        for col in ("tier", "stable_story_id", "high_suspicion"):
            if col not in out.columns:
                out[col] = pd.Series(dtype=object if col != "high_suspicion" else bool)
        return out

    out = scorecard.copy()
    if "stable_story_id" not in out.columns or out["stable_story_id"].isna().all():
        if not stories.empty and "stable_story_id" in stories.columns:
            stab = (
                stories.groupby("story_id")["stable_story_id"]
                .first()
                .to_dict()
            )
        else:
            stab = {
                sid: stable_story_id(grp["platform_post_id"].tolist())
                for sid, grp in stories.groupby("story_id")
            } if not stories.empty else {}
        out["stable_story_id"] = out["story_id"].map(stab)

    has_ent = _story_has_entities(stories) if not stories.empty else {}
    tiers = []
    for _, row in out.iterrows():
        sid = int(row["story_id"])
        size = int(row["size"])
        gap = float(row.get("corroboration_gap", 0.0) or 0.0)
        if size >= DEFAULT_MIN_SIZE:
            tiers.append(TIER_MAIN)
        elif (
            size >= THIN_MIN_SIZE
            and gap >= THIN_MIN_GAP
            and has_ent.get(sid, False)
        ):
            tiers.append(TIER_THIN)
        else:
            tiers.append(None)
    out["tier"] = tiers
    out = out[out["tier"].notna()].copy()
    if out.empty:
        return out.reset_index(drop=True)

    amp = out["amplifier_botness"].fillna(0.0) >= 0.5
    coord = out["coordination_overlap"].fillna(0.0) > 0.0
    main_cut = out["story_suspicion_index"].fillna(0.0) >= 0.5
    out["high_suspicion"] = np.where(
        out["tier"] == TIER_THIN,
        amp | coord,
        main_cut,
    )
    return out.reset_index(drop=True)


def persist_story_columns(scorecard: pd.DataFrame) -> list[str]:
    """Column order written by persist_stories (also used in tests)."""
    cols = [
        "story_id", "stable_story_id", "tier", "high_suspicion",
        "size", "n_posts", "keywords", "hashtags", "representative_text",
        "representative_post_id", "member_post_ids", "corrob_sim", "corroboration_gap",
        "amplifier_botness", "coordination_overlap", "source_concentration",
        "story_suspicion_index",
    ]
    return [c for c in cols if c in scorecard.columns]


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


_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_STOP = frozenset(STOPWORDS)


def _content_words(text: object) -> int:
    """Count of real word tokens after dropping URLs, @mentions and the # symbol.

    Hashtag words still count (their meaning survives), but links and mentions do
    not - a post that is only a link or a couple of mentions has zero content."""
    if not isinstance(text, str):
        return 0
    t = _URL_RE.sub(" ", text)
    t = _MENTION_RE.sub(" ", t)
    return len(_WORD_RE.findall(t.replace("#", " ")))


def _salient_terms(text: object) -> set[str]:
    """Distinctive content words of a post (>= 3 chars, non-stopword, lower-cased).

    Used to gate corroboration on shared claim vocabulary: a trusted post about the
    SAME topic but a DIFFERENT claim (e.g. an unrelated road accident vs a fabricated
    presidential-motorcade crash) shares almost none of these terms."""
    if not isinstance(text, str):
        return set()
    t = _URL_RE.sub(" ", text)
    t = _MENTION_RE.sub(" ", t)
    return {
        w for w in (m.lower() for m in _WORD_RE.findall(t.replace("#", " ")))
        if len(w) >= 3 and w not in _STOP
    }


_ENTITY_RE = re.compile(r"[A-Z][A-Za-z]{2,}")


def _entity_terms(text: object) -> set[str]:
    """Proper-noun-ish tokens (capitalised / all-caps words, lower-cased) - a cheap
    named-entity proxy. Two accident stories share generic words (accident, injured)
    but different entities (Ruto/Embu vs matatu/Nakuru); shared entities are what
    separate the same claim from merely the same topic."""
    if not isinstance(text, str):
        return set()
    t = _URL_RE.sub(" ", text)
    t = _MENTION_RE.sub(" ", t)
    return {m.lower() for m in _ENTITY_RE.findall(t)} - _STOP


def _drop_low_information(
    df: pd.DataFrame, min_words: int = MIN_CONTENT_WORDS
) -> pd.DataFrame:
    """Drop low-information posts (bare links, one-liners, pure emoji) that chain
    unrelated claims together during single-linkage clustering. Stories only."""
    if df.empty:
        return df
    return df[df["text"].map(_content_words) >= min_words].copy()


def _component_cohesion(v: np.ndarray) -> float:
    """Mean cosine of members to their (renormalised) centroid: 1 = identical, low
    = a dispersed chain. Members are L2-normalised, so this is a plain dot product."""
    return float((v @ _renorm(v.mean(axis=0))).mean())


def _reject_chaining_blobs(
    x: np.ndarray,
    labels: np.ndarray,
    author_ids: np.ndarray,
    max_authors: int = MAX_COHERENT_AUTHORS,
    min_cohesion: float = MIN_COHESION,
) -> np.ndarray:
    """Boolean keep-mask dropping components that are both large (> max_authors
    distinct authors) and dispersed (cohesion < min_cohesion) - the single-linkage
    chaining blob. Both conditions required, so tight or small clusters are safe."""
    author_ids = np.asarray(author_ids)
    keep = np.ones(len(labels), dtype=bool)
    for comp in np.unique(labels):
        m = labels == comp
        if np.unique(author_ids[m]).size <= max_authors:
            continue
        if _component_cohesion(x[m]) < min_cohesion:
            keep &= ~m
    return keep


def candidate_stories(
    con: duckdb.DuckDBPyConnection,
    days: int = DEFAULT_DAYS,
    tau: float = DEFAULT_TAU,
    min_size: int = DEFAULT_MIN_SIZE,
    platform: str = "x",
    model: str = MODEL,
    include_thin: bool = False,
) -> pd.DataFrame:
    """Claim-level stories: connected components of the cosine >= tau graph over
    recent post embeddings (same primitive as coordination.content_clusters, but
    joined to latest_posts and filtered to the last `days`, at a lower story-level
    tau). Retweets, same-author duplicate text and low-information posts (bare
    links / one-liners) are dropped before clustering, and degenerate chaining
    blobs (large + dispersed components) are rejected after. Keeps components with
    >= min_size distinct authors (or >= THIN_MIN_SIZE when include_thin=True;
    tiering happens later in assign_tiers after corroboration).

    Returns one row per member post: story_id, stable_story_id, author_id,
    author_handle, text, created_at, is_repost, hashtags, conversation_id,
    embedding. story_id is a contiguous 0..k index; stable_story_id is durable."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import radius_neighbors_graph

    keep_min = THIN_MIN_SIZE if include_thin else min_size
    cols = [
        "platform_post_id", "story_id", "stable_story_id", "author_id", "author_handle",
        "text", "created_at", "is_repost", "hashtags", "conversation_id", "embedding",
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
    df = _drop_low_information(_filter_clustering_posts(df))
    if df.empty:
        return pd.DataFrame(columns=cols)
    x = np.asarray(df["embedding"].tolist(), dtype="float32")
    # embeddings are L2-normalised: cosine >= tau <=> euclidean <= sqrt(2 - 2 tau)
    g = radius_neighbors_graph(x, radius=float(np.sqrt(2 - 2 * tau)), mode="connectivity")
    _, labels = connected_components(g, directed=False)
    keep = _reject_chaining_blobs(x, labels, df["author_id"].to_numpy())
    if not keep.all():
        df = df[keep].copy()
        labels = labels[keep]
    df["_comp"] = labels
    # keep components with >= keep_min distinct authors
    author_counts = df.groupby("_comp")["author_id"].nunique()
    keep = author_counts[author_counts >= keep_min].index
    df = df[df["_comp"].isin(keep)].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    # relabel surviving components to a contiguous 0..k, largest first
    order = df["_comp"].value_counts().index.tolist()
    remap = {c: i for i, c in enumerate(order)}
    df["story_id"] = df["_comp"].map(remap)
    return attach_stable_story_ids(df)[cols].reset_index(drop=True)


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
          AND {_original_posts_sql("lp")}
        """
    ).df()


def corroboration(
    con: duckdb.DuckDBPyConnection,
    stories: pd.DataFrame,
    days: int = DEFAULT_DAYS,
    platform: str = "x",
    model: str = MODEL,
    centroids: dict[int, np.ndarray] | None = None,
    min_shared_terms: int = MIN_SHARED_TERMS,
) -> pd.DataFrame:
    """Per story: max cosine of its centroid to any trusted-source post that also
    shares the story's claim vocabulary, plus that nearest trusted post (for the
    human to judge the gap).

    The lexical + entity gate makes this claim-level, not topic-level: a trusted post
    corroborates a story only if it shares >= min_shared_terms salient words AND (when
    the story names entities) at least one entity. A fabricated claim that merely sits
    near real coverage of the same topic in embedding space (an unrelated accident,
    say) shares generic words but no entities, so it is correctly scored as
    uncorroborated (maximal gap).

    Returns story_id, corrob_sim, nearest_handle, nearest_text, nearest_post_id.
    No trusted post clears the gate (or none in range) -> corrob_sim 0.0 and null
    nearest fields (a maximal gap)."""
    cols = ["story_id", "corrob_sim", "nearest_handle", "nearest_text", "nearest_post_id"]
    if stories.empty:
        return pd.DataFrame(columns=cols)
    if centroids is None:
        centroids = story_centroids(stories)
    trusted = _trusted_posts(con, days, platform, model)
    rows = []
    if trusted.empty:
        for sid in centroids:
            rows.append({"story_id": sid, "corrob_sim": 0.0, "nearest_handle": None,
                         "nearest_text": None, "nearest_post_id": None})
        return pd.DataFrame(rows, columns=cols)
    tv = np.asarray(trusted["embedding"].tolist(), dtype="float32")
    trusted_terms = [_salient_terms(t) for t in trusted["text"]]
    trusted_ents = [_entity_terms(t) for t in trusted["text"]]
    story_terms, story_ents = {}, {}
    for sid, grp in stories.groupby("story_id"):
        story_terms[int(sid)] = set().union(*(_salient_terms(t) for t in grp["text"]))
        story_ents[int(sid)] = set().union(*(_entity_terms(t) for t in grp["text"]))
    for sid, c in centroids.items():
        terms, ents = story_terms.get(int(sid), set()), story_ents.get(int(sid), set())
        gate = np.array(
            [
                len(terms & tt) >= min_shared_terms
                # when the story names entities, the match must share one - this is what
                # rejects same-topic / different-claim coverage (see _entity_terms)
                and (not ents or len(ents & te) >= 1)
                for tt, te in zip(trusted_terms, trusted_ents)
            ],
            dtype=bool,
        )
        sims = np.where(gate, tv @ c, -np.inf)
        j = int(np.argmax(sims))
        if not np.isfinite(sims[j]):  # no trusted post shares the claim vocabulary
            rows.append({"story_id": sid, "corrob_sim": 0.0, "nearest_handle": None,
                         "nearest_text": None, "nearest_post_id": None})
            continue
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
    grouped = stories.groupby("story_id")["text"]
    sids = sorted(grouped.groups)
    docs = [" ".join(grouped.get_group(s).dropna().tolist()) for s in sids]
    terms = _ctfidf_top_terms(docs)
    return {int(s): t for s, t in zip(sids, terms)}


def _story_hashtags(stories: pd.DataFrame, top: int = 5) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for sid, grp in stories.groupby("story_id"):
        counts: dict[str, int] = {}
        for tags in grp["hashtags"]:
            if np.ndim(tags) == 0:
                continue
            for t in tags:
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
    if "stable_story_id" not in stories.columns:
        stories = attach_stable_story_ids(stories)
    centroids = story_centroids(stories)
    if corrob is None:
        corrob = corroboration(
            con, stories, days=days, platform=platform, model=model, centroids=centroids
        )

    auth = authenticity_score(con, platform=platform).set_index("platform_user_id")
    coord_ids = _coordination_author_ids(con, platform)
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
                "stable_story_id": grp["stable_story_id"].iloc[0],
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
        # more distinct accounts carrying the claim => wider push => higher rank
        "reach": rank(df["size"]),
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
    cols = persist_story_columns(keep)
    buf = keep[cols].copy()
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
