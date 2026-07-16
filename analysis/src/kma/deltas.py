"""Narrative / sentiment deltas sliced by region, language, and an EXPERIMENTAL
ethnic-community proxy.

TRIBE PROXY - READ THIS. Kenya has no ethnicity signal in the data. Swahili and
English are cross-ethnic lingua francas, so language does not indicate community.
The only rough proxy is self-declared profile `location` -> the community that
is historically dominant in that county. This is a COARSE, AGGREGATE-ONLY
heuristic that conflates geography with ethnicity, misses the large mixed/urban
and diaspora populations, and is wrong for any individual. It exists as a flagged
experiment only. Never present it as a headline metric or attach it to a person.

    from kma.db import connect
    from kma.deltas import slice_sentiment
    slice_sentiment(connect(), "region")
    slice_sentiment(connect(), "community")   # experimental
"""

from __future__ import annotations

import duckdb

from kma.db import authors_source, labels_source, posts_source

TRIBE_DISCLAIMER = (
    "EXPERIMENTAL location->community proxy: conflates geography with ethnicity, "
    "aggregate-only, wrong for individuals and for mixed/urban/diaspora users."
)

# First match wins; put more specific tokens earlier. Tokens are matched against
# lowercased free-text profile location.
REGION_RULES: list[tuple[list[str], str]] = [
    (["nairobi", "nbo", "cbd"], "Nairobi"),
    (["mombasa", "kilifi", "kwale", "lamu", "tana", "taita", "malindi", "coast", "pwani"], "Coast"),
    (["kisumu", "siaya", "homa bay", "homabay", "migori", "nyanza", "bondo", "kisii", "nyamira"], "Nyanza"),
    (["kakamega", "bungoma", "busia", "vihiga", "western", "mumias"], "Western"),
    (["eldoret", "uasin gishu", "kericho", "bomet", "nandi", "baringo", "marakwet", "nakuru", "narok", "kajiado", "rift"], "Rift Valley"),
    (["nyeri", "murang", "kiambu", "kirinyaga", "nyandarua", "thika", "central", "mount kenya", "mt kenya"], "Central"),
    (["machakos", "makueni", "kitui", "embu", "meru", "isiolo", "marsabit", "tharaka", "eastern"], "Eastern"),
    (["garissa", "wajir", "mandera", "north eastern", "northeastern"], "North Eastern"),
    (["turkana", "samburu", "west pokot", "lodwar"], "North Rift"),
]

# EXPERIMENTAL. See TRIBE_DISCLAIMER. County/place token -> dominant community.
COMMUNITY_RULES: list[tuple[list[str], str]] = [
    (["nyeri", "murang", "kiambu", "kirinyaga", "nyandarua", "thika"], "Kikuyu"),
    (["kisumu", "siaya", "homa bay", "homabay", "migori", "bondo"], "Luo"),
    (["kakamega", "bungoma", "busia", "vihiga", "mumias"], "Luhya"),
    (["eldoret", "uasin gishu", "kericho", "bomet", "nandi", "baringo", "marakwet"], "Kalenjin"),
    (["machakos", "makueni", "kitui"], "Kamba"),
    (["kisii", "nyamira"], "Kisii"),
    (["meru", "tharaka", "embu"], "Meru/Embu"),
    (["mombasa", "kilifi", "kwale", "lamu", "tana", "taita", "malindi"], "Coastal"),
    (["garissa", "wajir", "mandera"], "Somali"),
    (["narok", "kajiado"], "Maasai"),
]


def _case(col: str, rules: list[tuple[list[str], str]]) -> str:
    whens = []
    for toks, label in rules:
        conds = " OR ".join(f"lower({col}) LIKE '%{tok}%'" for tok in toks)
        whens.append(f"WHEN {conds} THEN '{label}'")
    return "CASE " + " ".join(whens) + " ELSE NULL END"


def region_case(col: str = "a.location") -> str:
    return _case(col, REGION_RULES)


def community_case(col: str = "a.location") -> str:
    return _case(col, COMMUNITY_RULES)


_SENTIMENT_NUM = (
    "CASE l.sentiment WHEN 'positive' THEN 1 WHEN 'negative' THEN -1 ELSE 0 END"
)


def slice_sentiment(
    con: duckdb.DuckDBPyConnection,
    dimension: str = "region",
    platform: str = "x",
    min_posts: int = 10,
):
    """Sentiment aggregated over a slice. `dimension` in {region, lang, community}.
    `community` is the EXPERIMENTAL ethnic proxy (see TRIBE_DISCLAIMER)."""
    dim = {
        "region": region_case(),
        "community": community_case(),
        "lang": "p.lang",
    }[dimension]
    return con.sql(
        f"""
        WITH p AS (
            SELECT * FROM {posts_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        ), a AS (
            SELECT * FROM {authors_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_user_id ORDER BY collected_at DESC
            ) = 1
        ), l AS (
            SELECT * FROM {labels_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY labeled_at DESC
            ) = 1
        )
        SELECT {dim} AS slice,
               count(*) AS posts,
               round(avg({_SENTIMENT_NUM}), 3) AS mean_sentiment,
               round(avg(({_SENTIMENT_NUM} = -1)::INT), 3) AS neg_share,
               round(avg(({_SENTIMENT_NUM} = 1)::INT), 3) AS pos_share
        FROM p
        JOIN a ON p.author_id = a.platform_user_id
        JOIN l ON p.platform_post_id = l.platform_post_id
        GROUP BY 1
        HAVING slice IS NOT NULL AND count(*) >= {min_posts}
        ORDER BY posts DESC
        """
    )


def map_location(location: object, dimension: str = "region") -> str | None:
    """Map a free-text profile location to region or community label (or None)."""
    if not isinstance(location, str) or not location.strip():
        return None
    loc = location.lower()
    rules = REGION_RULES if dimension == "region" else COMMUNITY_RULES
    for toks, label in rules:
        if any(tok in loc for tok in toks):
            return label
    return None


MIN_LOCATION_COVERAGE = 0.20


def slice_claim(
    authors: "pd.DataFrame",
    dimension: str = "region",
    min_coverage: float = MIN_LOCATION_COVERAGE,
) -> "pd.DataFrame":
    """Aggregate volume (+ optional sentiment) for a claim's author set.

    `authors` columns: author_id or platform_user_id, location; optional sentiment
    (-1/0/1 or label). Returns aggregate rows only (never per-author community).

    Always includes columns: slice, n_authors, coverage_pct,
    insufficient_location_signal, disclaimer.
    """
    import pandas as pd

    if dimension not in ("region", "community"):
        raise ValueError("dimension must be 'region' or 'community'")

    empty_cols = [
        "slice", "n_authors", "mean_sentiment", "coverage_pct",
        "insufficient_location_signal", "disclaimer",
    ]
    disclaimer = TRIBE_DISCLAIMER if dimension == "community" else (
        "Region inferred from free-text profile location; many authors unmapped."
    )
    if authors is None or authors.empty:
        return pd.DataFrame(columns=empty_cols)

    df = authors.copy()
    id_col = "author_id" if "author_id" in df.columns else "platform_user_id"
    if id_col not in df.columns:
        raise ValueError("authors needs author_id or platform_user_id")
    if "location" not in df.columns:
        df["location"] = None

    df = df.drop_duplicates(subset=[id_col])
    df["_slice"] = df["location"].map(lambda loc: map_location(loc, dimension))
    n_total = len(df)
    n_mapped = int(df["_slice"].notna().sum())
    coverage = n_mapped / n_total if n_total else 0.0
    insufficient = coverage < min_coverage

    if n_mapped == 0:
        return pd.DataFrame(
            [
                {
                    "slice": None,
                    "n_authors": 0,
                    "mean_sentiment": None,
                    "coverage_pct": round(coverage * 100, 1),
                    "insufficient_location_signal": True,
                    "disclaimer": disclaimer,
                }
            ],
            columns=empty_cols,
        )

    sent_col = None
    if "sentiment" in df.columns:
        if df["sentiment"].dtype == object:
            sent_col = df["sentiment"].map(
                {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            )
        else:
            sent_col = pd.to_numeric(df["sentiment"], errors="coerce")

    rows = []
    for label, grp in df[df["_slice"].notna()].groupby("_slice"):
        mean_sent = None
        if sent_col is not None:
            mean_sent = round(float(sent_col.loc[grp.index].mean()), 3)
        rows.append(
            {
                "slice": label,
                "n_authors": len(grp),
                "mean_sentiment": mean_sent,
                "coverage_pct": round(coverage * 100, 1),
                "insufficient_location_signal": insufficient,
                "disclaimer": disclaimer,
            }
        )
    out = pd.DataFrame(rows, columns=empty_cols).sort_values(
        "n_authors", ascending=False, ignore_index=True
    )
    return out
