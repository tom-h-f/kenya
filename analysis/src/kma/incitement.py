"""Ethnic-incitement triage: Kenya-specific coded-term lexicon + zero-shot NLI.

Two independent signals, both triage-only:

1. `lexicon_scan` - regex hits on documented coded terms (NCIC advisories and
   the PeaceTech Lab Kenya lexicon are the sources for the core entries; each
   entry records its category and false-positive risk). A hit means "a human
   should read this", never "this is hate speech" - most of these words have
   innocent everyday senses.
2. `score_new` - zero-shot NLI scores (dehumanisation / violence call /
   othering-expulsion, plus an ordinary-political-criticism contrast class),
   persisted incrementally to the R2 `incitement/` prefix. NOT `labels/`:
   latest_labels dedups on platform_post_id alone, so a second writer there
   would shadow the sentiment/emotion rows.

Known limits: the NLI model is multilingual (EN/SW) but not Sheng-aware, and
the platform lang field is unreliable for Swahili; lexicon recall for Sheng
obfuscations is limited to the listed variants. Fine-tuning on Kenyan
hate-speech corpora is a documented future option, not part of this pass.

    from kma.db import connect
    from kma.incitement import lexicon_scan, score_new
    con = connect()
    hits = lexicon_scan(con)
    score_new(con, limit=500)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import duckdb
import pandas as pd
import pyarrow as pa

from kma.classify import STANCE_MODEL, _pipe, _run
from kma.db import BUCKET, incitement_source, posts_source

# category -> lexicon entries. `pattern` is a case-insensitive regex fragment;
# `fp_risk` marks terms with common innocent senses (mende = cockroach the
# insect, nyoka = snake) whose hits are only meaningful with NLI corroboration.
LEXICON: dict[str, dict[str, dict]] = {
    "dehumanisation": {
        "madoadoa": {"pattern": r"madoa\s?doa", "lang": "sw", "fp_risk": "low",
                     "notes": "'spots/stains' - marking a community for removal; NCIC-flagged, 2007-08 era"},
        "kwekwe": {"pattern": r"\bkwekwe\b", "lang": "sw", "fp_risk": "medium",
                   "notes": "'weeds' (to be uprooted); NCIC-flagged"},
        "sangari": {"pattern": r"\bsangari\b", "lang": "kln", "fp_risk": "low",
                    "notes": "'couch grass to uproot'; 2007-08 era"},
        "madimoni": {"pattern": r"\bmadimoni\b", "lang": "sw", "fp_risk": "medium",
                     "notes": "'demons'"},
        "mende": {"pattern": r"\bmende\b", "lang": "sw", "fp_risk": "high",
                  "notes": "'cockroaches' - everyday word; needs NLI corroboration"},
        "nyoka": {"pattern": r"\bnyoka\b", "lang": "sw", "fp_risk": "high",
                  "notes": "'snakes' - everyday word; needs NLI corroboration"},
        "fumigate": {"pattern": r"\bfumigat\w+|\bfukiza", "lang": "en/sw", "fp_risk": "medium",
                     "notes": "pest-control framing of people; 2017 era"},
    },
    "expulsion": {
        "rudi_kwao": {"pattern": r"\b(wa)?rudi\w*\s+kwao\b", "lang": "sw", "fp_risk": "medium",
                      "notes": "'(they should) go back to their homeland'"},
        "fukuza": {"pattern": r"\b(tu|wa)?fukuz\w+", "lang": "sw", "fp_risk": "high",
                   "notes": "'chase away/expel' - generic verb; context decides target"},
        "waondoke": {"pattern": r"\bwaondoke\b", "lang": "sw", "fp_risk": "medium",
                     "notes": "'they must leave'"},
    },
    "veiled_threat": {
        "watajua_hawajui": {"pattern": r"watajua\s+hawajui", "lang": "sw", "fp_risk": "low",
                            "notes": "'they will know they don't know' - NCIC-flagged veiled threat"},
        "kimeumana": {"pattern": r"\bkimeumana\b", "lang": "sheng", "fp_risk": "medium",
                      "notes": "'it has bitten / things have escalated' - 2017-era coded"},
    },
    "othering": {
        "watu_wa_kule": {"pattern": r"watu\s+wa\s+kule", "lang": "sw", "fp_risk": "medium",
                         "notes": "'those people from there'"},
        "wabara": {"pattern": r"\bwabara\b|watu\s+wa\s+bara", "lang": "sw", "fp_risk": "medium",
                   "notes": "coastal term for upcountry people; MRC-era othering"},
        "41_vs_1": {"pattern": r"\b41\s*(vs\.?|versus|against)\s*1\b", "lang": "en", "fp_risk": "low",
                    "notes": "2007-era anti-Kikuyu coalition framing"},
        "uthamaki": {"pattern": r"\buthamaki\b", "lang": "kik", "fp_risk": "medium",
                     "notes": "Kikuyu kingship/supremacy discourse marker (context term)"},
    },
}

HYPOTHESES: dict[str, str] = {
    "dehumanisation": "a statement that dehumanises an ethnic group or community",
    "violence_call": "a statement calling for or approving violence against a group of people",
    "othering": "a statement that a community does not belong in a place and should leave",
    "political_criticism": "ordinary political criticism that does not target an ethnic community",
}

MODEL = STANCE_MODEL


def _compiled() -> list[tuple[str, str, re.Pattern]]:
    return [
        (category, term, re.compile(entry["pattern"], re.IGNORECASE))
        for category, terms in LEXICON.items()
        for term, entry in terms.items()
    ]


def scan_text(text: str) -> tuple[list[str], list[str]]:
    """(matched terms, matched categories) for one text."""
    hits, cats = [], []
    for category, term, rx in _compiled():
        if rx.search(text):
            hits.append(term)
            if category not in cats:
                cats.append(category)
    return hits, cats


def lexicon_scan(
    con: duckdb.DuckDBPyConnection, platform: str = "x"
) -> pd.DataFrame:
    """Regex pass over all latest posts. Returns one row per post with >= 1 hit:
    platform_post_id, author_handle, text, lexicon_hits, lexicon_categories."""
    df = con.sql(
        f"""
        SELECT platform_post_id, author_handle, text FROM (
            SELECT * FROM {posts_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        WHERE text IS NOT NULL
        """
    ).df()
    scans = df["text"].map(scan_text)
    df["lexicon_hits"] = [h for h, _ in scans]
    df["lexicon_categories"] = [c for _, c in scans]
    return df[df["lexicon_hits"].str.len() > 0].reset_index(drop=True)


def _scored_ids(con: duckdb.DuckDBPyConnection, platform: str) -> set[str]:
    try:
        rel = con.sql(
            f"SELECT DISTINCT platform_post_id FROM {incitement_source(platform)}"
        )
    except duckdb.Error:
        return set()
    return set(rel.df()["platform_post_id"].tolist())


def _pending(
    con: duckdb.DuckDBPyConnection, platform: str, limit: int | None
) -> pd.DataFrame:
    """Unscored posts, lexicon hits first (the NLI pass is hours for the full
    corpus; priority order lets a bounded run cover the triage-relevant tail)."""
    df = con.sql(
        f"""
        SELECT platform_post_id, text FROM (
            SELECT * FROM {posts_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        WHERE text IS NOT NULL AND length(trim(text)) > 0
        """
    ).df()
    df = df[~df["platform_post_id"].isin(_scored_ids(con, platform))]
    if df.empty:
        return df.assign(lexicon_hits=None, lexicon_categories=None)
    scans = df["text"].map(scan_text)
    df["lexicon_hits"] = [h for h, _ in scans]
    df["lexicon_categories"] = [c for _, c in scans]
    df["_prio"] = df["lexicon_hits"].map(len) > 0
    df = df.sort_values("_prio", ascending=False).drop(columns="_prio")
    return df.head(limit) if limit else df


def score_new(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    limit: int | None = None,
    batch_size: int = 16,
) -> int:
    """Zero-shot incitement scores for unscored posts (lexicon hits first);
    persist one Parquet run to R2 `incitement/`. Returns the number scored."""
    df = _pending(con, platform, limit)
    if df.empty:
        return 0
    res = _run(
        _pipe("zero-shot-classification", MODEL),
        df["text"].tolist(),
        batch_size=batch_size,
        candidate_labels=list(HYPOTHESES.values()),
        multi_label=True,
    )
    if isinstance(res, dict):
        res = [res]
    by_label = []
    for r in res:
        scores = dict(zip(r["labels"], r["scores"]))
        by_label.append(
            {name: float(scores[h]) for name, h in HYPOTHESES.items()}
        )
    now = datetime.now(timezone.utc)
    table = pa.table(
        {
            "platform_post_id": df["platform_post_id"].tolist(),
            "lexicon_hits": df["lexicon_hits"].tolist(),
            "lexicon_categories": df["lexicon_categories"].tolist(),
            "dehumanisation_score": [b["dehumanisation"] for b in by_label],
            "violence_call_score": [b["violence_call"] for b in by_label],
            "othering_score": [b["othering"] for b in by_label],
            "political_criticism_score": [
                b["political_criticism"] for b in by_label
            ],
            "model": [MODEL] * len(df),
            "scored_at": [now] * len(df),
        }
    )
    key = (
        f"incitement/platform={platform}"
        f"/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    )
    con.register("_inc_buf", table)
    try:
        con.execute(
            f"COPY _inc_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_inc_buf")
    return len(df)
