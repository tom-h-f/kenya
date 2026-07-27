"""Automated hate/coded measurement helpers (no labels, no human queue).

Two orthogonal series for Kenya-election monitoring:

1. ``explicit_toxic`` - afro-xlmr offensive/hate (or ``hate_flag``), scoped to
   posts that are not clearly off-domain.
2. ``coded_suspect`` - NCIC/PeaceTech lexicon hit corroborated by incitement
   NLI, gated by each term's ``fp_risk``.

Domain and lexicon are derived from post text at read time (live
``scan_text``); NLI scores come from the existing ``incitement/`` prefix when
present. Nothing here is persisted to R2.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from kma.incitement import LEXICON, scan_text

_FP_RANK = {"low": 0, "medium": 1, "high": 2}

# Menace thresholds by min fp_risk of matched lexicon terms.
_CODED_MENACE_MIN = {"low": 0.45, "medium": 0.55, "high": 0.65}
_CODED_POL_MARGIN = {"low": -0.05, "medium": 0.0, "high": 0.10}

_KENYA_RE = re.compile(
    r"\b("
    r"kenya|kenyan|kenyans|"
    r"ruto|raila|gachagua|kasongo|kindiki|murkomen|matiangi|sifuna|"
    r"wantam|tutam|iebc|ncic|"
    r"nairobi|kisumu|mombasa|nakuru|eldoret|kakamega|bungoma|"
    r"luo|luos|kikuyu|kikuyus|kalenjin|kalenjins|luhya|luhyas|"
    r"kamba|kisii|meru|somali|gusii|abagusii|"
    r"mt\.?\s*kenya|murima|wamunyoro|"
    r"odm|uda|dcp|"
    r"githeri|mbogi"
    r")\b",
    re.IGNORECASE,
)

_OFFDOMAIN_RE = re.compile(
    r"\b("
    r"america|american|americans|"
    r"trump|biden|congress|whitehouse|white\s+house|"
    r"mamdani|zohran|remigrat\w*|"
    r"sweden|swedish|belgium|belgian|colorado|canada|canadian|"
    r"ndaa|zionican|"
    r"france|french|germany|german"
    r")\b",
    re.IGNORECASE,
)

_FP_RISK_BY_TERM: dict[str, str] = {
    term: meta["fp_risk"]
    for _cat, terms in LEXICON.items()
    for term, meta in terms.items()
}


def domain_bucket(text: str | None) -> str:
    """Return ``kenya``, ``offdomain``, or ``ambiguous`` from post text."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "ambiguous"
    s = str(text)
    kenya = bool(_KENYA_RE.search(s))
    off = bool(_OFFDOMAIN_RE.search(s))
    if off and not kenya:
        return "offdomain"
    if kenya:
        return "kenya"
    return "ambiguous"


def min_fp_risk(lexicon_hits: Iterable[str] | None) -> str | None:
    """Lowest ``fp_risk`` among hits, or None if empty/unknown."""
    if lexicon_hits is None:
        return None
    try:
        if isinstance(lexicon_hits, float) and pd.isna(lexicon_hits):
            return None
    except (TypeError, ValueError):
        pass
    ranks: list[tuple[int, str]] = []
    for term in lexicon_hits:
        risk = _FP_RISK_BY_TERM.get(str(term))
        if risk in _FP_RANK:
            ranks.append((_FP_RANK[risk], risk))
    if not ranks:
        return None
    return min(ranks)[1]


def _as_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coded_suspect(
    lexicon_hits: Iterable[str] | None,
    dehumanisation: float | None = None,
    violence_call: float | None = None,
    othering: float | None = None,
    political_criticism: float | None = None,
) -> bool:
    """True when a lexicon hit clears the fp_risk-gated NLI menace bar."""
    risk = min_fp_risk(lexicon_hits)
    if risk is None:
        return False

    scores = [
        _as_optional_float(dehumanisation),
        _as_optional_float(violence_call),
        _as_optional_float(othering),
    ]
    present = [s for s in scores if s is not None]
    if not present:
        # Without NLI, only low-fp terms are still blocked: plan requires NLI
        # for all tiers (high/medium without NLI cannot fire; low also needs NLI).
        return False

    menace = max(present)
    pol = _as_optional_float(political_criticism)
    if pol is None:
        pol = 0.0

    return menace >= _CODED_MENACE_MIN[risk] and menace >= pol + _CODED_POL_MARGIN[risk]


def attach_measurement_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain / live lexicon / coded / explicit measurement columns.

    Expects at least ``text``, ``label``, and ``hate_flag``. NLI columns are
    optional (``dehumanisation_score``, ``violence_call_score``,
    ``othering_score``, ``political_criticism_score``).
    """
    out = df.copy()
    texts = out["text"] if "text" in out.columns else pd.Series([None] * len(out))

    out["domain"] = texts.map(domain_bucket)
    out["in_kenya_scope"] = out["domain"] != "offdomain"

    scans = texts.map(lambda t: scan_text(t) if isinstance(t, str) else ([], []))
    out["lexicon_hits_live"] = [h for h, _ in scans]
    out["lexicon_categories_live"] = [c for _, c in scans]

    dehum = out.get("dehumanisation_score", pd.Series([None] * len(out)))
    viol = out.get("violence_call_score", pd.Series([None] * len(out)))
    oth = out.get("othering_score", pd.Series([None] * len(out)))
    pol = out.get("political_criticism_score", pd.Series([None] * len(out)))

    out["coded_suspect"] = [
        coded_suspect(hits, d, v, o, p)
        for hits, d, v, o, p in zip(
            out["lexicon_hits_live"], dehum, viol, oth, pol, strict=True
        )
    ]

    label = out["label"] if "label" in out.columns else pd.Series(["neither"] * len(out))
    hate_flag = (
        out["hate_flag"].fillna(False).astype(bool)
        if "hate_flag" in out.columns
        else pd.Series([False] * len(out))
    )
    out["explicit_toxic"] = ((label != "neither") | hate_flag) & out["in_kenya_scope"]
    # coded rate is also Kenya-scoped for prevalence denominators
    out["coded_suspect_kenya"] = out["coded_suspect"] & out["in_kenya_scope"]
    return out
