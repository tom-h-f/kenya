"""Hate-seeking search: term selection, query rendering, rotation state.

The hate-speech classifier is a **locator, not a label**. It detects explicit
toxicity well and the 2026 coded register badly (on known-coded posts its mean
`p_hate` actually drops). So term search is not asked to recognise coded
speech - it only has to find a foothold account, after which the account-scoped
passes (timelines, retweeter census, reply threads, follow crawl) pull that
account at full recall with no operator and no model.

Three term sources feed one rotated budget:

  lexicon         config/hate_terms.yaml, mirroring kma.incitement.LEXICON
  targets-of-hate ethnonym x menace conjunctions, never a bare ethnonym
  mined           machine-derived candidates, human-gated by default

Everything here is pure: `render` and `select_terms` do no I/O, so the whole
query surface is testable offline (`monitor hate-seek --dry-run`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kenya_monitor.collectors.x import build_query
from kenya_monitor.config import (
    HATE_MINE_MAX_TERMS,
    HATE_SEEK_MAX_TERMS,
    HATE_SEEK_STATE_PATH,
    HATE_TARGET_MAX_TERMS,
    HateTermSet,
)

log = logging.getLogger("kenya_monitor")

LEXICON = "lexicon"
TARGETS_OF_HATE = "targets-of-hate"
MINED = "mined"

# Which R2 `posts/type=` partition each source writes to. Both are registered in
# kma.db.TARGETED_TYPES and excluded from every prevalence measurement.
PARTITIONS = {
    LEXICON: "hate_search",
    MINED: "hate_search",
    TARGETS_OF_HATE: "hate_target_search",
}

# Anchor tier per false-positive risk. `low` terms are near-unique phrases where
# anchoring would cost the recall that matters most; `high` terms are everyday
# Swahili words that without scoping return a region-wide noise firehose.
ANCHOR_TIER = {"low": None, "medium": "core", "high": "wide"}


@dataclass(frozen=True)
class HateQuery:
    term: str
    query: str
    fp_risk: str
    source: str

    @property
    def target_type(self) -> str:
        return PARTITIONS[self.source]


def anchors_for(q: HateQuery, anchors: dict[str, list[str]]) -> list[str] | None:
    tier = ANCHOR_TIER.get(q.fp_risk)
    return anchors.get(tier) if tier else None


def render(
    q: HateQuery,
    anchors: dict[str, list[str]],
    since: str | None = None,
    until: str | None = None,
    min_faves: int | None = None,
    include_retweets: bool = True,
) -> str:
    """The full X query string for one term. Pure - the unit under test."""
    return build_query(
        q.query,
        min_faves=min_faves,
        since=since,
        until=until,
        include_retweets=include_retweets,
        anchors=anchors_for(q, anchors),
    )


def lexicon_queries(terms: HateTermSet) -> list[HateQuery]:
    return [
        HateQuery(term=t.term, query=t.query, fp_risk=t.fp_risk, source=LEXICON)
        for t in terms.terms
    ]


def target_queries(terms: HateTermSet) -> list[HateQuery]:
    """Ethnonym x menace conjunctions. A bare ethnonym is ordinary discussion;
    the conjunction is what makes the query about threat rather than about a
    community. Always `high` risk, so always widely anchored."""
    if not terms.ethnonyms or not terms.menace:
        return []
    menace = " OR ".join(f'"{m}"' if " " in m else m for m in terms.menace)
    return [
        HateQuery(
            term=f"target:{e.lower()}",
            query=f"{e} ({menace})",
            fp_risk="high",
            source=TARGETS_OF_HATE,
        )
        for e in terms.ethnonyms
    ]


def mined_queries(mined: list[dict] | None) -> list[HateQuery]:
    """Mined terms always carry `fp_risk: high` - a machine-derived term has no
    provenance, so it gets the widest Kenya anchoring available."""
    return [
        HateQuery(
            term=m["term"], query=m.get("query") or m["term"], fp_risk="high", source=MINED
        )
        for m in (mined or [])
    ]


def load_state(path: Path = HATE_SEEK_STATE_PATH) -> dict[str, str]:
    """term -> ISO timestamp of the last pass that searched it."""
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text()).get("last_searched", {}))


def save_state(state: dict[str, str], path: Path = HATE_SEEK_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"updated_at": datetime.now(timezone.utc).isoformat(), "last_searched": state},
            indent=2,
        )
    )


def _rotate(pool: list[HateQuery], state: dict[str, str], cap: int) -> list[HateQuery]:
    """Least-recently-searched first, so the whole register is covered across
    passes without raising the per-pass cost."""
    if cap <= 0:
        return []
    return sorted(pool, key=lambda q: state.get(q.term, ""))[:cap]


def select_terms(
    terms: HateTermSet,
    mined: list[dict] | None = None,
    state: dict[str, str] | None = None,
    max_terms: int = HATE_SEEK_MAX_TERMS,
    max_targets: int = HATE_TARGET_MAX_TERMS,
    max_mined: int = HATE_MINE_MAX_TERMS,
) -> list[HateQuery]:
    """One pass worth of queries: rotated lexicon terms, plus reserved slots for
    ethnonym and mined queries. `max_terms` is the total budget."""
    state = state or {}
    picked = _rotate(target_queries(terms), state, min(max_targets, max_terms))
    remaining = max_terms - len(picked)
    picked += _rotate(mined_queries(mined), state, min(max_mined, remaining))
    remaining = max_terms - len(picked)
    picked += _rotate(lexicon_queries(terms), state, remaining)
    return picked


def mark_searched(
    queries: list[HateQuery], state: dict[str, str] | None = None
) -> dict[str, str]:
    now = datetime.now(timezone.utc).isoformat()
    out = dict(state or {})
    for q in queries:
        out[q.term] = now
    return out
