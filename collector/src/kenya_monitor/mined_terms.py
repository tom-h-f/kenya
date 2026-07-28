"""Candidate coded terms mined from hate-seed cohorts, and the gate on them.

Mined terms would drive searches about ethnic hate. Promoting a machine-derived
term automatically risks two distinct harms: searching on a word the model
merely correlated with a toxic cohort can implicate people who use it
innocently, and feeding it back into collection biases the corpus toward
whatever the miner already believed. So the default is write-and-log:
candidates land here, `monitor hate-mine --dry-run` shows each with its counts
and example posts, and a human moves the survivors into `hate_terms.yaml`.

Reuses the `DynamicEntry` shape and expiry semantics from `adaptive.py` so
promoted terms age out the same way as every other dynamic target.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kenya_monitor.config import (
    DYNAMIC_EXPIRY_DAYS,
    HATE_MINE_MAX_TERMS,
    MINED_TERMS_PATH,
)

log = logging.getLogger("kenya_monitor")

SOURCE = "hate-mined"


@dataclass
class MinedTerm:
    term: str
    novelty: float = 0.0
    lift: float = 0.0
    n_cohort: int = 0
    n_authors: int = 0
    n_corpus: int = 0
    examples: list[str] = field(default_factory=list)
    source: str = SOURCE
    added_at: str = ""
    last_confirmed: str = ""
    approved: bool = False  # flipped only by a human, or by autopromote


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path = MINED_TERMS_PATH) -> list[MinedTerm]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [MinedTerm(**e) for e in raw.get("terms", [])]


def save(terms: list[MinedTerm], path: Path = MINED_TERMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"updated_at": _now_iso(), "terms": [asdict(t) for t in terms]}, indent=2)
    )


def merge(
    existing: list[MinedTerm],
    candidates: list[dict],
    examples: dict[str, list[str]] | None = None,
    expiry_days: int = DYNAMIC_EXPIRY_DAYS,
    now: datetime | None = None,
) -> list[MinedTerm]:
    """Refresh confirmed candidates, add new ones, drop stale ones.

    A term's `approved` flag survives re-mining - a human decision is not
    undone by the next pass - and expiry is by last confirmation, so a term
    that stops appearing in the cohort ages out on its own.
    """
    now = now or datetime.now(timezone.utc)
    now_iso = now.isoformat()
    cutoff = now - timedelta(days=expiry_days)
    examples = examples or {}
    by_term = {t.term: t for t in existing}

    for c in candidates:
        term = c["term"]
        prior = by_term.get(term)
        entry = MinedTerm(
            term=term,
            novelty=float(c.get("novelty") or 0.0),
            lift=float(c.get("lift") or 0.0),
            n_cohort=int(c.get("c_in") or 0),
            n_authors=int(c.get("a_in") or 0),
            n_corpus=int(c.get("c_all") or 0),
            examples=examples.get(term, prior.examples if prior else []),
            added_at=prior.added_at if prior else now_iso,
            last_confirmed=now_iso,
            approved=bool(prior.approved) if prior else False,
        )
        by_term[term] = entry

    alive = [
        t for t in by_term.values()
        if not t.last_confirmed or datetime.fromisoformat(t.last_confirmed) > cutoff
    ]
    return sorted(alive, key=lambda t: (not t.approved, -t.novelty))


def active(
    terms: list[MinedTerm], autopromote: bool = False, cap: int = HATE_MINE_MAX_TERMS
) -> list[dict]:
    """Terms eligible to be searched this pass.

    Without `autopromote`, only human-approved terms qualify. Everything
    returned is handed to `hate_seek` as `fp_risk: high`, so it always gets the
    widest Kenya anchoring available.
    """
    usable = [t for t in terms if t.approved or autopromote]
    return [{"term": t.term} for t in usable[:cap]]
