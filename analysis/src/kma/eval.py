"""Ground-truth eval: known disinfo cases the story pipeline SHOULD surface.

Encodes real fabricated / distorted stories that circulated on X as labeled cases,
then walks the pipeline stage-by-stage (presence -> embedding -> clustering ->
scoring) and reports where each case drops out. This is the regression instrument
for the story-detection methodology: a good fix turns these green; a change that
hurts recall turns them red.

    from kma.db import connect
    from kma import eval as gt
    con = connect()
    report = gt.evaluate(con, days=14)   # one DataFrame, one row per case

Cases anchor by claim text patterns (robust to re-collection), not post ids. The
drop-out stage is the first pipeline step that loses the case:
  collection  -> not in the corpus at all
  embedding   -> collected but not embedded
  clustering  -> embedded but no qualifying cluster (sub-threshold OR swallowed by
                 a degenerate blob component)
  scoring     -> forms a coherent cluster but ranks below the triage cut
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from kma import semantic
from kma import stories as st
from kma.db import latest_posts

# A story whose author count exceeds this is treated as a degenerate chaining blob
# rather than a single claim (real claim clusters seen in the data are well under
# 100 authors; the pathological component was 6,336). Tunable as data grows.
BLOB_AUTHORS = 200

# A "surface" case passes if its cluster ranks within the top TOP_FRACTION of the
# scorecard - a realistic human-triage cut, not an absolute top-N (story counts swing
# with min_size). Floored at TOP_MIN so tiny runs stay sane.
TOP_FRACTION = 0.15
TOP_MIN = 20

# expect values: "surface" (must reach the triage cut) or "known-limitation" (tracked
# but not required to pass - see the motorcade case for why).
EXPECT_SURFACE = "surface"
EXPECT_KNOWN_LIMITATION = "known-limitation"


@dataclass
class GroundTruthCase:
    name: str
    claim: str  # canonical claim; used as the semantic-search query
    text_like: list[str]  # ILIKE patterns (lower-cased) matching the claim text
    anchor_handles: list[str]  # confirmed authors of the fabricated claim
    approx_date: str
    verdict: str = "disinfo"
    expect: str = EXPECT_SURFACE
    note: str = ""
    tags: list[str] = field(default_factory=list)


GROUND_TRUTH: list[GroundTruthCase] = [
    GroundTruthCase(
        name="sacco-savings-borrow",
        claim="government to borrow tap SACCO member savings for national infrastructure fund",
        text_like=["%sacco%savings%finance%", "%tap%sacco%trillion%",
                   "%borrow%more than%sacco%", "%sh1 trillion%sacco%"],
        anchor_handles=["pocketpowerr", "MutembeiTV", "money254HQ", "MwagoIsaac"],
        approx_date="2026-07-05",
        expect=EXPECT_SURFACE,
        note="Distorted claim amplified by ~19 accounts. A trusted outlet (ntvkenya) "
             "covered the topic, so its corroboration gap is legitimately low - it must "
             "be flagged via reach / botness / coordination, not the gap. This is the "
             "case the reach signal + reweighting were tuned to surface.",
        tags=["positive", "has-coverage", "amplified"],
    ),
    GroundTruthCase(
        name="ruto-motorcade-crash",
        claim="President Ruto speeding motorcade crash accident injured Embu Meru highway",
        text_like=["%motorcade%collid%", "%ruto%motorcade%accident%",
                   "%speeding%security%accident%", "%ruto%speeding%motorcade%"],
        anchor_handles=["ouma_neko", "Benson_Mwiti_25"],
        approx_date="2026-07-05",
        expect=EXPECT_KNOWN_LIMITATION,
        note="Fabricated crash, no reputable coverage - but only ~2 accounts carried it. "
             "KNOWN LIMITATION: below min_size, and even at min_size=2 with a maximal gap "
             "a 2-account claim has no reach/botness/coordination signal to lift it in the "
             "amplification scorecard. Its topic also overlaps real accident coverage and "
             "its lone entity (Ruto) is ubiquitous, so entity-gated corroboration only "
             "partially isolates it. Catching this class needs a dedicated small-cluster "
             "high-gap view (see docs/analysis/phase-4-stories.md).",
        tags=["positive", "no-coverage", "small"],
    ),
]


def resolve_posts(
    con: duckdb.DuckDBPyConnection, case: GroundTruthCase, platform: str = "x"
) -> pd.DataFrame:
    """Corpus posts matching a case's claim text (collection-presence check).

    Matches on the claim text patterns (ILIKE); anchor handles are used later to
    locate the cluster, not here - a handle posts many unrelated things."""
    likes = " OR ".join(f"lower(text) LIKE '{p.lower()}'" for p in case.text_like)
    return con.sql(
        f"""
        SELECT platform_post_id, author_id, author_handle, created_at, text
        FROM ({latest_posts(con, platform).sql_query()})
        WHERE {likes}
        ORDER BY created_at
        """
    ).df()


def _triage_cut(n_stories: int) -> int:
    """Rank within which a story counts as surfaced (top TOP_FRACTION, floored)."""
    return max(int(np.ceil(TOP_FRACTION * n_stories)), TOP_MIN)


def _drop_stage(row: dict) -> str | None:
    if row["n_present"] == 0:
        return "collection"
    if row["n_embedded"] == 0:
        return "embedding"
    if row["story_id"] is None:
        return "clustering (sub-threshold)"
    if row["is_blob"]:
        return "clustering (blob/chaining)"
    if row["rank"] is None or row["rank"] > _triage_cut(row["n_stories"]):
        return "scoring"
    return None


def evaluate(
    con: duckdb.DuckDBPyConnection,
    days: int = 14,
    tau: float = st.DEFAULT_TAU,
    min_size: int = st.DEFAULT_MIN_SIZE,
    platform: str = "x",
    model: str = st.MODEL,
    cases: list[GroundTruthCase] | None = None,
) -> pd.DataFrame:
    """Run every ground-truth case through the pipeline once; return a report frame.

    One row per case: presence, embedding coverage, the story it lands in, whether
    that story is a degenerate blob, its suspicion rank, a green/red pass flag, and
    the first drop-out stage (None when green)."""
    cases = cases or GROUND_TRUTH
    embedded = semantic._embedded_ids(con, platform, model)
    stories = st.candidate_stories(con, days=days, tau=tau, min_size=min_size,
                                   platform=platform, model=model)
    if stories.empty:
        cards = pd.DataFrame(columns=["story_id", "story_suspicion_index"])
    else:
        cards = st.story_scorecard(con, stories, days=days, platform=platform, model=model)
    rank_of = {int(sid): i + 1 for i, sid in enumerate(cards["story_id"])}
    authors_of = (
        stories.groupby("story_id")["author_id"].nunique().to_dict()
        if not stories.empty else {}
    )

    rows = []
    for case in cases:
        present = resolve_posts(con, case, platform)
        ids = set(present["platform_post_id"])
        n_embedded = len(ids & embedded)

        story_id, story_authors, is_blob, rank, susp = None, None, False, None, None
        if not stories.empty and ids:
            mine = stories[stories["platform_post_id"].isin(ids)]
            if not mine.empty:
                # the story holding the most of this case's posts
                story_id = int(mine["story_id"].value_counts().idxmax())
                story_authors = int(authors_of.get(story_id, 0))
                is_blob = story_authors > BLOB_AUTHORS
                rank = rank_of.get(story_id)
                hit = cards[cards["story_id"] == story_id]
                susp = float(hit["story_suspicion_index"].iloc[0]) if len(hit) else None

        row = {
            "case": case.name,
            "verdict": case.verdict,
            "expect": case.expect,
            "n_present": len(present),
            "present_authors": present["author_id"].nunique(),
            "n_embedded": n_embedded,
            "story_id": story_id,
            "story_authors": story_authors,
            "is_blob": is_blob,
            "rank": rank,
            "n_stories": int(stories["story_id"].nunique()) if not stories.empty else 0,
            "triage_cut": _triage_cut(int(stories["story_id"].nunique())) if not stories.empty else 0,
            "suspicion": susp,
        }
        row["drop_stage"] = _drop_stage(row)
        surfaced = row["drop_stage"] is None
        # known-limitation cases are tracked but never required to pass; only the
        # "surface" cases gate a green run.
        row["pass"] = surfaced if case.expect == EXPECT_SURFACE else True
        row["surfaced"] = surfaced
        rows.append(row)
    return pd.DataFrame(rows)
