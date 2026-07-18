"""Narrative laundering: which claims start on the fringe and get picked up by
big accounts - and who does the seeding.

Stories = stories.candidate_stories (live; R2 stories/ has no persisted runs).
Per story, distinct authors ordered by first captured post:
- opening stratum (followers of the first authors)
- lag from first fringe post to first big-account post
- seeder suspicion (authenticity heuristic) + coordination-cluster membership

Ordering is within-capture only: the earliest collected post is not
necessarily patient-zero (SAMPLING_CAVEAT printed on exit).
"""

from __future__ import annotations

import pandas as pd

import _lib
from kma import db
from kma import stories as st
from kma.authenticity import heuristic_score

FRINGE_FOLLOWERS = 1_000
BIG_FOLLOWERS = 100_000
DAYS = 14


def main() -> None:
    args = _lib.parse_args("Narrative laundering", default_sample=30)
    con = _lib.connect()

    members = st.candidate_stories(con, days=DAYS, include_thin=True)
    print(
        f"{members['story_id'].nunique()} stories, {len(members)} member posts "
        f"(last {DAYS}d)"
    )
    if args.sample:
        members = members[members["story_id"] < args.sample]

    authors = db.latest_authors(con).df()[
        ["platform_user_id", "followers_count", "verified", "blue"]
    ]
    m = members.merge(
        authors, left_on="author_id", right_on="platform_user_id", how="left"
    )
    susp = heuristic_score(con).df().set_index("platform_user_id")["suspicion"]
    coord = set(_lib.coordination_clusters()["author_id"])

    rows = []
    for sid, grp in m.groupby("story_id"):
        firsts = (
            grp.sort_values("created_at").drop_duplicates("author_id")
        )
        seed = firsts.iloc[0]
        big = firsts[firsts["followers_count"] >= BIG_FOLLOWERS]
        fringe_first = firsts["followers_count"].iloc[0] < FRINGE_FOLLOWERS
        lag_to_big = (
            (big["created_at"].iloc[0] - seed["created_at"]).total_seconds() / 3600
            if not big.empty
            else None
        )
        first3 = firsts.head(3)
        rows.append(
            {
                "story_id": sid,
                "n_authors": firsts["author_id"].nunique(),
                "seed_handle": seed["author_handle"],
                "seed_followers": seed["followers_count"],
                "seed_suspicion": round(
                    float(susp.get(seed["author_id"], float("nan"))), 3
                ),
                "seed_in_coord": seed["author_id"] in coord,
                "fringe_seeded": fringe_first,
                "picked_up_by_big": not big.empty,
                "big_handle": None if big.empty else big["author_handle"].iloc[0],
                "lag_to_big_h": None if lag_to_big is None else round(lag_to_big, 1),
                "early_in_coord": int(first3["author_id"].isin(coord).sum()),
                "text": seed["text"][:110],
            }
        )
    stories_df = pd.DataFrame(rows)
    laundered = stories_df[
        stories_df["fringe_seeded"] & stories_df["picked_up_by_big"]
    ].sort_values("lag_to_big_h")
    _lib.show(
        laundered,
        "fringe-seeded stories later carried by >=100k-follower accounts",
    )
    suspicious_seeds = stories_df[
        (stories_df["seed_suspicion"] >= 0.5)
        | stories_df["seed_in_coord"]
        | (stories_df["early_in_coord"] >= 2)
    ].sort_values("n_authors", ascending=False)
    _lib.show(
        suspicious_seeds,
        "stories seeded/opened by high-suspicion or coordination-cluster accounts",
    )
    _lib.save(stories_df, "06_story_launder.csv")

    _lib.print_caveats()


if __name__ == "__main__":
    main()
