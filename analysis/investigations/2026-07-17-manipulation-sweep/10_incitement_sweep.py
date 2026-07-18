"""Incitement sweep: read persisted incitement scores and cross them with the
other lenses.

- flagged posts: lexicon hit AND high NLI score (joint rule - either signal
  alone over-triggers; see 10_validation_100.csv)
- daily trend of flagged volume by category
- overlap with coordination clusters and the 09 convergence set
- aggregate region/community slice of flagged authors via deltas.slice_claim
  (aggregate-only, min-coverage gated, disclaimer attached)
"""

from __future__ import annotations

import pandas as pd

import _lib
from kma import db
from kma.deltas import slice_claim

NLI_CUT = 0.85


def main() -> None:
    _lib.parse_args("Incitement sweep", default_sample=0)
    con = _lib.connect()

    inc = db.latest_incitement(con).df()
    print(f"{len(inc)} scored posts ({inc['lexicon_hits'].str.len().gt(0).sum()} lexicon hits)")
    posts = con.sql(
        f"""
        SELECT platform_post_id, author_id, author_handle, created_at,
               left(text, 110) AS text
        FROM {db.posts_source('x')}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY collected_at DESC
        ) = 1
        """
    ).df()
    m = inc.merge(posts, on="platform_post_id")
    m["max_nli"] = m[
        ["dehumanisation_score", "violence_call_score", "othering_score"]
    ].max(axis=1)
    flagged = m[(m["lexicon_hits"].str.len() > 0) & (m["max_nli"] >= NLI_CUT)]
    flagged = flagged.sort_values("max_nli", ascending=False)
    _lib.show(
        flagged[["author_handle", "created_at", "lexicon_hits",
                 "dehumanisation_score", "violence_call_score",
                 "othering_score", "political_criticism_score", "text"]],
        f"flagged: lexicon hit AND max NLI >= {NLI_CUT}",
        n=15,
    )
    _lib.save(
        flagged.drop(columns=["max_nli"]), "10_flagged.csv"
    )

    tail = m[
        (m["lexicon_hits"].str.len() == 0)
        & (m["dehumanisation_score"] >= 0.9)
        & (m["violence_call_score"] >= 0.9)
        & (m["political_criticism_score"] <= 0.4)
    ].sort_values("violence_call_score", ascending=False)
    _lib.show(
        tail[["author_handle", "created_at", "dehumanisation_score",
              "violence_call_score", "othering_score",
              "political_criticism_score", "text"]],
        "NLI-only extreme tail (no lexicon hit - lexicon-expansion candidates)",
        n=15,
    )
    _lib.save(tail.drop(columns=["max_nli"]), "10_nli_tail.csv")

    trend = (
        flagged.assign(day=pd.to_datetime(flagged["created_at"], utc=True).dt.date)
        .explode("lexicon_categories")
        .groupby(["day", "lexicon_categories"])
        .size()
        .rename("n")
        .reset_index()
    )
    _lib.show(trend, "daily flagged volume by category", n=20)
    _lib.save(trend, "10_trend.csv")

    coord = set(_lib.coordination_clusters()["author_id"])
    conv = pd.read_csv(
        _lib.OUT / "09_convergence.csv", dtype={"platform_user_id": str}
    )
    multi = set(conv[conv["n_lenses"] >= 2]["platform_user_id"])
    flagged_authors = set(flagged["author_id"])
    print(
        f"\nflagged authors: {len(flagged_authors)}; in coordination cluster: "
        f"{len(flagged_authors & coord)}; in >=2-lens convergence set: "
        f"{len(flagged_authors & multi)}"
    )

    authors = db.latest_authors(con).df()[["platform_user_id", "location"]]
    fa = authors[authors["platform_user_id"].isin(flagged_authors)]
    for dim in ("region", "community"):
        s = slice_claim(fa, dimension=dim)
        _lib.show(s, f"flagged-author aggregate by {dim}")
        if not s.empty:
            print("  disclaimer:", s["disclaimer"].iloc[0][:160])
        _lib.save(s, f"10_slice_{dim}.csv")

    _lib.print_caveats()


if __name__ == "__main__":
    main()
