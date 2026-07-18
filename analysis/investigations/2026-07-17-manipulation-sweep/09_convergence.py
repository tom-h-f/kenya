"""Convergence matrix: accounts flagged by >= 2 independent lenses.

Lenses (one column each, from the out/ artifacts of scripts 00-06):
- coord_cluster: SVN-validated coordination cluster member (00)
- fast_replier: habitual first replier, z <= -2 vs uniform thread rank (04)
- seeder: seeds >= 2 multi-author copypasta components (03)
- awakened: dormant-then-activated posting pattern (01)
- regular: posting-interval CV < 0.8 with >= 20 posts (01)
- fringe_seed: seeded a story that big accounts later carried (06)
- suspicion: authenticity heuristic >= 0.5

Independence caveat: lenses share the capture, not the features; convergence
raises triage priority, it does not prove inauthenticity.
"""

from __future__ import annotations

import pandas as pd

import _lib
from kma import db
from kma.authenticity import heuristic_score


def main() -> None:
    _lib.parse_args("Convergence matrix", default_sample=0)
    con = _lib.connect()
    out = _lib.OUT

    authors = db.latest_authors(con).df()[
        ["platform_user_id", "handle", "followers_count", "verified"]
    ]
    by_handle = authors.set_index("handle")["platform_user_id"]

    flags: dict[str, set] = {}
    flags["coord_cluster"] = set(_lib.coordination_clusters()["author_id"])

    fast = pd.read_csv(out / "04_fast_repliers.csv")
    flags["fast_replier"] = set(
        fast[fast["z_vs_uniform"] <= -2]["author_handle"].map(by_handle).dropna()
    )

    seeders = pd.read_csv(out / "03_seeders.csv")
    flags["seeder"] = set(
        seeders[seeders["seeds"] >= 2]["seed_handle"].map(by_handle).dropna()
    )

    awakened = pd.read_csv(out / "01_awakened.csv", dtype={"platform_user_id": str})
    flags["awakened"] = set(awakened["platform_user_id"])

    regular = pd.read_csv(out / "01_regular_intervals.csv", dtype={"author_id": str})
    flags["regular"] = set(regular[regular["cv"] < 0.8]["author_id"])

    launder = pd.read_csv(out / "06_story_launder.csv")
    flags["fringe_seed"] = set(
        launder[launder["fringe_seeded"] & launder["picked_up_by_big"]][
            "seed_handle"
        ].map(by_handle).dropna()
    )

    susp = heuristic_score(con).df()
    flags["suspicion"] = set(
        susp[susp["suspicion"] >= 0.5]["platform_user_id"]
    )

    all_ids = sorted(set().union(*flags.values()))
    matrix = pd.DataFrame({"platform_user_id": all_ids})
    for name, ids in flags.items():
        matrix[name] = matrix["platform_user_id"].isin(ids)
    matrix["n_lenses"] = matrix[list(flags)].sum(axis=1)
    matrix = matrix.merge(
        authors, on="platform_user_id", how="left"
    ).sort_values(["n_lenses", "followers_count"], ascending=[False, True])

    top = matrix[matrix["n_lenses"] >= 2]
    _lib.show(
        top[["handle", "n_lenses", "followers_count", "verified"] + list(flags)],
        "accounts flagged by >= 2 independent lenses",
        n=25,
    )
    _lib.save(matrix, "09_convergence.csv")
    print(
        f"\n{len(top)} accounts >= 2 lenses; "
        f"{int((matrix['n_lenses'] >= 3).sum())} accounts >= 3 lenses"
    )
    _lib.print_caveats()


if __name__ == "__main__":
    main()
