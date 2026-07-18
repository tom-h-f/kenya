"""Follows-graph corroboration (weak lens by design).

The follow crawl BFSes outward from suspicion-flagged seeds, so edge
observation is conditioned on the crawl frontier: density comparisons against
the whole corpus are structurally biased and are NOT made here. Claims are
restricted to the crawled set:

- mutual-follow (reciprocal) pairs among coordination-cluster members
- follow density among flagged accounts vs random same-size subsets OF THE
  CRAWLED SET (the only defensible null)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _lib
from kma import db

DRAWS = 500


def main() -> None:
    args = _lib.parse_args("Follows forensics", default_sample=100)
    con = _lib.connect()

    follows = db.latest_follows(con).df()[["follower_id", "followed_id"]]
    crawled = set(follows["follower_id"])
    universe = sorted(crawled | set(follows["followed_id"]))
    print(
        f"{len(follows)} follow edges; {len(crawled)} crawled accounts, "
        f"{len(universe)} total in graph"
    )
    edge_set = set(map(tuple, follows.itertuples(index=False)))

    clusters = _lib.coordination_clusters()
    flagged = sorted(set(clusters["author_id"]) & set(universe))
    print(f"{len(flagged)} coordination-cluster members appear in follow graph")

    def internal_edges(ids: list[str]) -> int:
        s = set(ids)
        return sum((a in s and b in s) for a, b in edge_set)

    obs = internal_edges(flagged)
    mutual = sum(
        (b, a) in edge_set
        for a, b in edge_set
        if a in set(flagged) and b in set(flagged) and a < b
    )
    rng = np.random.default_rng(7)
    draws = DRAWS if not args.sample else 100
    null = np.array(
        [
            internal_edges(list(rng.choice(universe, len(flagged), replace=False)))
            for _ in range(draws)
        ]
    )
    p = float((1 + (null >= obs).sum()) / (draws + 1))
    result = pd.DataFrame(
        [
            {
                "flagged_in_graph": len(flagged),
                "internal_edges": obs,
                "mutual_pairs": mutual,
                "null_mean": round(float(null.mean()), 2),
                "null_p95": float(np.quantile(null, 0.95)),
                "p_vs_random_subset": p,
            }
        ]
    )
    _lib.show(result, "coordination-cluster follow density vs crawled-set null")
    _lib.save(result, "08_follow_density.csv")

    _lib.print_caveats()


if __name__ == "__main__":
    main()
