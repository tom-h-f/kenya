"""A2 again, on what v2 now reports: its listed communities against size-matched v1 clusters.

    cd analysis && uv run python investigations/2026-09-12-component-ranking/04_a2_sample.py \\
        out/2026-09-05-promotion-off/communities_production.parquet --out <dir>

The 2026-09-11 size-matched run read v2 groups cut from a top 500 that belonged
to one dense block, built on a text trace whose pairs were mostly unrelated.
This draws the cases from the community report instead (`03_communities.py`,
production text trace with the word-overlap floor): each listed community is
one case, as a reader would see it - its listed accounts - and each is paired
with the unused v1 cluster nearest its size, largest first, as before.

Blinding as before: random case ids, the method only in `a2_key.csv`, which the
reader never sees. `05_a2_headless.py` and `04_a2_report.py` in the textsim
investigation read the output unchanged.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kma import db
from kma.db import connect


def match(v2_sizes: list[int], v1_sizes: pd.Series, seed: int = 0) -> list[tuple[int, int, int, int]]:
    """Largest v2 case first, each taking the unused v1 cluster nearest its size."""
    rng = np.random.default_rng(seed)
    available = v1_sizes.copy()
    pairs = []
    for index in np.argsort(v2_sizes)[::-1]:
        size = v2_sizes[index]
        gaps = (available - size).abs()
        chosen = rng.choice(gaps[gaps == gaps.min()].index.to_numpy())
        pairs.append((int(index), size, chosen, int(available[chosen])))
        available = available.drop(chosen)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("communities", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="for tie-breaking and case ids")
    args = ap.parse_args()

    scored = pd.read_parquet(args.communities)
    listed = scored[scored["listed"]]
    v2 = [(int(c), group["user_id"].astype(str).tolist()) for c, group in listed.groupby("community", sort=False)]
    print(f"v2: {len(v2)} listed communities, sizes {sorted((len(m) for _, m in v2), reverse=True)}")

    con = connect()
    v1 = db.coordination_run_latest(con, kind="clusters").df()
    v1_sizes = v1.groupby("cluster_id").size()
    pairs = match([len(m) for _, m in v2], v1_sizes, seed=args.seed)

    id_type = v1["author_id"].dtype
    cases = []
    for index, size, cluster_id, cluster_size in pairs:
        cases.append(("v2", f"community{v2[index][0]}", size, v2[index][1]))
        cases.append(("v1", str(cluster_id), cluster_size,
                      v1.loc[v1["cluster_id"] == cluster_id, "author_id"].tolist()))

    order = np.random.default_rng(args.seed + 1).permutation(len(cases))
    members, key = [], []
    for case_id, position in enumerate(order, start=1):
        method, source, size, authors = cases[position]
        members += [(case_id, author) for author in authors]
        key.append({"case_id": case_id, "method": method, "source_id": source, "size": size})

    sample = pd.DataFrame(members, columns=["cluster_id", "author_id"])
    sample["author_id"] = sample["author_id"].astype(id_type)
    args.out.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out / "a2_sample.parquet")
    pd.DataFrame(key).to_csv(args.out / "a2_key.csv", index=False)
    print(pd.DataFrame(pairs, columns=["v2_case", "v2_size", "v1_cluster", "v1_size"]).to_string(index=False))
    print(f"\n{len(key)} blinded cases, {len(sample)} member rows -> {args.out}")


if __name__ == "__main__":
    main()
