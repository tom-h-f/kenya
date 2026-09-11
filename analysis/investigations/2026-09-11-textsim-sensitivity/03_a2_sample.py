"""Draw A2's size-matched case set: every v2 group against the nearest-size v1 cluster.

    cd analysis && uv run python investigations/2026-09-11-textsim-sensitivity/03_a2_sample.py \\
        investigations/2026-09-11-textsim-sensitivity/out/2026-09-05-promotion-off/sweep_min0.75

The 2026-09-09 comparison read v1's seven LARGEST clusters against v2 groups of
4-40 members. Engagement pods are exactly the large ones, so the gap it found
(1 of 7 against 6 of 7 political) could be size rather than method. This pairs
each v2 group with the v1 cluster nearest its size instead, the two largest v2
blobs included.

v2 groups are Leiden (CPM, resolution 0.5, seed 0) over the fused network among
the 0.85 top 500. 2026-09-09 reported this split as 255, 145, 28, 19, 16, 8, 7,
6 but recorded neither the seed nor the partition, and none of seeds 0-9
reproduces it exactly - seed 0 gives 255, 143, 27, 19, 16, 12, ... So the seed
is fixed here and the sizes it produced are printed beside the reported ones:
the two large blobs are the same objects, the small tail differs, and this
split, unlike that one, can be re-drawn.

Blinding: every case gets a random id, and which method produced it lives only
in `a2_key.csv`, which never goes to the reader.
"""

import argparse
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd

from kma import db
from kma.db import connect

REPORTED = [255, 145, 28, 19, 16, 8, 7, 6]
RESOLUTION = 0.5
LEIDEN_SEED = 0
MIN_MEMBERS = 4


def v2_groups(edges: pd.DataFrame) -> list[list[str]]:
    graph = ig.Graph.TupleList(edges.itertuples(index=False), directed=False)
    part = la.find_partition(graph, la.CPMVertexPartition, resolution_parameter=RESOLUTION, seed=LEIDEN_SEED)
    return sorted(([graph.vs[v]["name"] for v in members] for members in part), key=len, reverse=True)


def match(v2: list[list[str]], v1_sizes: pd.Series, seed: int = 0) -> list[tuple[int, int, int, int]]:
    """Largest v2 group first, each taking the unused v1 cluster nearest its size."""
    rng = np.random.default_rng(seed)
    available = v1_sizes.copy()
    pairs = []
    for index, group in enumerate(v2):
        gaps = (available - len(group)).abs()
        nearest = gaps[gaps == gaps.min()].index.to_numpy()
        chosen = rng.choice(nearest)
        pairs.append((index, len(group), chosen, int(available[chosen])))
        available = available.drop(chosen)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--seed", type=int, default=0, help="for tie-breaking and case ids, not Leiden")
    args = ap.parse_args()

    edges = pd.read_parquet(args.sweep_dir / "top500_fused_edges_t0.85.parquet")
    groups = v2_groups(edges)
    kept = [g for g in groups if len(g) >= MIN_MEMBERS]
    print(f"v2 split (CPM {RESOLUTION}, seed {LEIDEN_SEED}): {[len(g) for g in groups][:12]}")
    print(f"reported 2026-09-09:           {REPORTED}")
    print(f"-> {len(kept)} groups of {MIN_MEMBERS}+ members")

    con = connect()
    v1 = db.coordination_run_latest(con, kind="clusters").df()
    v1_sizes = v1.groupby("cluster_id").size()
    pairs = match(kept, v1_sizes, seed=args.seed)

    id_type = v1["author_id"].dtype
    cases = []
    for index, size, cluster_id, cluster_size in pairs:
        cases.append(("v2", f"group{index}", size, kept[index]))
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
    out = args.sweep_dir.parent / "a2_matched"
    out.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out / "a2_sample.parquet")
    pd.DataFrame(key).to_csv(out / "a2_key.csv", index=False)

    print(pd.DataFrame(pairs, columns=["v2_group", "v2_size", "v1_cluster", "v1_size"]).to_string(index=False))
    print(f"\n{len(key)} blinded cases, {len(sample)} member rows -> {out}")


if __name__ == "__main__":
    main()
