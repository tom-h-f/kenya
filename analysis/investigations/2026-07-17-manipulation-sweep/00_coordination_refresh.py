"""Compute SVN-validated coordination layers + Leiden clusters over the current
corpus and cache them under out/ for the other sweep scripts.

R2 has no persisted coordination/ runs, and persisting via
coordination.persist_clusters would feed the collector's adaptive targeting -
a live side effect this sweep deliberately avoids. Local parquet cache instead.

--sample runs the co_retweet channel only (wiring check); --full runs WAVE_A.
"""

from __future__ import annotations

import pandas as pd

import _lib
from kma import coordination as co
from kma import db


def main() -> None:
    args = _lib.parse_args("Coordination refresh (local cache, no R2 persist)")
    channels = ["co_retweet"] if args.sample else co.WAVE_A
    con = db.connect()

    print(f"building layers: {channels}")
    layers = co.build_layers(con, channels=channels)
    for ch, e in layers.items():
        print(f"  {ch}: {len(e)} validated edges")

    edges = pd.concat(
        [e.assign(channel=ch) for ch, e in layers.items() if not e.empty],
        ignore_index=True,
    ) if any(len(e) for e in layers.values()) else pd.DataFrame()
    members, summary = co.clusters(layers)
    print(f"{len(members)} member rows in {len(summary)} clusters (>=3 members)")

    _lib.save(edges, "00_coordination_edges.parquet")
    merged = members.merge(summary, on="cluster_id") if not members.empty else members
    _lib.save(merged, "00_coordination_clusters.parquet")
    _lib.show(summary, "cluster summary")
    _lib.print_caveats()


if __name__ == "__main__":
    main()
