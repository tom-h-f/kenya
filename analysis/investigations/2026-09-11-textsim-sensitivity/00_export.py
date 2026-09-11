"""Export the text-similarity inputs for one snapshot, so the GPU pass can run off-R2.

    cd analysis && uv run python investigations/2026-09-11-textsim-sensitivity/00_export.py \\
        --snapshot 2026-09-05-promotion-off [--limit 50000]

Mirrors `modal_textsim.build` up to the similarity search, row for row: the same
eligible rows (`coord2.text_rows`, ordered by post_id), the same latest-embedding
dedup, the same inner join. Writes `out/<snapshot>[__limitN]/rows.parquet` and
`matrix.npy`. The matrix is float32 because that is what the GPU pass computes
in, so nothing is lost by storing it that way.

Runs where the R2 credentials are; the GPU box never needs them.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from kma import bench, coord2
from kma.db import connect

HERE = Path(__file__).parent


def export(snapshot: str, limit: int = 0):
    con = connect()
    manifest = bench.load(snapshot, con=con)
    posts = bench.pinned_source(manifest, "posts")
    embeddings = bench.pinned_source(manifest, "embeddings")

    rows = coord2.text_rows(con, coord2.posts_view(posts))
    if limit:
        rows = rows.head(limit)

    con.register("_rows", rows[["post_id"]])
    vecs = con.sql(
        f"""SELECT e.platform_post_id AS post_id, e.embedding
            FROM {embeddings} e JOIN _rows r ON r.post_id = e.platform_post_id
            QUALIFY row_number() OVER (PARTITION BY e.platform_post_id ORDER BY e.embedded_at DESC) = 1"""
    ).df()

    joined = rows.merge(vecs, on="post_id", how="inner").reset_index(drop=True)
    matrix = np.vstack(joined["embedding"].to_numpy()).astype(np.float32)
    return joined[["post_id", "user_id", "created_at"]], matrix, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default="2026-09-05-promotion-off")
    ap.add_argument("--limit", type=int, default=0, help="0 for every eligible row")
    args = ap.parse_args()

    start = time.perf_counter()
    rows, matrix, eligible = export(args.snapshot, args.limit)
    scope = f"__limit{args.limit}" if args.limit else ""
    out = HERE / "out" / f"{args.snapshot}{scope}"
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "rows.parquet")
    np.save(out / "matrix.npy", matrix)

    print(f"eligible rows {eligible:,}, with an embedding {len(rows):,}, matrix {matrix.shape}")
    print(f"wrote {out} in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
