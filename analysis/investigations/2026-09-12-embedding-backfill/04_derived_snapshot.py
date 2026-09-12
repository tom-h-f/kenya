"""A snapshot with the same posts as another, but today's embeddings.

    cd analysis && uv run python investigations/2026-09-12-embedding-backfill/04_derived_snapshot.py \\
        --from 2026-09-05-promotion-off --name 2026-09-05-promotion-off__emb20260912 [--dry-run]

`bench.pinned_source` reads an explicit object list, never a glob, so the
2026-09-05 snapshot cannot see the 567,030 embeddings backfilled on 2026-09-12
- it was frozen when 62% of text-eligible posts had vectors.

Taking a fresh snapshot instead would change two things at once: coverage AND a
week of newly collected posts. This changes one. Every prefix keeps the parent
snapshot's objects; `embeddings/` is relisted as it is now. So a text trace
built on this snapshot differs from the 2026-09-05 one only by the vectors that
were missing.

`rows` is left empty on the relisted objects, as `snapshot(rows=False)` does:
the row-count checks are not available on this manifest.
"""

import argparse
from datetime import datetime, timezone

import pandas as pd

from kma import bench
from kma.db import BUCKET, connect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="parent", default="2026-09-05-promotion-off")
    ap.add_argument("--name", required=True)
    ap.add_argument("--prefix", default="embeddings", help="the one prefix to relist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = connect()
    parent = bench.load(args.parent, con=con)
    kept = parent[parent["prefix"] != args.prefix]

    client = bench._r2_client()
    fresh = []
    for obj in bench.list_objects(client, BUCKET, args.prefix):
        fresh.append({
            **obj, "prefix": args.prefix, "path": f"r2://{BUCKET}/{obj['key']}",
            **bench.partition_values(obj["key"]),
        })
    relisted = pd.DataFrame(fresh)
    was = int((parent["prefix"] == args.prefix).sum())
    print(f"{args.prefix}: {was:,} objects in {args.parent}, {len(relisted):,} now "
          f"(+{len(relisted) - was:,})")

    now = datetime.now(timezone.utc)
    manifest = pd.concat([kept, relisted], ignore_index=True)
    manifest["snapshot"] = args.name
    manifest["created_at"] = now
    for column in ("code_version_min", "code_version_max", "kma_sha"):
        manifest[column] = parent[column].iloc[0] if len(parent) else ""
    print(f"manifest: {len(manifest):,} objects over {manifest['prefix'].nunique()} prefixes")
    if args.dry_run:
        return

    key = f"{bench.SNAPSHOT_PREFIX}/snapshot={args.name}/manifest.parquet"
    con.register("_manifest", manifest)
    try:
        con.execute(f"COPY _manifest TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)")
    finally:
        con.unregister("_manifest")
    print(f"wrote r2://{BUCKET}/{key}")


if __name__ == "__main__":
    main()
