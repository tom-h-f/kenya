"""Persist rescored parts to the R2 `relevance/` prefix under a model slug.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/11_write_scores.py <slug> out/scores_new [--dry-run]

Same key layout as `06_write.py` and `relevance.write_scores`:
`relevance/platform=x/model=<slug>/dt=<date>/run=<stamp>.parquet`. One object per
part, retried, resumable through a manifest beside the parts. Refuses to write a
slug whose prefix already has objects unless --append is given, so a rerun after
a half-finished pass is deliberate.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from kma.db import BUCKET, connect

ATTEMPTS = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slug")
    ap.add_argument("parts", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(args.parts.glob("part_*.parquet"))
    manifest = args.parts / "written.json"
    done = set(json.loads(manifest.read_text())["done"]) if manifest.exists() else set()
    todo = [f for f in files if f.name not in done]
    print(f"{len(files)} parts, {len(done)} written, {len(todo)} to go")
    if args.dry_run:
        return
    con = connect()
    for path in todo:
        part = pd.read_parquet(path)
        part["platform_post_id"] = part["platform_post_id"].astype(str)
        now = datetime.now(timezone.utc)
        buf = part.assign(model=args.slug, scored_at=now)[["platform_post_id", "model", "p_kenya", "scored_at"]]
        key = (f"relevance/platform=x/model={args.slug}/dt={now:%Y-%m-%d}/"
               f"run={now:%Y%m%dT%H%M%SZ}-{path.stem}.parquet")
        con.register("_rel_buf", buf)
        try:
            for attempt in range(1, ATTEMPTS + 1):
                try:
                    con.execute(f"COPY _rel_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)")
                    break
                except duckdb.Error as exc:
                    if attempt == ATTEMPTS:
                        raise
                    print(f"  retry {attempt}: {str(exc)[:80]}", flush=True)
                    time.sleep(5 * attempt)
        finally:
            con.unregister("_rel_buf")
        done.add(path.name)
        manifest.write_text(json.dumps({"done": sorted(done)}))
        print(f"  wrote {len(part):,} -> {key}", flush=True)
        time.sleep(1.1)


if __name__ == "__main__":
    main()
