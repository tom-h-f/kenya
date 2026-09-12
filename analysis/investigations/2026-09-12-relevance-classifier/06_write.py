"""Persist the relevance scores to the R2 `relevance/` prefix.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/06_write.py [--dry-run]

One row per post: platform_post_id, model, p_kenya, scored_at. The probability
is persisted, not a verdict - the threshold is a reader's choice and is not
tuned. Keys follow the other enrichment prefixes
(`relevance/platform=x/model=<slug>/dt=<date>/run=<stamp>.parquet`), so
`db.relevance_source`-style globbing works the same way.

Nothing reads this yet: `kma.measure.domain_bucket` still decides `kenya_share`.
Adopting it is a separate change, because it moves every Kenya-share figure in
the project.

Resumable through `out/relevance_written.json`, with per-chunk retries: the
embedding backfill lost two passes to dropped uploads and a sleeping laptop.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from kma.db import BUCKET, connect

HERE = Path(__file__).parent
ATTEMPTS = 3
MODEL_SLUG = "kenya-relevance-afroxlmr-2026-09-12"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = HERE / "out"
    scores = pd.read_parquet(out / "scores.parquet")
    scores["platform_post_id"] = scores["platform_post_id"].astype(str)
    manifest = out / "relevance_written.json"
    done = set(json.loads(manifest.read_text())["done"]) if manifest.exists() else set()
    starts = [s for s in range(0, len(scores), args.chunk) if s not in done]
    print(f"{len(scores):,} scored, {len(done)} chunks written, {len(starts)} to go; "
          f"p>=0.5 on {float((scores['p_kenya'] >= 0.5).mean()):.3f}")
    if args.dry_run:
        return

    con = connect()
    for start in starts:
        part = scores.iloc[start : start + args.chunk]
        now = datetime.now(timezone.utc)
        buf = part.assign(model=MODEL_SLUG, scored_at=now)[
            ["platform_post_id", "model", "p_kenya", "scored_at"]
        ]
        key = f"relevance/platform=x/model={MODEL_SLUG}/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
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
        done.add(start)
        manifest.write_text(json.dumps({"chunk": args.chunk, "done": sorted(done)}))
        print(f"  wrote {len(part):,} -> {key}", flush=True)
        time.sleep(1.1)


if __name__ == "__main__":
    main()
