"""Export the persisted relevance scores of one model slug to a local parquet.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/02_export_scores.py [slug]
"""

import sys
import time
from pathlib import Path

from kma import relevance
from kma.db import connect, relevance_source

HERE = Path(__file__).parent

slug = sys.argv[1] if len(sys.argv) > 1 else relevance.MODEL
con = connect()
start = time.perf_counter()
target = HERE / "out" / f"scores_{slug}.parquet"
con.execute(f"""COPY (
    SELECT platform_post_id, p_kenya FROM {relevance_source('x', slug)}
    QUALIFY row_number() OVER (PARTITION BY platform_post_id ORDER BY scored_at DESC) = 1
) TO '{target}' (FORMAT parquet)""")
n = con.sql(f"SELECT count(*) FROM '{target}'").fetchone()[0]
print(f"{n:,} scores in {time.perf_counter() - start:.0f}s -> {target}")
