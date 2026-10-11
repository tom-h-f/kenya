"""Derive the Kenyan-handle list from how the corpus itself uses each handle.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/06_handles.py

A handle is Kenyan when most of the non-retweet posts that mention it carry a
Kenyan anchor from the keyword lexicon (`measure._KENYA_RE`) and few carry a
neighbouring-country term. Not circular: neither input is a learned score, and
a Tanzanian account mentioned mostly in Tanzanian posts fails the second test.
Writes out/kenyan_handles.txt (one lowercase handle per line) and prints the
size at each setting so the choice is on the record.
"""

import duckdb
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
REGION_MAX = 0.05

con = duckdb.connect()
con.execute(f"""
CREATE TABLE m AS
SELECT lower(h) AS handle, f.anchor, (f.regions <> '') AS region
FROM read_parquet('{OUT / "corpus.parquet"}') c
JOIN read_parquet('{OUT / "features.parquet"}') f USING (platform_post_id),
     unnest(c.mentions) AS t(h)
WHERE NOT f.is_rt
""")
stats = con.sql("""
SELECT handle, count(*) n, avg(anchor::int) anchor_share, avg(region::int) region_share
FROM m GROUP BY handle
""").df()
stats.to_parquet(OUT / "handle_stats.parquet")
print(len(stats), "handles seen")
for n_min, share in [(5, 0.6), (10, 0.3), (20, 0.3), (20, 0.4), (30, 0.4)]:
    keep = stats[(stats.n >= n_min) & (stats.anchor_share >= share) & (stats.region_share <= REGION_MAX)]
    print(f"n>={n_min} anchor>={share}: {len(keep):,} handles")
# Replies rarely repeat an anchor word, so the anchor share of a clearly Kenyan
# account sits at 0.3-0.6 (@naomikibandi 0.47, @williamsruto 0.34); 0.3 with a
# floor of 20 mentions keeps those and still reads as Kenyan down the list.
N_MIN, SHARE = 20, 0.3
keep = stats[(stats.n >= N_MIN) & (stats.anchor_share >= SHARE) & (stats.region_share <= REGION_MAX)]
(OUT / "kenyan_handles.txt").write_text("\n".join(sorted(keep.handle)) + "\n")
print(f"wrote {len(keep):,} handles (n>={N_MIN}, anchor share>={SHARE}, region share<={REGION_MAX})")
for h in ("ruto_tutam2027", "lindamwananchi_", "naomikibandi", "samia_suluhu", "tundulissu"):
    print(h, stats[stats.handle == h].to_dict("records"))
