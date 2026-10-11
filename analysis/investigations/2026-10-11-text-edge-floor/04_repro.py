"""Reproduction gate: the locally rebuilt user-level trace against the persisted production trace."""
import pandas as pd
from kma import coord2_run
from kma.db import BUCKET, connect
from common import WINDOW, wdir

con = connect()
key = coord2_run.textsim_key(WINDOW, 0.85, overlap=0.10)
prod = con.sql(f"SELECT * FROM read_parquet('r2://{BUCKET}/{key}')").df()
mine = pd.read_parquet(wdir(WINDOW) / "pairs_manifest" / "edges.parquet")
print(key, len(prod), "mine", len(mine))
p = set(zip(prod.source.astype(str), prod.target.astype(str)))
m = set(zip(mine.source.astype(str), mine.target.astype(str)))
print("both", len(p & m), "prod only", len(p - m), "mine only", len(m - p))
j = prod.astype({"source": str, "target": str}).merge(mine.astype({"source": str, "target": str}), on=["source", "target"])
print("weight max abs diff", float((j.weight_x - j.weight_y).abs().max()))
