"""Per-floor recall of the 300 reader-labelled 0.85 pairs (2026-09-11 pair audit), on current cleaning."""
import sys
from pathlib import Path

import pandas as pd

from kma import coord2

root = Path(sys.argv[1])
j = pd.read_parquet(root / "pair_audit_judged_t0.85.parquet").dropna(subset=["label", "text_a", "text_b"])
j["ov"] = [coord2.text_overlap(coord2.clean_text(a), coord2.clean_text(b)) for a, b in zip(j.text_a, j.text_b)]
print(len(j), j.label.value_counts().to_dict())
rows = []
for f in (0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
    k = j[j.ov >= f]
    row = {"floor": f}
    for lab in ("same_message", "same_topic", "unrelated"):
        n = (j.label == lab).sum()
        row[lab] = f"{int((k.label == lab).sum())}/{n}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
