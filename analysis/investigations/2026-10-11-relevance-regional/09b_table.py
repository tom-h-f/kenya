"""One row per run: the numbers that decide which model ships.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/09b_table.py out/data/runs old strip_s0 ...
"""

import sys
from pathlib import Path

import pandas as pd

from importlib import import_module

ev = import_module("09_eval")

runs = Path(sys.argv[1])
rows = []
for name in sys.argv[2:]:
    r = ev.evaluate(runs / name)
    g = r["regional"]
    rows.append({
        "run": name,
        "h88_P_w": r["human88"]["weighted"]["precision"], "h88_R_w": r["human88"]["weighted"]["recall"],
        "h88_F1_w": r["human88"]["weighted"]["f1"],
        "b2h_F1_w": r["b2human"]["weighted"]["f1"],
        "reg_P_w": g["weighted"]["precision"], "reg_R_w": g["weighted"]["recall"], "reg_F1_w": g["weighted"]["f1"],
        "reg_F1_unw": g["unweighted"]["f1"],
        "fp_w": g["fp_rate_weighted"]["rate"], "fp_unw": g["fp_rate_unweighted"]["rate"],
        "tz_off": f"{g['tanzania_offdomain']['called_kenya']}/{g['tanzania_offdomain']['n']}",
        "tz200_fp": r["tz200"]["tanzania_only_fp_rate"], "tz200_krec": r["tz200"]["kenya_recall"],
        "hardpos_R": g["hard_positive_recall"]["recall"],
    })
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(Path(__file__).parent / "candidates.csv", index=False)
