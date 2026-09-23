"""Profile each IOHunter operation's shape in the five traces, to pick the
canary's template and to calibrate it.

    cd analysis && uv run python investigations/2026-09-22-canary/01_profile_ops.py <data-dir>

`<data-dir>` holds `<country>/0.7_datasets.pkl`, as fetched from the
`iohunter-bench` volume with `modal volume get` (a download, not a run - it
still works when the workspace is over its spend limit, which runs do not).

The release ships networks, not tweets, so the shape available is the one the
detector consumes: per trace, how densely the operation's accounts link to each
other and to the organic accounts around them.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

COUNTRIES = ("UAE", "cuba", "russia", "venezuela", "iran", "china")
TRACES = ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")


def profile(path: Path) -> dict:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    labels = np.asarray(payload["labels"]).astype(int)
    pos = set(np.flatnonzero(labels == 1).tolist())
    n_pos, n_bg = len(pos), len(labels) - len(pos)
    pairs = n_pos * (n_pos - 1) / 2
    out = {"nodes": int(len(labels)), "positives": n_pos, "traces": {}}
    for name in TRACES:
        g = payload.get(name)
        if g is None:
            continue
        op_op = op_bg = bg_bg = 0
        touched: set[int] = set()
        for a, b in g.edges():
            a, b = int(a), int(b)
            if a == b:
                continue
            ia, ib = a in pos, b in pos
            if ia and ib:
                op_op += 1
                touched |= {a, b}
            elif ia or ib:
                op_bg += 1
                touched.add(a if ia else b)
            else:
                bg_bg += 1
        out["traces"][name] = {
            "op_touched": round(len(touched) / max(n_pos, 1), 3),
            "op_op_density": round(op_op / pairs, 4) if pairs else None,
            "bg_density": round(bg_bg / (n_bg * (n_bg - 1) / 2), 6) if n_bg > 1 else None,
            "op_op_deg": round(2 * op_op / max(n_pos, 1), 2),
            "op_bg_deg": round(op_bg / max(n_pos, 1), 2),
        }
    return out


def main() -> None:
    root = Path(sys.argv[1])
    rows = {c: profile(root / c / "0.7_datasets.pkl") for c in COUNTRIES if (root / c).exists()}
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
