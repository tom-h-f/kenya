"""Compare models on every eval set, from the prediction files each run wrote.

    cd analysis && uv run python investigations/2026-10-11-relevance-regional/09_eval.py out/data/runs old handles_s0 strip_s0 [--out report.json]

Per model and set: precision / recall / F1 of the Kenya call (unclear dropped, as
`measure_eval.score` drops it), unweighted and corpus-weighted.

Weights. human88, b2human, b2model: the gate-bucket shares of `measure_sample.csv`
(kenya 0.225, ambiguous 0.747, offdomain 0.028), as the 2026-09-13 figures used -
this reproduces the recorded 0.889 / 0.827. regional: each post stands for
N_stratum / n_stratum posts of the corpus universe, so the figures are over the
whole corpus mix, not over the hard strata alone.

The regional false-positive rate is the share of posts a reader labelled
`offdomain` that the model calls Kenyan, over the posts that name a neighbouring
or African country (strata `region_*`); tz200's is the 2026-10-11 Tanzania figure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 0.5


def prf(actual: np.ndarray, predicted: np.ndarray, w: np.ndarray | None = None) -> dict:
    w = np.ones(len(actual)) if w is None else w
    tp, fp, fn = w[predicted & actual].sum(), w[predicted & ~actual].sum(), w[~predicted & actual].sum()
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p + r else float("nan")
    return {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f1), 3)}


def counts(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {"tp": int((actual & predicted).sum()), "fp": int((~actual & predicted).sum()),
            "fn": int((actual & ~predicted).sum()), "tn": int((~actual & ~predicted).sum())}


def gate_weights(frame: pd.DataFrame, shares: dict) -> np.ndarray:
    return (frame["bucket"].map(shares) / frame.groupby("bucket")["bucket"].transform("size")).to_numpy()


def binary_set(frame: pd.DataFrame, weights: np.ndarray | None, t: float) -> dict:
    keep = frame["label"].isin(("kenya", "offdomain")).to_numpy()
    f = frame[keep]
    w = None if weights is None else weights[keep]
    actual, pred = (f["label"] == "kenya").to_numpy(), (f["p_kenya"] >= t).to_numpy()
    return {"n": int(keep.sum()), **counts(actual, pred), "unweighted": prf(actual, pred),
            **({"weighted": prf(actual, pred, w)} if w is not None else {})}


def regional_fp(f: pd.DataFrame, t: float, weighted: bool) -> dict:
    off = f[(f["label"] == "offdomain") & f["stratum"].str.startswith("region_")]
    w = off["weight"].to_numpy() if weighted else np.ones(len(off))
    called = (off["p_kenya"] >= t).to_numpy()
    return {"n_offdomain": len(off), "called_kenya": int(called.sum()),
            "rate": round(float(w[called].sum() / w.sum()), 3) if len(off) else None}


def evaluate(run: Path, t: float = THRESHOLD) -> dict:
    shares = json.loads((run.parent.parent / "gate_shares.json").read_text())
    out: dict = {}
    for name in ("human88", "b2human", "b2model"):
        path = run / f"preds_{name}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            out[name] = binary_set(frame, None, t)
            kept = frame[frame["label"].isin(("kenya", "offdomain"))].reset_index(drop=True)
            actual, pred = (kept["label"] == "kenya").to_numpy(), (kept["p_kenya"] >= t).to_numpy()
            out[name]["weighted"] = prf(actual, pred, gate_weights(kept, shares))
    path = run / "preds_tz200.parquet"
    if path.exists():
        tz = pd.read_parquet(path)
        only = tz[tz["label"] == "tanzania_only"]
        kenya = tz[tz["label"] == "kenya"]
        out["tz200"] = {"tanzania_only_n": len(only), "tanzania_only_called_kenya": int((only["p_kenya"] >= t).sum()),
                        "tanzania_only_fp_rate": round(float((only["p_kenya"] >= t).mean()), 3),
                        "kenya_n": len(kenya), "kenya_called_kenya": int((kenya["p_kenya"] >= t).sum()),
                        "kenya_recall": round(float((kenya["p_kenya"] >= t).mean()), 3)}
    path = run / "preds_regional.parquet"
    if path.exists():
        r = pd.read_parquet(path)
        w = r["weight"].to_numpy()
        out["regional"] = binary_set(r, w, t)
        out["regional"]["fp_rate_weighted"] = regional_fp(r, t, True)
        out["regional"]["fp_rate_unweighted"] = regional_fp(r, t, False)
        per_group = {}
        for g, grp in r.groupby(r["stratum"].str.extract(r"^region_(.+)_(?:hi|lo)$")[0].fillna("-")):
            if g != "-":
                per_group[g] = regional_fp(grp, t, True) | {"unweighted": regional_fp(grp, t, False)["rate"]}
        out["regional"]["fp_by_group"] = per_group
        tzo = r[(r["label"] == "offdomain") & (r["country"] == "tanzania")]
        out["regional"]["tanzania_offdomain"] = {"n": len(tzo), "called_kenya": int((tzo["p_kenya"] >= t).sum())}
        kenya_hard = r[(r["label"] == "kenya") & r["stratum"].isin(
            ["anchor_region", "diaspora", "swahili_kenyan", "handle_short", "c8"])]
        out["regional"]["hard_positive_recall"] = {
            "n": len(kenya_hard), "recall": round(float((kenya_hard["p_kenya"] >= t).mean()), 3) if len(kenya_hard) else None}
        three = pd.crosstab(r["label"], np.where(r["p_kenya"] >= t, "called_kenya", "called_other"))
        out["regional"]["confusion_3way"] = three.to_dict()
        by_stratum = {}
        for s, grp in r.groupby("stratum"):
            lab = grp[grp["label"].isin(("kenya", "offdomain"))]
            by_stratum[s] = {"n": len(grp), "kenya": int((grp["label"] == "kenya").sum()),
                             "offdomain": int((grp["label"] == "offdomain").sum()),
                             "unclear": int((grp["label"] == "unclear").sum()),
                             "called_kenya": int((grp["p_kenya"] >= t).sum())}
        out["regional"]["by_stratum"] = by_stratum
    return out


def sweep(run: Path, thresholds=(0.3, 0.5, 0.7, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    shares = json.loads((run.parent.parent / "gate_shares.json").read_text())
    rows = []
    r = pd.read_parquet(run / "preds_regional.parquet")
    h = pd.read_parquet(run / "preds_human88.parquet")
    h = h[h["label"].isin(("kenya", "offdomain"))].reset_index(drop=True)
    for t in thresholds:
        reg = binary_set(r, r["weight"].to_numpy(), t)
        hum = prf((h["label"] == "kenya").to_numpy(), (h["p_kenya"] >= t).to_numpy(), gate_weights(h, shares))
        rows.append({"threshold": t, "regional_w_P": reg["weighted"]["precision"], "regional_w_R": reg["weighted"]["recall"],
                     "regional_w_F1": reg["weighted"]["f1"], "regional_fp_rate": regional_fp(r, t, True)["rate"],
                     "human88_w_P": hum["precision"], "human88_w_R": hum["recall"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()
    report = {n: evaluate(args.runs / n, args.threshold) for n in args.names}
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, default=str))
    for n, res in report.items():
        print(f"== {n}")
        for k in ("human88", "b2human", "b2model"):
            if k in res:
                print(f"  {k}: n={res[k]['n']} unw {res[k]['unweighted']} w {res[k]['weighted']} "
                      f"fp {res[k]['fp']} fn {res[k]['fn']}")
        if "tz200" in res:
            print("  tz200:", res["tz200"])
        if "regional" in res:
            g = res["regional"]
            print(f"  regional: n={g['n']} unw {g['unweighted']} w {g['weighted']}")
            print(f"    fp rate weighted {g['fp_rate_weighted']} unweighted {g['fp_rate_unweighted']}; "
                  f"tanzania offdomain {g['tanzania_offdomain']}; hard-positive recall {g['hard_positive_recall']}")
    if args.sweep:
        for n in args.names:
            print(f"\n== sweep {n}")
            print(sweep(args.runs / n).to_string(index=False))


if __name__ == "__main__":
    main()
