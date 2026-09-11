"""Unblind A2's size-matched run: what the reader called each method's cases.

    cd analysis && uv run python investigations/2026-09-11-textsim-sensitivity/04_a2_report.py \\
        investigations/2026-09-11-textsim-sensitivity/out/2026-09-05-promotion-off/a2_matched/a2_key.csv \\
        --verdicts-key coordination/platform=x/kind=verdicts/dt=.../run=....parquet

The key never went to the reader. It is joined here, after every verdict was
written, and the script refuses to report if any case lacks a verdict.

Definitions, stated because the 2026-09-09 table's categories cannot be
recovered - those verdicts were never persisted:
  political         the reader chose political_campaign or influence_operation
  political, high   political at high confidence - the nearest defined
                    equivalent of 2026-09-09's "strongly coordinated-political"

The size-band breakdown is the point of the exercise: the 2026-09-09 sample set
v1's largest clusters against v2's mid-sized groups, and engagement pods are
the large ones.
"""

import argparse
from pathlib import Path

import pandas as pd

from kma.db import BUCKET, connect

POLITICAL = {"political_campaign", "influence_operation"}
BANDS = [(1, 10, "small (<=10)"), (11, 99, "mid (11-99)"), (100, 10**9, "large (100+)")]


def band(size: int) -> str:
    return next(label for low, high, label in BANDS if low <= size <= high)


def summarise(frame: pd.DataFrame) -> dict:
    political = frame["cluster_type"].isin(POLITICAL)
    return {
        "cases": len(frame),
        "political": f"{int(political.sum())} ({political.mean():.0%})",
        "political_high": int((political & (frame["confidence"] == "high")).sum()),
        "kenya_relevant": f"{int(frame['kenya_relevant'].fillna(False).astype(bool).sum())}",
        "types": frame["cluster_type"].value_counts().to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("key", type=Path)
    ap.add_argument("--verdicts-key", required=True, help="the verdicts_key the Modal run returned")
    args = ap.parse_args()

    key = pd.read_csv(args.key)
    verdicts = connect().sql(f"SELECT * FROM read_parquet('r2://{BUCKET}/{args.verdicts_key}')").df()
    merged = key.merge(verdicts, left_on="case_id", right_on="cluster_id", how="left")
    missing = merged.loc[merged["cluster_type"].isna(), "case_id"].tolist()
    if missing:
        raise SystemExit(f"cases without a verdict: {missing} - not reporting a partial unblinding")

    merged["band"] = merged["size"].map(band)
    print("== by method ==")
    for method, frame in merged.groupby("method"):
        print(f"  {method}: {summarise(frame)}")
    print("\n== by method and size band ==")
    for (method, size_band), frame in merged.groupby(["method", "band"]):
        print(f"  {method:3} {size_band:13} {summarise(frame)}")

    out = args.key.with_name("a2_unblinded.csv")
    merged.drop(columns=["cluster_id"]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
