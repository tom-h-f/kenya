"""A blind sheet that judges the MODEL, the way B2 judged the gate.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/10_human_check.py --n 100

The first human pass validated the LABELLER (Tom against claude-opus-5, 90 of
100, kappa 0.83). This validates what shipped: 100 fresh posts none of which are
in the training data or the earlier sheets, stratified by what the model says so
both of its error directions get looked at, and blind - the sheet carries no
score and no bucket.

    ... --score out/human_check_filled.csv     # after the labels are filled in

scores it: precision and recall of the model and of the keyword gate on the same
posts, corpus-weighted by the gate's buckets as `measure_eval.score` is.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kma import measure, relevance
from kma.db import connect, posts_source

HERE = Path(__file__).parent
BANDS = {"model says kenya": (0.5, 1.01), "model says off": (-0.01, 0.5)}


def draw(con, n: int, seed: int, exclude: set[str]) -> pd.DataFrame:
    scored = con.sql(f"""
        WITH s AS ({relevance.latest_scores_cte()})
        SELECT CAST(p.platform_post_id AS VARCHAR) AS post_id, p.text, s.p_kenya
        FROM (SELECT platform_post_id, text FROM {posts_source('x')}
              WHERE text IS NOT NULL AND length(trim(text)) > 0
              QUALIFY row_number() OVER (
                  PARTITION BY platform_post_id ORDER BY collected_at DESC) = 1) p
        JOIN s ON s.platform_post_id = p.platform_post_id
        USING SAMPLE reservoir({int(n) * 40} ROWS) REPEATABLE ({int(seed)})
    """).df()
    scored = scored[~scored["post_id"].isin(exclude)]
    rng = np.random.default_rng(seed)
    parts = []
    for band, (low, high) in BANDS.items():
        rows = scored[(scored["p_kenya"] >= low) & (scored["p_kenya"] < high)]
        parts.append(rows.sample(min(n // len(BANDS), len(rows)),
                                 random_state=int(rng.integers(1 << 31))).assign(band=band))
    sheet = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sheet


def score(path: Path, key: Path) -> None:
    filled = pd.read_csv(path)
    if "p_kenya" not in filled:
        # The sheet is blind on purpose, so the scores come back from the key.
        filled = filled.merge(pd.read_csv(key)[["post_id", "p_kenya"]], on="post_id")
    filled = filled[filled["label"].isin(("kenya", "offdomain"))].reset_index(drop=True)
    filled["bucket"] = filled["text"].map(measure.domain_bucket)
    actual = (filled["label"] == "kenya").to_numpy()
    shares = filled["bucket"].value_counts(normalize=True)
    weight = (filled["bucket"].map(shares).to_numpy()
              / filled.groupby("bucket")["bucket"].transform("size").to_numpy())
    for name, predicted in (("model", (filled["p_kenya"] >= relevance.THRESHOLD).to_numpy()),
                            ("keyword gate", (filled["bucket"] == "kenya").to_numpy())):
        tp = weight[predicted & actual].sum()
        fp = weight[predicted & ~actual].sum()
        fn = weight[~predicted & actual].sum()
        print(f"{name:>13}: precision {tp / (tp + fp):.3f} recall {tp / (tp + fn):.3f} "
              f"(n={len(filled)}, {int(actual.sum())} Kenyan)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--exclude", type=Path, action="append", default=[],
                    help="csv/parquet of post_id already labelled or trained on")
    ap.add_argument("--score", type=Path, help="score a filled-in sheet instead of drawing one")
    args = ap.parse_args()

    if args.score:
        return score(args.score, HERE / "out" / "human_check_key.csv")

    seen: set[str] = set()
    for path in args.exclude:
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        column = "post_id" if "post_id" in frame else "platform_post_id"
        seen |= set(frame[column].astype(str))

    sheet = draw(connect(), args.n, args.seed, seen)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    # Two files: the blind one to label, and the key with the scores in it.
    sheet.assign(label="")[["post_id", "text", "label"]].to_csv(out / "human_check_sheet.csv", index=False)
    sheet.to_csv(out / "human_check_key.csv", index=False)
    print(f"{len(sheet)} posts ({sheet['band'].value_counts().to_dict()}), "
          f"{len(seen):,} excluded -> {out / 'human_check_sheet.csv'}")
    print("Label each row kenya / offdomain / unclear, then:")
    print(f"  uv run python {Path(__file__).name} --score {out / 'human_check_filled.csv'}")


if __name__ == "__main__":
    main()
