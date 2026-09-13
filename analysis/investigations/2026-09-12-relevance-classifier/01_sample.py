"""Draw posts to label for a learned relevance classifier.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/01_sample.py \\
        --exclude /Users/tom/Code/Misc/kenya-monitor-2027/analysis/out/measure_sample.csv --n 3000

`kma.measure.domain_bucket` is a keyword gate: precision 0.971, human-confirmed,
but recall 0.61-0.66, and it misses plain Kenyan politics - Orengo, Karua,
Sifuna, #RutoMustGo. 20 of its 21 misses in the B2 sample sat in the
`ambiguous` bucket, which is 74.7% of the corpus. So the budget goes mostly
there.

Same population as `measure_eval.draw_sample` (non-empty text, the pinned
snapshot), stratified by the gate's bucket with `--ambiguous-share` of the rows
from `ambiguous` and the rest split between `kenya` and `offdomain`. Every post
in `--exclude` - the B2 sample, including the 100 human-labelled posts - is
left out, so the evaluation set never leaks into training.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kma import bench, coord2, measure
from kma.db import connect

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default="2026-09-05-promotion-off")
    ap.add_argument("--exclude", type=Path, action="append", default=[], help="csv/parquet with post_id")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--ambiguous-share", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    con = connect()
    view = coord2.posts_view(bench.pinned_source(bench.load(args.snapshot, con=con), "posts"))
    posts = con.sql(
        f"""SELECT post_id, user_id, text FROM ({view})
            WHERE text IS NOT NULL AND length(trim(text)) > 0
            ORDER BY post_id"""
    ).df()
    posts["post_id"] = posts["post_id"].astype(str)
    posts["bucket"] = [measure.domain_bucket(t) for t in posts["text"]]
    shares = posts["bucket"].value_counts(normalize=True)

    held_out = set()
    for path in args.exclude:
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        held_out |= set(frame["post_id"].astype(str))
    pool = posts[~posts["post_id"].isin(held_out)]
    print(f"{len(posts):,} posts; {len(posts) - len(pool):,} held out; bucket shares {shares.round(3).to_dict()}")

    quota = {"ambiguous": round(args.n * args.ambiguous_share)}
    quota["kenya"] = (args.n - quota["ambiguous"]) // 2
    quota["offdomain"] = args.n - quota["ambiguous"] - quota["kenya"]
    rng = np.random.default_rng(args.seed)
    parts = [pool[pool["bucket"] == b].sample(min(k, int((pool["bucket"] == b).sum())),
                                              random_state=int(rng.integers(1 << 31)))
             .assign(stratum_share=float(shares[b])) for b, k in quota.items()]
    sample = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    sample.to_parquet(out / "to_label.parquet")
    print(f"{len(sample):,} posts to label {sample['bucket'].value_counts().to_dict()} -> {out / 'to_label.parquet'}")


if __name__ == "__main__":
    main()
