"""Compare text-trace variants' top 500 against the current 0.85 ranking.

    cd analysis && uv run python investigations/2026-09-11-textsim-sensitivity/10_compare.py \\
        out/<export-dir>/sweep_min0.75/top500_t0.85.parquet \\
        lexical-floor=out/<export-dir>/sweep_lex0.10/top500_t0.85.parquet \\
        bge-m3=out/<export-dir>__bge-m3/sweep_min0.75/top500_t0.80.parquet \\
        --a2 out/a2_matched

For each variant: how much of the current top 500 it keeps, the Kenya share
and activity of what it ranks, and - with `--a2` - how many members of each
A2 v2 group survive, since those groups are what the size-matched run read.
"""

import argparse
from pathlib import Path

import pandas as pd


def describe(top: pd.DataFrame) -> dict:
    return {
        "kenya_share mean": round(top["kenya_share"].mean(), 3),
        "kenya_share median": round(top["kenya_share"].median(), 3),
        "n_posts median": int(top["n_posts"].median()),
        "one-post accounts": int((top["n_posts"] == 1).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline", type=Path)
    ap.add_argument("variants", nargs="+", help="name=path to a top500 parquet")
    ap.add_argument("--a2", type=Path, help="the a2_matched directory, for per-group survival")
    args = ap.parse_args()

    tops = {"current 0.85": pd.read_parquet(args.baseline)}
    for spec in args.variants:
        name, path = spec.split("=", 1)
        tops[name] = pd.read_parquet(path)
    base = set(tops["current 0.85"]["user_id"].astype(str))

    rows = []
    for name, top in tops.items():
        ids = set(top["user_id"].astype(str))
        rows.append({"ranking": name, "kept of current 500": len(ids & base),
                     "Jaccard": round(len(ids & base) / len(ids | base), 3), **describe(top)})
    print(pd.DataFrame(rows).to_string(index=False))

    if args.a2:
        sample = pd.read_parquet(args.a2 / "a2_sample.parquet")
        key = pd.read_csv(args.a2 / "a2_key.csv")
        groups = key[key["method"] == "v2"].sort_values("size", ascending=False)
        table = []
        for case in groups.itertuples():
            members = set(sample.loc[sample["cluster_id"] == case.case_id, "author_id"].astype(str))
            table.append({"case": case.case_id, "size": case.size,
                          **{name: len(members & set(top["user_id"].astype(str))) for name, top in tops.items()}})
        print("\nA2 v2 group members still in each top 500:")
        print(pd.DataFrame(table).to_string(index=False))


if __name__ == "__main__":
    main()
