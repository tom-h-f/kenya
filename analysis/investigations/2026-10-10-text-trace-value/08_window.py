"""Q6: how much of the text trace's listed value survives a short text window.

    cd analysis && uv run python investigations/2026-10-10-text-trace-value/08_window.py --tag current

From 03_arms.py's listings. Value is measured two ways: the accounts that the
full text trace adds to the listing (listed with text, absent from the no-text
listing) and the "text-dependent" accounts (listed in a community that falls
apart without its text-only edges). For each short-window arm: how many of
those it still lists, and the share of text edges / text-only accounts it holds.
a<N> text edges whose later post is in the last N days; c<N> pairs with both
posts collected in the last N days (what a real N-day text window builds).
"""

import argparse
import json

import pandas as pd

from common import WINDOW, wdir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default=WINDOW)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    base = wdir(args.window)
    arms = base / f"arms_{args.tag}"
    table = pd.read_csv(arms / "arms.csv").set_index("arm")
    lst = lambda a: set(pd.read_parquet(arms / f"listed_{a}.parquet")["user_id"])
    full, none = lst("full"), lst("none")
    added = full - none
    ct = pd.read_csv(arms / "listed_communities_text_dependence.csv") if (
        arms / "listed_communities_text_dependence.csv").exists() else None
    fl = pd.read_parquet(arms / "listed_full.parquet")
    pure = set(fl.loc[fl["community"].isin(ct.loc[ct["pure_text"], "community"]), "user_id"]) if ct is not None else set()
    rows = []
    full_edges = table.loc["full", "text_edges_after_floor"]
    for a in table.index:
        if a in ("none", "full"):
            continue
        l = lst(a)
        rows.append({
            "arm": a, "text_edges": int(table.loc[a, "text_edges_after_floor"]),
            "share_of_full_text_edges": round(table.loc[a, "text_edges_after_floor"] / full_edges, 3),
            "listed": len(l),
            "kept_of_full_added": len(l & added), "full_added": len(added),
            "kept_of_text_dependent": len(l & pure), "text_dependent": len(pure),
            "listed_jaccard_vs_full": round(len(l & full) / len(l | full), 3),
            "listed_jaccard_vs_none": round(len(l & none) / len(l | none), 3),
        })
    out = pd.DataFrame(rows)
    out.to_csv(arms / "window.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
