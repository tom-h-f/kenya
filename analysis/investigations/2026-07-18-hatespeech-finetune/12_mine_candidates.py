# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb>=1",
#     "python-dotenv>=1",
#     "pandas>=2",
#     "pyarrow>=18",
#     "datasketch>=1.6",
#     "scikit-learn>=1.4",
# ]
# ///
"""Mine labelling candidates from the scored 2026 corpus (Plan A1b).

Runs on tac2 only (needs R2 creds from the monorepo .env). Reads
out/corpus_scored.parquet (12_score_corpus.py) and writes
out/label_batch_001.parquet + out/12_mine_report.json.

Strata are assigned in priority order so that every lexicon and NLI-tail row
survives even when the classifier also ranks it highly; `is_lexicon` /
`is_nli_tail` keep the overlap visible. The random control stratum is drawn
from rows no model picked - it is the only honest measure of prevalence and
of the miner's precision, so it is never dropped for space.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

from _common import OUT, SEED

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from kma.db import connect, latest_incitement, latest_posts  # noqa: E402

prep = import_module("00_prep")

TARGETS = {
    "p_hate_top": 1200,
    "p_offensive_top": 800,
    "nli_tail": 150,
    "random_control": 300,
}
NLI_THRESHOLD = 0.9
AUTHOR_CAP = 10
STRATUM_PRIORITY = [
    "lexicon",
    "nli_tail",
    "p_hate_top",
    "p_offensive_top",
    "random_control",
]


def take(pool: pd.DataFrame, mask: pd.Series | None, n: int | None, stratum: str,
         taken: set[str], random: bool = False) -> pd.DataFrame:
    """n rows of pool matching mask that are not already claimed.

    `random=True` samples uniformly instead of taking the head - the NLI tail
    is meant to probe what the classifier misses, so ranking it by p_hate
    would defeat the point.
    """
    keep = ~pool["platform_post_id"].isin(taken)
    if mask is not None:
        keep &= mask.reindex(pool.index).fillna(False)
    rows = pool[keep]
    if n is not None:
        rows = rows.sample(min(n, len(rows)), random_state=SEED) if random \
            else rows.head(n)
    taken.update(rows["platform_post_id"])
    return rows.assign(stratum=stratum)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink every target for a smoke run")
    ap.add_argument("--out", default="label_batch_001.parquet")
    args = ap.parse_args()

    targets = {k: max(1, int(v * args.scale)) for k, v in TARGETS.items()}
    counts: dict[str, object] = {"targets": targets}

    scored = pd.read_parquet(OUT / "corpus_scored.parquet")
    counts["scored_pool"] = len(scored)

    con = connect()
    meta = latest_posts(con, "x").df()[
        ["platform_post_id", "author_handle", "created_at"]
    ]
    incite = latest_incitement(con, "x").df()[
        ["platform_post_id", "lexicon_hits", "dehumanisation_score",
         "violence_call_score", "othering_score", "political_criticism_score"]
    ]

    pool = scored.merge(meta, on="platform_post_id", how="left").merge(
        incite, on="platform_post_id", how="left"
    )
    counts["pool_with_meta"] = len(pool)
    counts["pool_missing_author"] = int(pool["author_handle"].isna().sum())
    counts["pool_with_incitement"] = int(pool["dehumanisation_score"].notna().sum())

    pool["is_lexicon"] = pool["lexicon_hits"].map(
        lambda h: hasattr(h, "__len__") and len(h) > 0
    )
    nli_max = pool[
        ["dehumanisation_score", "violence_call_score", "othering_score"]
    ].max(axis=1)
    pool["is_nli_tail"] = (nli_max >= NLI_THRESHOLD) & (
        nli_max > pool["political_criticism_score"]
    )
    counts["lexicon_in_pool"] = int(pool["is_lexicon"].sum())
    counts["nli_tail_in_pool"] = int(pool["is_nli_tail"].sum())

    pool = pool.sort_values("p_hate", ascending=False).reset_index(drop=True)
    taken: set[str] = set()
    parts = [
        take(pool, pool["is_lexicon"], None, "lexicon", taken),
        take(pool, pool["is_nli_tail"].fillna(False),
             targets["nli_tail"], "nli_tail", taken, random=True),
        take(pool, None, targets["p_hate_top"], "p_hate_top", taken),
        take(pool.sort_values("p_offensive", ascending=False), None,
             targets["p_offensive_top"], "p_offensive_top", taken),
    ]
    rest = pool[~pool["platform_post_id"].isin(taken)]
    parts.append(
        rest.sample(min(targets["random_control"], len(rest)), random_state=SEED)
        .assign(stratum="random_control")
    )

    cand = pd.concat(parts, ignore_index=True)
    counts["selected_raw"] = len(cand)

    cand["_rank"] = cand["stratum"].map(STRATUM_PRIORITY.index)
    cand = cand.sort_values(["_rank", "p_hate"], ascending=[True, False])

    clusters = prep.near_dup_clusters(cand["text"].reset_index(drop=True))
    cand = cand.reset_index(drop=True).assign(_cluster=clusters.values)
    cand = cand.drop_duplicates("_cluster", keep="first")
    counts["after_near_dedupe"] = len(cand)

    cand = cand.groupby("author_handle", dropna=False, sort=False).head(AUTHOR_CAP)
    counts["after_author_cap"] = len(cand)

    cand = cand.rename(columns={"platform_post_id": "post_id"})[
        ["post_id", "author_handle", "created_at", "text", "stratum",
         "p_hate", "p_offensive", "is_lexicon", "is_nli_tail"]
    ].reset_index(drop=True)

    counts["per_stratum"] = cand["stratum"].value_counts().to_dict()
    counts["authors"] = int(cand["author_handle"].nunique())
    counts["max_per_author"] = int(cand["author_handle"].value_counts().max())
    counts["p_hate_min_selected"] = float(
        cand.loc[cand["stratum"] == "p_hate_top", "p_hate"].min()
    )

    cand.to_parquet(OUT / args.out, index=False)
    (OUT / "12_mine_report.json").write_text(json.dumps(counts, indent=2, default=str))
    print(json.dumps(counts, indent=2, default=str))
    print(f"wrote {OUT / args.out} ({len(cand)} rows)")


if __name__ == "__main__":
    main()
