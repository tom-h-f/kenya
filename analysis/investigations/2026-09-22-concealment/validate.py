"""Do the concealment features separate operation groups from matched controls?

Operations: X's attributed IRA (2018, 3,613 accounts) and Iran (2018, 770)
releases, the two per-campaign user files that survive on the Internet Archive,
plus the combined 21-country archive drawn at random across campaigns.

Controls: accounts from this project's Kenyan corpus, matched to each operation
member on creation MONTH. Matching holds era fixed, so a feature can only
separate the two by what happens inside a month. Using Kenyan accounts as
controls for a classifier is forbidden in this repo (era and language make the
classes trivially separable); month matching removes the era half of that, and
the features are language-free except bio duplication, whose caveat is stated
in findings.md. The Kenyan pool will contain some coordinated accounts, which
biases every effect toward zero.

Keep rule, fixed before the run: Cohen's d >= 0.5 AND AUC >= 0.70 on BOTH
labelled campaigns at group size 25.

    uv run python investigations/2026-09-22-concealment/validate.py \
        --ira PATH --iran PATH --archive PATH --kenya author_history.parquet
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from kma import concealment

GROUP_SIZES = (10, 25, 50)
GROUPS_PER_CELL = 500
Z_GROUPS = 100
KEEP_D = 0.5
KEEP_AUC = 0.70
KEEP_K = 25


def load_ops(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame[frame["userid"] != "userid"]
    created = pd.to_datetime(frame["account_creation_date"], format="mixed", errors="coerce", utc=True)
    out = pd.DataFrame({
        "created_at": created,
        "bio": frame["user_profile_description"].replace("", None),
    })
    return out[out["created_at"] > pd.Timestamp("2006-03-01", tz="UTC")].reset_index(drop=True)


def load_kenya(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["created_at", "bio"])
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    return frame.dropna(subset=["created_at"]).reset_index(drop=True)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else None


def auc(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0 or len(b) == 0:
        return None
    return float(mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)))


def cell(ops: pd.DataFrame, pool: dict, k: int, rng: np.random.Generator) -> dict:
    """Paired draws: an operation group, then its month-matched control."""
    values: dict[str, tuple[list, list]] = {n: ([], []) for n in concealment.FEATURES}
    skipped = 0
    for _ in range(GROUPS_PER_CELL):
        group = ops.iloc[rng.choice(len(ops), size=k, replace=False)]
        control = concealment.matched_group(group, pool, rng)
        if control is None:
            skipped += 1
            continue
        for name, (column, fn) in concealment.FEATURES.items():
            a, b = fn(group[column]), fn(control[column])
            if a is not None and b is not None:
                values[name][0].append(a)
                values[name][1].append(b)
    out = {"k": k, "pairs": GROUPS_PER_CELL - skipped, "skipped_no_month_match": skipped}
    for name, (a, b) in values.items():
        a, b = np.asarray(a), np.asarray(b)
        out[name] = {
            "n": int(len(a)),
            "op_mean": float(a.mean()) if len(a) else None,
            "control_mean": float(b.mean()) if len(b) else None,
            "d": cohens_d(a, b),
            "auc": auc(a, b),
        }
    return out


def z_rates(ops: pd.DataFrame, kenya: pd.DataFrame, pool: dict, k: int, rng) -> dict:
    """What the dossier would report: the share of groups at z > 2 against the
    month-matched null, for operation groups and for random Kenyan groups. The
    second is the false-alarm rate the reader has to expect."""
    out = {}
    for label, source in (("op", ops), ("kenya_random", kenya)):
        hits = {n: [] for n in concealment.FEATURES}
        for i in range(Z_GROUPS):
            group = source.iloc[rng.choice(len(source), size=k, replace=False)]
            scored = concealment.score_group(group, pool, seed=int(rng.integers(1 << 31)))
            for name in concealment.FEATURES:
                z = scored.get(f"{name}_z")
                if z is not None:
                    hits[name].append(z > 2)
        out[label] = {n: (float(np.mean(v)) if v else None, len(v)) for n, v in hits.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ira", type=Path, required=True)
    parser.add_argument("--iran", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--kenya", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    t0 = time.monotonic()
    kenya = load_kenya(args.kenya)
    pool = concealment.month_pool(kenya)
    campaigns = {
        "ira": load_ops(args.ira),
        "iran": load_ops(args.iran),
        "archive_mixed": load_ops(args.archive),
    }
    print(f"kenya pool {len(kenya):,} accounts over {len(pool)} months; "
          + ", ".join(f"{k} {len(v):,}" for k, v in campaigns.items()), flush=True)

    rng = np.random.default_rng(20260922)
    results = {"sizes": {}, "z_rates": {}, "keep": {}}
    for name, ops in campaigns.items():
        results["sizes"][name] = [cell(ops, pool, k, rng) for k in GROUP_SIZES]
        results["z_rates"][name] = z_rates(ops, kenya, pool, KEEP_K, rng)
        print(f"{name} done {time.monotonic() - t0:.0f}s", flush=True)

    for feature in concealment.FEATURES:
        verdicts = []
        for name in ("ira", "iran"):
            row = next(c for c in results["sizes"][name] if c["k"] == KEEP_K)[feature]
            verdicts.append(
                row["d"] is not None and row["d"] >= KEEP_D
                and row["auc"] is not None and row["auc"] >= KEEP_AUC
            )
        results["keep"][feature] = all(verdicts)

    results["elapsed_s"] = round(time.monotonic() - t0)
    results["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
