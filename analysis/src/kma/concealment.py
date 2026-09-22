"""Group-level signs of accounts set up together, each read against a null.

Co-action separates "together" from "not together". It cannot separate an open
campaign from an operation, because campaigners also act together. What
separates them is concealment, and the blind reader's rationale for every
non-operation verdict so far has been the same: real-looking aged accounts,
diverse avatars, no concealment of origin
(`docs/plans/2026-09-15-finding-campaigns.md`).

Every feature here is a property of a GROUP, never of a person: "were these
accounts provisioned together" is a question about the set. None labels an
account, and none is a verdict - they are evidence for a reader, beside the
co-action that surfaced the group.

Every feature is reported as its raw value AND as a z against a null of
same-size groups whose members are MATCHED ON CREATION MONTH. The matching is
the point. A group's accounts can sit close in creation date for a dull reason:
they joined in the same era, as any community formed around one event does. The
month-matched null holds era fixed, so what survives is clumping finer than a
month - accounts created on the same days - which is what a batch of
provisioned accounts leaves and a community does not.

WHAT WAS VALIDATED AND WHAT WAS NOT, measured 2026-09-22 (see
`analysis/investigations/2026-09-22-concealment/findings.md`):

- `creation_burst` and `bio_duplication` were tested on X's attributed IRA and
  Iran operations against month-matched groups of Kenyan-corpus accounts.
  Kept or dropped per the effect sizes recorded there.
- Handle/bio change and avatar reuse CANNOT be validated: the IO archive holds
  one profile snapshot per account and no image field. They are measured on the
  Kenya corpus for feasibility only and are not scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Two accounts created this many days apart or closer count as a pair. One day
# either side absorbs the UTC boundary: an operator working one evening in
# Nairobi straddles two UTC dates.
BURST_DAYS = 1

# Null draws per group. The z is reported, not tested against a cut, so this
# only has to make the null's mean and spread stable.
NULL_DRAWS = 200

# A bio shorter than this after normalisation is left out of duplication:
# "Kenyan", a single emoji or a lone hashtag repeats across strangers.
MIN_BIO_CHARS = 12


def creation_burst(created: pd.Series, days: int = BURST_DAYS) -> float | None:
    """Largest share of the group created inside any span of `days` + 1 days.

    Not "share with a neighbour within a day": that saturates by chance. Twenty
    accounts spread at random over one 31-day month mostly have a neighbour a
    day away, so a genuine one-day batch scored only z 2.3 against its null in
    the tests. The largest cohort does not saturate - chance puts three or so of
    twenty on one span, a batch puts all of them.

    None, not 0, below two dated accounts: a group of one has no cohort, and a
    zero would read as a measured absence of clumping."""
    stamps = pd.to_datetime(created, utc=True, errors="coerce").dropna()
    if len(stamps) < 2:
        return None
    day = np.sort((stamps.dt.tz_localize(None).dt.floor("D") - pd.Timestamp("1970-01-01")).dt.days.to_numpy())
    ends = np.searchsorted(day, day + days, side="right")
    return float((ends - np.arange(len(day))).max() / len(day))


def normalise_bio(bio: object) -> str | None:
    if not isinstance(bio, str):
        return None
    text = " ".join(bio.lower().split())
    return text if len(text) >= MIN_BIO_CHARS else None


def bio_duplication(bios: pd.Series) -> float | None:
    """Share of accounts with a usable bio whose normalised bio another member
    also carries, verbatim.

    Verbatim on purpose. A near-duplicate measure would need a threshold, and a
    campaign's supporters paraphrase the same slogan honestly; a copied bio is
    what a batch of provisioned accounts leaves. None below two usable bios."""
    usable = bios.map(normalise_bio).dropna()
    if len(usable) < 2:
        return None
    counts = usable.map(usable.value_counts())
    return float((counts > 1).mean())


FEATURES = {
    "creation_burst": ("created_at", creation_burst),
    "bio_duplication": ("bio", bio_duplication),
}


def month_pool(population: pd.DataFrame) -> dict:
    """Index a reference population by creation month, for matched draws."""
    month = pd.to_datetime(population["created_at"], utc=True, errors="coerce")
    frame = population.assign(_month=month.dt.strftime("%Y-%m"))
    frame = frame[frame["_month"].notna()]
    return {m: g.drop(columns="_month").reset_index(drop=True) for m, g in frame.groupby("_month")}


def matched_group(group: pd.DataFrame, pool: dict, rng: np.random.Generator) -> pd.DataFrame | None:
    """One control group: for each member, a random reference account created in
    the same calendar month. None if any member's month has no reference
    account, rather than silently shrinking the group - size moves every
    feature here."""
    months = pd.to_datetime(group["created_at"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    picks = []
    for month, n in months.dropna().value_counts().items():
        candidates = pool.get(month)
        if candidates is None or len(candidates) == 0:
            return None
        idx = rng.integers(0, len(candidates), size=n)
        picks.append(candidates.iloc[idx])
    if not picks:
        return None
    return pd.concat(picks, ignore_index=True)


def score_group(
    group: pd.DataFrame,
    pool: dict,
    draws: int = NULL_DRAWS,
    seed: int = 0,
) -> dict:
    """Each feature's raw value, its month-matched null mean and sd, and z.

    A z is None when the null is degenerate (sd 0) or undefined: an all-zero
    null with a zero observation is no evidence either way, and dividing by a
    zero spread would manufacture an infinite one."""
    rng = np.random.default_rng(seed)
    nulls: dict[str, list[float]] = {name: [] for name in FEATURES}
    for _ in range(draws):
        control = matched_group(group, pool, rng)
        if control is None:
            break
        for name, (column, fn) in FEATURES.items():
            if column in control.columns:
                value = fn(control[column])
                if value is not None:
                    nulls[name].append(value)
    out: dict = {}
    for name, (column, fn) in FEATURES.items():
        observed = fn(group[column]) if column in group.columns else None
        values = np.asarray(nulls[name], dtype=float)
        mean = float(values.mean()) if len(values) else None
        sd = float(values.std(ddof=1)) if len(values) > 1 else None
        z = (observed - mean) / sd if observed is not None and sd else None
        out[name] = observed
        out[f"{name}_null_mean"] = mean
        out[f"{name}_z"] = z
    out["null_draws"] = len(nulls["creation_burst"])
    return out
