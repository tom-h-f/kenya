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

MEASURED 2026-09-22, AND NOTHING PASSED. Both testable features were run
against X's attributed IRA (3,836 accounts) and Iran (770) releases, with
controls drawn from 400,000 accounts of this corpus matched on creation month,
500 group pairs per cell (`investigations/2026-09-22-concealment/validate.py`,
`out/results.json`). The keep rule, fixed before the run, was Cohen's d >= 0.5
AND AUC >= 0.70 on both campaigns at group size 25:

    feature           campaign  k=25 d   k=25 AUC   k=50 d   k=50 AUC
    creation_burst    IRA       0.522    0.616      0.846    0.707
    creation_burst    Iran      0.428    0.601      0.700    0.653
    bio_duplication   IRA      -0.074    0.496     -0.151    0.486
    bio_duplication   Iran      0.031    0.500      0.184    0.515

So `KEPT` is empty and no dossier carries a concealment section yet.

`creation_burst` points the right way every time and strengthens with group
size, so it is a real but weak effect, not noise. It is not strong enough to
put in front of a reader: at k=25 an operation group clears z 2 on 16% of draws
(IRA) against 5% for random Kenyan groups, which is three false alarms for
every eight hits.

`bio_duplication` is refuted, and in the interesting direction: matched Kenyan
controls duplicate bios slightly MORE than IRA accounts do. An operation buys
personas; ordinary accounts leave the same stock phrases. Do not rebuild it -
this is the self-amplification result again (OBJECTIVES A3).

Two limits on the above, both of which would raise the effect rather than lower
it, so re-testing is worthwhile before the feature is abandoned:

- Groups are drawn at RANDOM from a campaign, so a 73-account batch created on
  2014-05-30 rarely contributes two members to one group of 25. Real detected
  communities are not random subsets; if co-action tracks provisioning, they
  are batch-aligned and would score far higher. Testing that needs operation
  groups defined by co-action, which needs `ioa_tweets.csv` (113.7 GB) and so
  needs Modal.
- The control pool is this corpus, which contains coordinated accounts of its
  own, biasing every effect toward zero.

CANNOT BE VALIDATED AT ALL, and therefore not scored: handle change, bio change
and avatar reuse. Every operation dataset available carries one profile
snapshot per account and no avatar field. Their feasibility on this corpus was
measured the same day over 90 days of author snapshots (10,946,134 snapshots,
2,703,056 accounts): 9,585 accounts changed handle, 59,381 changed bio, 53,255
changed avatar - and avatar REUSE is not measurable at all, because 0 accounts
share a non-default `profile_image_url` (X serves a per-account URL, so
matching images would need the image bytes, not the link).
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

# Features that passed the keep rule on the IO archive and reach the dossier.
# A feature that did not stays computable here for re-validation, and stays out
# of what a reader is shown.
KEPT: tuple[str, ...] = ()

# Size of the reference sample the month-matched null draws from. A null draw
# needs a handful of accounts per creation month; 100,000 accounts over ~200
# months leaves hundreds in all but the earliest.
POOL_SIZE = 100_000


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


def reference_pool(con, platform: str = "x", size: int = POOL_SIZE, seed: int = 0) -> dict:
    """A month-indexed random sample of the corpus's accounts, latest profile each.

    Sampled from `latest_authors`' shape - one row per account - so an account
    collected many times is not many times more likely to be drawn. Built once
    per dossier run; every cluster draws its null from the same pool."""
    from kma.db import authors_source

    frame = con.sql(
        f"""
        WITH la AS (
            SELECT platform_user_id,
                   arg_max(struct_pack(created_at := created_at, bio := bio),
                           collected_at) AS r
            FROM {authors_source(platform)}
            GROUP BY platform_user_id
        )
        SELECT r.created_at AS created_at, r.bio AS bio FROM la
        USING SAMPLE {int(size)} ROWS (reservoir, {int(seed)})
        """
    ).df()
    return month_pool(frame)
