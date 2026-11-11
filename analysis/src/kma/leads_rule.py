"""The deterministic leads verdict: which changed communities reach a phone.

Stands where `adjudicate.judge_all` stood, with no model call and no key. It
reads the daily pass's own `daily_communities` row for each community and asks
one question: is this group held together by something other than retweeting?

A group is FLAGGED when it passed the Kenya gate (`kept`) and either

- the template detector labelled it (`template_label`), or
- at least `NON_RETWEET_SHARE` of the trace labels on its internal edges
  (`coord2_communities` `share_*`) come from the non-retweet traces: similar
  wording, hashtag sequences, shared URLs and fast retweets.

Co-retweeting alone is how every fandom, party base and news audience looks,
and it is 80-100% of the large groups the daily pass keeps. The other four
traces are the ones Kenya's documented hashtag and copy-paste operations leave.

A flag ALERTS only when it is news: the group is new, or its best-matching
predecessor in the previous comparable run was not flagged, AND no group
alerted in the last `MEMORY_DAYS` shares `SEEN_JACCARD` of its members. The
memory is what stops a small group that drops under the size floor and comes
back, or a large one hovering at the threshold, from alerting every few days:
without it the last 20 runs give 29 alerts.

`NON_RETWEET_SHARE` is a starting point, not a measurement: it was chosen
2026-11-11 from the last 20 daily runs, where it separates the small
text-similarity groups from the retweet networks, and nobody has labelled
whether those groups are campaigns. TRIAGE, NEVER A VERDICT.
"""

from __future__ import annotations

import pandas as pd

RULE_VERSION = "rule-2026-11-11"
ADJUDICATOR = "rule"

NON_RETWEET_SHARE = 0.5
NON_RETWEET_TRACES = (
    "share_text_similarity",
    "share_hashtag_sequence",
    "share_co_url",
    "share_fast_retweet",
)

MEMORY_DAYS = 14
SEEN_JACCARD = 0.5

# Alerts on groups this size or larger push to a phone; smaller ones land in
# IRM without a push. Over the 20 runs to 2026-11-11 the memory-applied rule
# gave 23 alerts: 17 from groups of 4-7 accounts, 6 from 8 or more.
PUSH_MIN_SIZE = 8
LARGE = "large"
SMALL = "small"


def band(size: int) -> str:
    return LARGE if size >= PUSH_MIN_SIZE else SMALL

FLAGGED = "rule_flagged"
CLEAR = "rule_clear"


def seen_before(members: frozenset, recent: list[frozenset]) -> bool:
    for old in recent:
        shared = len(members & old)
        if shared and shared / len(members | old) >= SEEN_JACCARD:
            return True
    return False


def signals(communities: pd.DataFrame) -> pd.DataFrame:
    """Per community: `community`, `kept`, `non_retweet_share`, `template`,
    `flagged`, `kenya_model`, `signals` (a short reason string)."""
    if communities.empty:
        return pd.DataFrame(columns=["community", "kept", "non_retweet_share", "template",
                                     "flagged", "kenya_model", "signals"])
    frame = communities.copy()
    present = [c for c in NON_RETWEET_TRACES if c in frame.columns]
    share = frame[present].fillna(0).sum(axis=1) if present else pd.Series(0.0, index=frame.index)
    template = (frame["template_label"].notna() if "template_label" in frame.columns
                else pd.Series(False, index=frame.index))
    kept = frame["kept"].fillna(False).astype(bool)
    flagged = kept & (template | (share >= NON_RETWEET_SHARE))

    def reason(i) -> str:
        parts = []
        if template[i]:
            parts.append(f"template:{frame.at[i, 'template_label']}")
        for c in present:
            v = frame.at[i, c]
            if pd.notna(v) and v >= 0.05:
                parts.append(f"{c.removeprefix('share_')}={v:.2f}")
        return ", ".join(parts)

    return pd.DataFrame({
        "community": frame["community"].astype(int),
        "kept": kept,
        "non_retweet_share": share.round(4),
        "template": template,
        "flagged": flagged,
        "kenya_model": frame.get("kenya_model"),
        "signals": [reason(i) for i in frame.index],
    })


def judge(
    diff: pd.DataFrame,
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
    *,
    members: dict | None = None,
    recent: list[frozenset] = (),
) -> pd.DataFrame:
    """One row per diffed community: its signals, whether it is flagged, and
    whether that flag is news (`alert`).

    `diff` is `leads.diff_communities` output (`group`, `previous`, `status`,
    `candidate_id`, ...); `current` and `previous` are the two runs'
    `daily_communities` tables. `members` maps a current community to its
    member ids and `recent` holds the member sets alerted in the last
    `MEMORY_DAYS`; without `members` the memory is not applied."""
    cols = list(diff.columns) + ["kept", "non_retweet_share", "template", "flagged",
                                 "kenya_model", "signals", "previously_flagged",
                                 "seen_before", "alert", "band"]
    if diff.empty:
        return pd.DataFrame(columns=cols)
    cur = signals(current).set_index("community")
    prev = signals(previous).set_index("community") if previous is not None else None

    out = diff.copy()
    group = out["group"].astype(int)
    for c in ("kept", "non_retweet_share", "template", "flagged", "kenya_model", "signals"):
        out[c] = group.map(cur[c]).values if len(cur) else None
    out["kept"] = out["kept"].fillna(False).astype(bool)
    out["template"] = out["template"].fillna(False).astype(bool)
    out["flagged"] = out["flagged"].fillna(False).astype(bool)

    def was_flagged(pid) -> bool:
        if prev is None or pid is None or pd.isna(pid):
            return False
        pid = int(pid)
        return bool(prev.at[pid, "flagged"]) if pid in prev.index else False

    out["previously_flagged"] = [was_flagged(p) for p in out["previous"]]
    out["seen_before"] = [
        bool(members) and seen_before(members.get(g, frozenset()), list(recent))
        for g in group
    ]
    out["alert"] = (out["flagged"] & ~out["seen_before"]
                    & ((out["status"] == "new") | ~out["previously_flagged"]))
    out["band"] = out["size"].astype(int).map(band)
    return out[cols]
