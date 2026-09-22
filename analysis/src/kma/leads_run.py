"""One leads pass: diff the latest runs, judge what changed, persist it.

    uv run kma-leads                          # dry run: prints the queue
    uv run kma-leads --source v1              # the same, on v1's clusters
    uv run kma-leads show <run_id> <candidate_id>
    uv run kma-leads --source v1 --judge --max-judged 5 --persist   # local, needs a key

The judging pass itself runs on Modal (`modal_leads.py`), because the reader
needs `ANTHROPIC_API_KEY` and that key stays off tf1 and pi0. This module is
everything around the reader - which runs to compare, who belongs to each
candidate, and where the verdicts and dossiers go - so it can be tested and
dry-run anywhere with R2 access.

## Output, under `leads/platform=x/`

- `kind=queue/dt=/run=<run_id>.parquet` - every candidate the diff produced,
  queued or skipped, with its match statistics. What was skipped is recorded,
  not just logged.
- `kind=dossiers/dt=/run=<run_id>/<candidate_id>.parquet` - the packet, as
  JSON, and the exact prompt the reader saw.
- `kind=verdicts/dt=/run=<run_id>.parquet` - one row per judged candidate,
  with `worth_a_look` and the dossier key. The tf1 poller reads only this.

## The first run judges nothing

With no earlier comparable listing, every community is "new" and ten of them
would be judged and alerted as if they had just appeared. A pass without a
predecessor records its queue as a baseline instead. Comparable means the same
detector parameters: a listing built at a different `min_entities` or text
threshold is a different object, and diffing across that change would report
the parameter change as a wave of new campaigns.

TRIAGE, NEVER A VERDICT, and PRIVATE: dossiers carry handles and post text,
so nothing under `leads/` may reach `DASHBOARD_BUCKET`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb
import pandas as pd

from kma import leads
from kma.db import BUCKET, connect

log = logging.getLogger("kma")

PLATFORM = "x"
COMMUNITIES_PREFIX = f"coord2/platform={PLATFORM}/kind=communities"
V1_PREFIX = f"coordination/platform={PLATFORM}/kind=clusters"
TRENDS_PREFIX = f"trend_candidates/platform={PLATFORM}"
LEADS_PREFIX = f"leads/platform={PLATFORM}"

# Parameters that make two listings comparable. Only those present on a run
# are compared, so the v1 stand-in, which has none, compares run to run.
SIGNATURE = ("min_entities", "text_threshold", "text_overlap", "window_days")

# A synthetic canary's member ids look like `canary:<canary_id>:<n>`. The
# contract with workstream C is in docs/analysis/leads.md.
CANARY_PREFIX = "canary:"

# Offset for trend candidates' dossier ids, which must be ints and must not
# collide with a community label in the same run.
TREND_ID_BASE = 10_000_000

# Posts that can name a trend tag's members: the tag's own census and the
# control arm it was discovered in, over the window discovery looks at.
TREND_MEMBER_TYPES = ("trend", "control")
TREND_MEMBER_DAYS = 14

RESET_SHARE = 0.6

WORTH_A_LOOK_TYPES = ("political_campaign", "unclear")

MODE_PRODUCTION = "production"
MODE_VALIDATION = "validation"


@dataclass
class Listing:
    run: str
    frame: pd.DataFrame


@dataclass
class Plan:
    run_id: str
    current: str | None
    previous: str | None
    baseline: bool
    reset: bool
    queued: pd.DataFrame
    skipped: pd.DataFrame
    communities: pd.DataFrame
    trends: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _files(con: duckdb.DuckDBPyConnection, prefix: str) -> list[str]:
    try:
        rows = con.sql(
            f"SELECT file FROM glob('r2://{BUCKET}/{prefix}/**/*.parquet') ORDER BY file"
        ).fetchall()
    except duckdb.Error:
        return []
    return [r[0] for r in rows]


def _run_of(path: str) -> str:
    return path.rsplit("run=", 1)[1].removesuffix(".parquet")


def _signature(frame: pd.DataFrame) -> tuple:
    return tuple(
        (c, None if pd.isna(frame[c].iloc[0]) else frame[c].iloc[0])
        for c in SIGNATURE if c in frame.columns and len(frame)
    )


def latest_listings(
    con: duckdb.DuckDBPyConnection, source: str = "v2"
) -> tuple[Listing | None, Listing | None]:
    """(previous, current) community listings, previous being the latest
    earlier run with the same detector parameters."""
    prefix = COMMUNITIES_PREFIX if source == "v2" else V1_PREFIX
    files = _files(con, prefix)
    if not files:
        return None, None

    def read(path: str) -> pd.DataFrame:
        frame = con.sql(f"SELECT * FROM read_parquet('{path}')").df()
        if source == "v1":
            frame = frame.rename(columns={"author_id": "user_id", "cluster_id": "community"})
        return frame.astype({"user_id": str})

    current = Listing(_run_of(files[-1]), read(files[-1]))
    want = _signature(current.frame)
    for path in reversed(files[:-1]):
        frame = read(path)
        if _signature(frame) == want:
            return Listing(_run_of(path), frame), current
    return None, current


def latest_trends(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    files = _files(con, TRENDS_PREFIX)
    if not files:
        return None, pd.DataFrame(columns=["tag", "authors", "emergence"])
    read = lambda p: con.sql(f"SELECT * FROM read_parquet('{p}')").df()  # noqa: E731
    return (read(files[-2]) if len(files) > 1 else None), read(files[-1])


def plan(
    previous: Listing | None,
    current: Listing | None,
    trends_previous: pd.DataFrame | None,
    trends_current: pd.DataFrame,
    *,
    limit: int = leads.MAX_CANDIDATES,
    min_kenya: float | None = 0.2,
    run_id: str | None = None,
) -> Plan:
    """Everything the pass will do, decided before anything is spent."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    notes: list[str] = []
    empty = pd.DataFrame(columns=["candidate_id", "status"])

    communities, baseline, reset = empty, False, False
    if current is None:
        notes.append("no community listing found; communities skipped")
    elif previous is None:
        baseline = True
        notes.append(f"no comparable listing before {current.run}; recorded as a baseline, nothing judged")
        communities = attach_canary_ids(leads.diff_communities(None, current.frame), current.frame)
    else:
        communities = attach_canary_ids(
            leads.diff_communities(previous.frame, current.frame), current.frame)
        share = float((communities["status"] == leads.STATUS_NEW).mean()) if len(communities) else 0.0
        # A partition that turns over wholesale is a detector or collection
        # change, not a wave of campaigns. Never reached in 29 measured v1 run
        # pairs (max 0.44); it is here for the day a parameter moves.
        if share > RESET_SHARE:
            reset = True
            notes.append(f"{share:.0%} of communities are new: treated as a partition reset, communities not judged")

    trends = leads.diff_trends(trends_previous, trends_current)

    # Canaries are judged even on a baseline or reset run: those days are not
    # canary failures, and skipping the canary would make them look like one.
    canaries = _canaries(communities)
    judged_communities = empty if (baseline or reset) else communities
    real = judged_communities[~judged_communities["candidate_id"].isin(canaries["candidate_id"])] \
        if len(canaries) else judged_communities
    if min_kenya is not None and "kenya_share" in real.columns:
        # A third of the A2 community listing was K-pop, anime and other
        # off-domain drives. Judging them spends the budget on groups that are
        # not this monitor's business.
        off = real["kenya_share"].fillna(0) < min_kenya
        if off.any():
            notes.append(f"{int(off.sum())} communit(ies) below Kenya share {min_kenya} not queued")
        real = real[~off]

    queued, skipped = leads.queue(real, trends, limit=limit)
    if len(canaries):
        # A canary rides outside the budget: it must never displace a real lead,
        # and a budget-full day must never make it look missing.
        canaries = canaries.assign(source=leads.SOURCE_COMMUNITY)
        queued = pd.concat([queued, canaries], ignore_index=True)
    return Plan(run_id, current.run if current else None, previous.run if previous else None,
                baseline, reset, queued, skipped, communities, trends, notes)


def _canaries(communities: pd.DataFrame) -> pd.DataFrame:
    if communities.empty or "canary_id" not in communities.columns:
        return communities.iloc[0:0]
    return communities[communities["canary_id"].notna()]


def attach_canary_ids(communities: pd.DataFrame, listing: pd.DataFrame) -> pd.DataFrame:
    """Mark communities containing canary members with their canary id."""
    if communities.empty:
        return communities.assign(canary_id=pd.Series(dtype=object))
    ids = listing["user_id"].astype(str)
    marked = listing[ids.str.startswith(CANARY_PREFIX)]
    by_group = marked.groupby("community")["user_id"].first().map(lambda u: u.split(":")[1])
    return communities.assign(canary_id=communities["group"].map(by_group))


def members(
    con: duckdb.DuckDBPyConnection,
    queued: pd.DataFrame,
    listing: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Dossier members (`cluster_id`, `author_id`) and the id -> candidate map."""
    frames, ids = [], {}
    comm = queued[queued["source"] == leads.SOURCE_COMMUNITY]
    if len(comm) and listing is not None:
        rows = listing[listing["community"].isin(comm["group"])]
        frames.append(rows.rename(columns={"community": "cluster_id", "user_id": "author_id"})
                      [["cluster_id", "author_id"]])
        ids.update({int(g): c for g, c in zip(comm["group"], comm["candidate_id"], strict=True)})

    trend = queued[queued["source"] == leads.SOURCE_TREND].reset_index(drop=True)
    for k, row in trend.iterrows():
        cid = TREND_ID_BASE + k
        authors = trend_authors(con, str(row["tag"]))
        frames.append(pd.DataFrame({"cluster_id": cid, "author_id": authors}))
        ids[cid] = row["candidate_id"]

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["cluster_id", "author_id"])
    return frame.astype({"author_id": str}), ids


def trend_authors(con: duckdb.DuckDBPyConnection, tag: str) -> list[str]:
    """Authors who used `tag` in the trend census or the control arm."""
    tag = tag.lower().lstrip("#").replace("'", "")
    found: set[str] = set()
    # One type at a time: DuckDB globs do not expand `{a,b}`, and a type with
    # no partitions yet must not cost the other its members.
    for kind in TREND_MEMBER_TYPES:
        glob = f"r2://{BUCKET}/posts/platform={PLATFORM}/type={kind}/dt=*/run=*.parquet"
        try:
            rows = con.sql(
                f"""
                SELECT DISTINCT author_id
                FROM read_parquet('{glob}', union_by_name=true, hive_partitioning=true)
                WHERE dt >= current_date - INTERVAL {TREND_MEMBER_DAYS} DAY
                  AND author_id IS NOT NULL
                  AND (list_contains(list_transform(hashtags, h -> lower(h)), '{tag}')
                       OR lower(text) LIKE '%#{tag}%')
                """
            ).fetchall()
        except duckdb.Error:
            log.exception("could not read %s posts for #%s", kind, tag)
            continue
        found.update(str(r[0]) for r in rows)
    return sorted(found)


def worth_a_look(verdict: dict, status: str) -> bool:
    """Whether a verdict should reach a human's phone.

    An influence-operation call always does, whatever its confidence. A new
    Kenya-relevant political campaign or an unclear call on a new
    Kenya-relevant group does too - a campaign starting is the event this
    monitor exists to catch, and "unclear" means the reader could not rule
    it out. A changed group re-alerts only if it is now called an operation,
    so an open campaign growing day by day does not ring every morning.

    Against the three A2 reads this flags 6 of 21, 7 of 17 and 6 of 11 v2
    communities, and 2, 1 and 0 of v1's."""
    kind = verdict.get("cluster_type")
    if kind == "influence_operation":
        return True
    return (
        status == leads.STATUS_NEW
        and kind in WORTH_A_LOOK_TYPES
        and verdict.get("kenya_relevant") is True
    )


def verdict_rows(
    p: Plan,
    verdicts: list[dict],
    ids: dict[int, str],
    dossier_keys: dict[str, str],
    *,
    adjudicator: str,
    rubric: str,
    mode: str,
) -> pd.DataFrame:
    queued = p.queued.set_index("candidate_id")
    rows = []
    for v in verdicts:
        cand = ids.get(int(v["cluster_id"]))
        if cand is None:
            continue
        q = queued.loc[cand]
        status = str(q["status"])
        canary = q.get("canary_id") if "canary_id" in queued.columns else None
        canary = None if canary is None or pd.isna(canary) else str(canary)
        rows.append({
            "run_id": p.run_id,
            "candidate_id": cand,
            "source": q["source"],
            "status": status,
            "label": str(q.get("tag")) if q["source"] == leads.SOURCE_TREND else None,
            "size": _num(q.get("size")),
            "previous_size": _num(q.get("previous_size")),
            "jaccard": _num(q.get("jaccard")),
            "joined": _num(q.get("joined")),
            "cluster_type": v.get("cluster_type"),
            "kenya_relevant": v.get("kenya_relevant"),
            "confidence": v.get("confidence"),
            "rationale": v.get("rationale"),
            "what_would_change_this": v.get("what_would_change_this"),
            "worth_a_look": worth_a_look(v, status),
            "canary_id": canary,
            "dossier_key": dossier_keys.get(cand),
            "listing_run": p.current,
            "previous_run": p.previous,
            "adjudicator": adjudicator,
            "rubric": rubric,
            "mode": mode,
            "adjudicated_at": datetime.now(timezone.utc),
        })
    return pd.DataFrame(rows)


def _num(value):
    return None if value is None or pd.isna(value) else float(value)


def _key(kind: str, run_id: str, suffix: str = ".parquet") -> str:
    day = f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
    return f"{LEADS_PREFIX}/kind={kind}/dt={day}/run={run_id}{suffix}"


def _copy(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, key: str) -> str:
    con.register("_leads_buf", frame)
    try:
        con.execute(f"COPY _leads_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)")
    finally:
        con.unregister("_leads_buf")
    return key


def persist_queue(con: duckdb.DuckDBPyConnection, p: Plan, mode: str) -> str:
    frame = pd.concat([p.queued.assign(queued=True), p.skipped.assign(queued=False)],
                      ignore_index=True)
    keep = [c for c in ("candidate_id", "source", "status", "group", "tag", "size",
                        "previous", "previous_size", "shared", "jaccard", "containment",
                        "joined", "eigenvalue", "kenya_share", "emergence", "authors",
                        "canary_id", "queued") if c in frame.columns]
    frame = frame[keep].copy()
    for c in ("group", "previous", "tag", "canary_id"):
        if c in frame.columns:
            frame[c] = frame[c].map(lambda x: None if x is None or pd.isna(x) else str(x))
    frame["run_id"] = p.run_id
    frame["listing_run"] = p.current
    frame["previous_run"] = p.previous
    frame["baseline"] = p.baseline
    frame["reset"] = p.reset
    frame["mode"] = mode
    frame["computed_at"] = datetime.now(timezone.utc)
    return _copy(con, frame, _key("queue", p.run_id))


def dossier_key(run_id: str, candidate: str) -> str:
    return _key("dossiers", run_id, "") + f"/{candidate.replace(':', '_')}.parquet"


def persist_dossier(con: duckdb.DuckDBPyConnection, run_id: str, candidate: str,
                    packet: dict, prompt: str) -> str:
    frame = pd.DataFrame([{"candidate_id": candidate, "run_id": run_id,
                           "packet": json.dumps(packet, default=str), "prompt": prompt}])
    return _copy(con, frame, dossier_key(run_id, candidate))


def persist_verdicts(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, run_id: str) -> str:
    return _copy(con, frame, _key("verdicts", run_id))


def execute(
    con: duckdb.DuckDBPyConnection,
    *,
    source: str = "v2",
    limit: int = leads.MAX_CANDIDATES,
    judge=None,
    model: str | None = None,
    persist: bool = False,
    mode: str = MODE_PRODUCTION,
    max_judged: int | None = None,
) -> dict:
    """The whole pass. `judge` is `adjudicate.judge_all`, or None for a dry
    run that builds nothing and spends nothing."""
    from kma import adjudicate, dossier

    previous, current = latest_listings(con, source)
    trends_prev, trends_cur = latest_trends(con)
    p = plan(previous, current, trends_prev, trends_cur, limit=limit)
    if max_judged is not None:
        extra = p.queued.iloc[max_judged:]
        p.queued = p.queued.iloc[:max_judged]
        p.skipped = pd.concat([extra, p.skipped], ignore_index=True)

    summary = {
        "run_id": p.run_id, "listing": p.current, "previous": p.previous,
        "baseline": p.baseline, "reset": p.reset, "notes": p.notes,
        "communities": _counts(p.communities), "trends": _counts(p.trends),
        "queued": p.queued[["candidate_id", "source", "status"]].to_dict("records"),
        "skipped": len(p.skipped), "judged": 0,
    }
    if persist:
        summary["queue_key"] = persist_queue(con, p, mode)
    if judge is None or p.queued.empty:
        return summary

    frame, ids = members(con, p.queued, current.frame if current else None)
    packets = dossier.build(con, frame, cluster_ids=list(ids))
    keys: dict[str, str] = {}
    if persist:
        for packet in packets:
            cand = ids[int(packet["cluster_id"])]
            keys[cand] = persist_dossier(con, p.run_id, cand, packet, adjudicate.prompt(packet))
    chosen = model or adjudicate.MODEL
    verdicts = judge(packets, model=chosen)
    rows = verdict_rows(p, verdicts, ids, keys, adjudicator=chosen,
                        rubric=adjudicate.RUBRIC_VERSION, mode=mode)
    summary["judged"] = len(rows)
    summary["verdicts"] = rows.drop(columns=["adjudicated_at"]).to_dict("records")
    if persist and len(rows):
        summary["verdicts_key"] = persist_verdicts(con, rows, p.run_id)
    return summary


def _counts(frame: pd.DataFrame) -> dict:
    if frame.empty or "status" not in frame.columns:
        return {}
    return frame["status"].value_counts().to_dict()


def show(con: duckdb.DuckDBPyConnection, run_id: str, candidate: str) -> str:
    key = dossier_key(run_id, candidate)
    return con.sql(f"SELECT prompt FROM read_parquet('r2://{BUCKET}/{key}')").fetchone()[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Diff the latest runs into a bounded adjudication queue.")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("show", help="print the prompt a reader saw for one candidate")
    s.add_argument("run_id")
    s.add_argument("candidate_id")
    ap.add_argument("--source", choices=("v2", "v1"), default="v2",
                    help="v1 reads coordination clusters as a stand-in listing")
    ap.add_argument("--limit", type=int, default=leads.MAX_CANDIDATES)
    ap.add_argument("--judge", action="store_true",
                    help="call the reader locally; needs ANTHROPIC_API_KEY in the environment")
    ap.add_argument("--max-judged", type=int, default=0, help="hard cap on reader calls, 0 for none")
    ap.add_argument("--persist", action="store_true", help="write queue, dossiers and verdicts to R2")
    ap.add_argument("--mode", choices=(MODE_PRODUCTION, MODE_VALIDATION), default=MODE_VALIDATION,
                    help="validation verdicts are never sent by the poller")
    args = ap.parse_args()

    con = connect()
    if args.cmd == "show":
        print(show(con, args.run_id, args.candidate_id))
        return
    judge = None
    if args.judge:
        from kma import adjudicate

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("--judge needs ANTHROPIC_API_KEY in the environment")
        judge = adjudicate.judge_all
    print(json.dumps(execute(con, source=args.source, limit=args.limit, judge=judge,
                             persist=args.persist, mode=args.mode,
                             max_judged=args.max_judged or None),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
