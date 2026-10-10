"""Public dashboard producer: one `summary.json`, public by construction.

    uv run kma-dashboard --out-dir /tmp/dash      # build, write /tmp/dash/summary.json only
    uv run kma-dashboard                          # build, publish to R2_DASHBOARD_BUCKET

Feeds kenya.tomfr.net. What may appear on that page is fixed by
`docs/analysis/2026-10-10-publishing-policy.md` (which supersedes
`2026-09-13-publishable-statistics.md`); this module is that
document made executable, and it can only emit what it names:

- the benchmark reproduction, the relevance gate's human-measured error, what
  the detector surfaced on a frozen snapshot, and the negative results. These
  are RECORDED figures (`analysis/figures/recorded.yaml`), transcribed once
  with their source and date, because none is derivable from a persisted
  artifact.
- pipeline and collection health in aggregate: rows per collection day, the
  latest daily run's freshness and stage timings, enrichment lag. These are
  LIVE figures, recomputed from persisted R2 output on every run.

Never emitted: any prevalence figure, any per-account value or handle (the
explorer carries account-level views; this summary stays aggregate), any
count of surfaced accounts or communities from the live run as a finding, the
raw toxicity series, corroboration claims. `withheld` on the payload lists the
policy in `docs/analysis/2026-10-10-publishing-policy.md`, so the page states
the gap instead of leaving it for a reader to fill.

## Persisted output only

The producer reads what the daily run and the collector already wrote:
`coord2/kind=daily_runs`, and object listings and parquet footers for `posts/`,
`embeddings/`, `relevance/`, `hatespeech/`, `incitement/`. It never rebuilds the
pipeline, loads a model or touches X. Footers, not row scans: row counts come
from parquet metadata.

## A missing source is "not available"

A card whose source is absent or unreadable carries `status: not_available`
and a fixed `reason`. It never carries a placeholder number. Exception text is
logged and never published: it can contain object keys. If every live card is
unavailable the run refuses to publish, so a broken credential cannot replace a
good page with an empty one.

## Public-safety guard

`assert_public` walks the finished payload and refuses a forbidden key, an
`@handle`, a profile URL, or a run of 12+ digits (an X id) before anything is
written. `tests/test_dashboard.py` asserts it against a built payload.

## Where it writes

`R2_DASHBOARD_BUCKET` (default `kenya-monitor-dashboard`) with the dedicated
`R2_DASHBOARD_ACCESS_KEY_ID` / `R2_DASHBOARD_SECRET_ACCESS_KEY` token. The
Worker binds that bucket only, which holds nothing but these blobs: an R2
binding cannot be scoped to a prefix, so the data bucket is never bound. The
producer refuses to publish into the data bucket.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from kma import db, figures, relevance

log = logging.getLogger("kma.dashboard")

SCHEMA_VERSION = 3
PLATFORM = "x"
PUBLIC_KEY = "public/summary.json"
DASHBOARD_BUCKET = os.getenv("R2_DASHBOARD_BUCKET", "kenya-monitor-dashboard")

# The daily is expected every 24 h. 36 h tolerates one slow or retried run and
# still flags a missed day.
STALE_AFTER_HOURS = 36
COLLECTION_DAYS = 30
ENRICHMENT_DAYS = 7
RUN_HISTORY_DAYS = 14

# Order is the order a reader meets them on the page.
LIVE_CARDS = ("pipeline", "collection", "enrichment")
# Derived from the collection and enrichment cards: no extra read.
DERIVED_CARDS = ("composition", "control_arm", "data_quality", "stage_coverage")
LOW_VOLUME_FRACTION = 0.5

FORBIDDEN_KEYS = frozenset({
    "user_id", "author_id", "account_id", "platform_user_id", "handle", "username",
    "screen_name", "display_name", "name", "bio", "text", "post_text", "platform_post_id",
    "post_id", "tweet_id", "url", "members", "accounts", "listed", "community",
    "shared_objects", "representative_posts", "amplification_targets", "object_text",
    "object_author", "target_handle", "author_handle",
})

_HANDLE = re.compile(r"(?<![\w.])@\w{2,}")
_LONG_DIGITS = re.compile(r"\d{12,}")
_PROFILE_URL = re.compile(r"(?:x|twitter)\.com/\w+", re.I)
_PARTITION = re.compile(r"(?:^|/)(\w+)=([^/]+)")

# Reasons are fixed strings. A reason is published; an exception message is not.
REASON_UNREADABLE = "The source could not be read in this run."
REASON_NO_DAILY = "No daily detector run has been persisted yet."

WITHHELD = (
    {
        "item": "Any prevalence rate",
        "why": (
            "Nothing in the collector samples the discourse at random, so no rate "
            "over it has a denominator. The control arm is the only denominator and "
            "it is small."
        ),
    },
    {
        "item": "A raw toxicity or hate series over time",
        "why": (
            "The raw series moved with the collector's own target list: replies "
            "carry 3.2x the hate rate and the mix shifted from search to replies. "
            "Only a composition-standardised or within-partition rate may be shown, "
            "and none is shown yet."
        ),
    },
    {
        "item": "Counts of surfaced accounts or communities as a finding",
        "why": (
            "The listing length is a triage budget and the community count depends "
            "on a clustering resolution and seed. Neither is a finding."
        ),
    },
    {
        "item": "Cross-channel corroboration of clusters",
        "why": "The earlier statement about it is false and was retired.",
    },
    {
        "item": "Any statement about what a named account is, or intends",
        "why": (
            "Co-activity is evidence that accounts act together. It does not show "
            "that an account is inauthentic, paid or acting with intent, and it says "
            "nothing about who runs it."
        ),
    },
    {
        "item": "Per-account hate, incitement, toxicity, deletion or suspension flags",
        "why": (
            "A strong accusation against a named person from a classifier with a "
            "documented weakness on coded speech. Only aggregates by topic, with "
            "denominators, may be shown."
        ),
    },
    {
        "item": "Location text, bios used to characterise a group, cross-platform identity links",
        "why": "Harm without a measurement that needs it.",
    },
)


def available(card_id: str, title: str, *, basis: str, source: str, **fields: Any) -> dict:
    return {"id": card_id, "title": title, "status": "ok", "basis": basis,
            "source": source, **fields}


def unavailable(card_id: str, title: str, *, reason: str, source: str) -> dict:
    return {"id": card_id, "title": title, "status": "not_available",
            "reason": reason, "source": source}


def public_violations(value: Any, path: str = "$") -> list[str]:
    """Every way a payload could carry something that identifies an account."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for k, v in value.items():
            here = f"{path}.{k}"
            if str(k).lower() in FORBIDDEN_KEYS:
                found.append(f"{here}: forbidden key")
            found += public_violations(v, here)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            found += public_violations(v, f"{path}[{i}]")
    elif isinstance(value, str):
        for label, pattern in (("handle", _HANDLE), ("snowflake-length digits", _LONG_DIGITS),
                               ("profile url", _PROFILE_URL)):
            if pattern.search(value):
                found.append(f"{path}: {label}")
    elif isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 10**10:
        found.append(f"{path}: integer the size of an account id")
    return found


def assert_public(payload: dict) -> None:
    problems = public_violations(payload)
    if problems:
        raise ValueError("refusing to publish, payload is not public-safe: " + "; ".join(problems))


@dataclass
class Sources:
    """The three reads the live cards need, as callables so tests can fake R2."""

    list_prefix: Callable[[str], pd.DataFrame]
    """Objects under a prefix: columns key, size, dt, type, model."""
    object_rows: Callable[[list[str]], dict[str, int]]
    """Row count per key, from parquet footers. Must answer every key."""
    daily_runs: Callable[[], pd.DataFrame | None]
    """`coord2/kind=daily_runs`, or None when no run was ever persisted."""


def parse_listing(objects: Iterable[Mapping]) -> pd.DataFrame:
    rows = []
    for obj in objects:
        parts = dict(_PARTITION.findall(obj["Key"]))
        rows.append({
            "key": obj["Key"], "size": int(obj["Size"]),
            "dt": parts.get("dt"), "type": parts.get("type"), "model": parts.get("model"),
        })
    return pd.DataFrame(rows, columns=["key", "size", "dt", "type", "model"])


def _scope_of(post_type: str | None) -> str:
    if post_type in db.BASELINE_TYPES:
        return "baseline"
    if post_type in db.TARGETED_TYPES:
        return "targeted"
    if post_type in db.CONTROL_TYPES:
        return "control"
    return "other"


def _closed_days(today: date, days: int) -> list[date]:
    """The `days` UTC days ending yesterday. Today's partition is still open."""
    yesterday = today - timedelta(days=1)
    return [yesterday - timedelta(days=i) for i in range(days - 1, -1, -1)]


def collection_card(
    posts: pd.DataFrame, object_rows: Callable[[list[str]], dict[str, int]], today: date,
    days: int = COLLECTION_DAYS,
) -> dict:
    title = "Collection volume"
    source = "posts/ partitions: object listing and parquet footers"
    if posts.empty or posts["dt"].dropna().empty:
        return unavailable("collection", title, reason=REASON_UNREADABLE, source=source)

    observed = sorted(set(posts["dt"].dropna()))
    first = date.fromisoformat(observed[0])
    span = [first + timedelta(days=i) for i in range((today - timedelta(days=1) - first).days + 1)]
    seen = set(observed)
    gaps = [d.isoformat() for d in span if d.isoformat() not in seen]

    window = _closed_days(today, days)
    wanted = {d.isoformat() for d in window}
    inside = posts[posts["dt"].isin(wanted)]
    counts = object_rows(list(inside["key"]))
    per_day = inside.assign(rows=inside["key"].map(counts), scope=inside["type"].map(_scope_of))
    if per_day["rows"].isna().any():
        raise ValueError("object_rows did not answer every posts key")

    series = []
    for d in window:
        day = per_day[per_day["dt"] == d.isoformat()]
        by_scope = day.groupby("scope")["rows"].sum()
        by_type = day.groupby("type")["rows"].sum()
        series.append({
            "dt": d.isoformat(),
            "rows": int(day["rows"].sum()),
            "objects": int(len(day)),
            "gap": day.empty,
            "by_scope": {s: int(by_scope.get(s, 0))
                         for s in ("baseline", "targeted", "control", "other")},
            "by_type": {str(t): int(n) for t, n in sorted(by_type.items())},
        })

    return available(
        "collection", title, source=source,
        basis=(
            "Rows the collector wrote to posts/ per UTC collection day (the dt "
            "partition), all post types. Includes re-collections of the same "
            "post, so it is not distinct posts and not Kenyan posts. Scope is "
            "the partition's own type, not first-seen type. A day with no "
            "partition is a gap day: nothing was collected and it cannot be "
            "backfilled. Today's open partition is excluded."
        ),
        first_collection_day=observed[0],
        last_collection_day=observed[-1],
        gap_days_total=len(gaps),
        gap_days=gaps,
        series_days=days,
        series=series,
    )


def enrichment_card(
    listings: Mapping[str, pd.DataFrame],
    object_rows: Callable[[list[str]], dict[str, int]], today: date,
    days: int = ENRICHMENT_DAYS,
) -> dict:
    title = "Enrichment lag"
    source = "posts/, embeddings/, relevance/, hatespeech/, incitement/: listings and footers"
    posts = listings.get("posts")
    if posts is None or posts["dt"].dropna().empty:
        return unavailable("enrichment", title, reason=REASON_UNREADABLE, source=source)
    posts_latest = date.fromisoformat(max(posts["dt"].dropna()))

    wanted = {d.isoformat() for d in _closed_days(today, days)}
    recent_keys = [k for frame in listings.values()
                   for k in frame.loc[frame["dt"].isin(wanted), "key"]]
    counts = object_rows(recent_keys) if recent_keys else {}

    stages = []
    for name, frame in listings.items():
        dts = frame["dt"].dropna()
        if dts.empty:
            stages.append({"stage": name, "latest_dt": None, "lag_days": None,
                           "rows_last_7d": None, "objects_last_7d": None})
            continue
        latest = max(dts)
        recent = frame[frame["dt"].isin(wanted)]
        stages.append({
            "stage": name,
            "latest_dt": latest,
            "lag_days": (posts_latest - date.fromisoformat(latest)).days,
            "rows_last_7d": int(sum(counts[k] for k in recent["key"])),
            "objects_last_7d": int(len(recent)),
        })
    return available(
        "enrichment", title, source=source,
        basis=(
            "Latest dt partition per stage, and rows written to it over the "
            f"last {days} closed UTC days. A stage's dt is when it ran, not the "
            "date of the posts it scored, so rows are throughput and not "
            "coverage. lag_days is the newest posts dt minus the stage's newest "
            "dt. relevance is the promoted model only."
        ),
        relevance_model=relevance.MODEL,
        window_days=days,
        stages=stages,
    )


_TIMING_COLUMNS = ("networks_s", "fuse_centrality_s", "relevance_s", "communities_s")


def pipeline_card(runs: pd.DataFrame | None, now: datetime) -> dict:
    title = "Daily detector run"
    source = "coord2/kind=daily_runs"
    if runs is None or runs.empty:
        return unavailable("pipeline", title, reason=REASON_NO_DAILY, source=source)

    runs = runs.assign(computed_at=pd.to_datetime(runs["computed_at"], utc=True))
    runs = runs.sort_values("computed_at")
    last = runs.iloc[-1]
    computed = last["computed_at"].to_pydatetime()
    age_h = max(0.0, (now - computed).total_seconds() / 3600)

    cutoff = now - timedelta(days=RUN_HISTORY_DAYS)
    ran_on = sorted({r.date().isoformat() for r in runs["computed_at"] if r >= cutoff})
    first_run_day = runs["computed_at"].iloc[0].date()
    # Only closed days after the very first persisted run: before it nothing was
    # expected, and today's run may simply not have fired yet.
    since = max(first_run_day, now.date() - timedelta(days=RUN_HISTORY_DAYS))
    missing = [d.isoformat() for d in _closed_days(now.date(), (now.date() - since).days)
               if d.isoformat() not in ran_on]

    def num(col: str):
        v = last.get(col)
        return None if v is None or pd.isna(v) else float(v)

    listed = num("listed_accounts")
    return available(
        "pipeline", title, source=source,
        basis=(
            "The newest persisted run of the daily detector, from its own run "
            f"record. Stale means older than {STALE_AFTER_HOURS} h when this "
            "page was built. The listing's length and contents are not published."
        ),
        run_id=str(last["run_id"]),
        window_id=str(last["window_id"]),
        window_end=str(last["window_end"]),
        window_days=int(last["window_days"]),
        computed_at=computed.isoformat(),
        age_hours=round(age_h, 1),
        stale_after_hours=STALE_AFTER_HOURS,
        stale=age_h > STALE_AFTER_HOURS,
        listing_nonempty=None if listed is None else listed > 0,
        parameters={
            "text_threshold": num("text_threshold"),
            "text_overlap": num("text_overlap"),
            "min_entities": None if num("min_entities") is None else int(num("min_entities")),
            "min_kenya": num("min_kenya"),
        },
        stage_seconds={c: num(c) for c in _TIMING_COLUMNS},
        history_days=RUN_HISTORY_DAYS,
        run_days=ran_on,
        missing_run_days=missing,
        runs_persisted=int(len(runs)),
    )


def composition_card(collection: dict) -> dict:
    title = "Collector composition"
    source = "posts/ partitions: object listing and parquet footers"
    series = collection["series"]
    total = sum(r["rows"] for r in series)
    if total == 0:
        return unavailable("composition", title, reason=REASON_UNREADABLE, source=source)

    by_type: dict[str, int] = {}
    for r in series:
        for t, n in r["by_type"].items():
            by_type[t] = by_type.get(t, 0) + n
    scopes = ("baseline", "targeted", "control", "other")

    def share(n: int, of: int) -> float | None:
        return None if of == 0 else round(n / of, 4)

    def scope_rows(rows: list[dict]) -> dict[str, int]:
        return {s: sum(r["by_scope"][s] for r in rows) for s in scopes}

    half = len(series) // 2
    first, second = series[:half], series[half:]
    first_total, second_total = sum(r["rows"] for r in first), sum(r["rows"] for r in second)
    first_rows, second_rows = scope_rows(first), scope_rows(second)
    return available(
        "composition", title, source=source,
        basis=(
            f"Rows written to posts/ over the last {collection['series_days']} closed "
            "UTC days, by the partition's own post type and by scope. Shares are of "
            "those rows: not distinct posts, not Kenyan posts, not a sample of the "
            "discourse. The mix is set by the collector's own target list, so a "
            "change between the two halves is the collector changing, not Kenya "
            "changing."
        ),
        window_days=collection["series_days"],
        total_rows=total,
        by_type=[
            {"type": t, "scope": _scope_of(t), "rows": n, "share": share(n, total)}
            for t, n in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        by_scope=[
            {"scope": s, "rows": n, "share": share(n, total)}
            for s, n in scope_rows(series).items()
        ],
        halves={
            "first_days": len(first),
            "second_days": len(second),
            "first_share": {s: share(n, first_total) for s, n in first_rows.items()},
            "second_share": {s: share(n, second_total) for s, n in second_rows.items()},
        },
    )


def control_arm_card(collection: dict) -> dict:
    title = "Control arm"
    source = "posts/ partitions, type=control: object listing and parquet footers"
    series = collection["series"]
    total = sum(r["rows"] for r in series)
    if total == 0:
        return unavailable("control_arm", title, reason=REASON_UNREADABLE, source=source)
    control = [r["by_scope"]["control"] for r in series]
    return available(
        "control_arm", title, source=source,
        basis=(
            "Rows written to the control partition, the only arm that is not drawn "
            "from the collector's target list, over the last "
            f"{collection['series_days']} closed UTC days. Its share of rows is how "
            "much of the corpus is control, not a rate of anything. A day with no "
            "control rows is counted, not filled."
        ),
        window_days=collection["series_days"],
        control_rows=sum(control),
        window_rows=total,
        control_share=round(sum(control) / total, 4),
        days_with_control=sum(1 for n in control if n > 0),
        days_without_control=sum(1 for n in control if n == 0),
    )


def data_quality_card(collection: dict) -> dict:
    title = "Data quality"
    source = "posts/ partitions: object listing and parquet footers"
    series = collection["series"]
    present = [r for r in series if not r["gap"]]
    if not present:
        return unavailable("data_quality", title, reason=REASON_UNREADABLE, source=source)
    rows = sorted(r["rows"] for r in present)
    mid = len(rows) // 2
    median = rows[mid] if len(rows) % 2 else (rows[mid - 1] + rows[mid]) / 2
    low = [r["dt"] for r in present if r["rows"] < LOW_VOLUME_FRACTION * median]
    return available(
        "data_quality", title, source=source,
        basis=(
            f"Over the last {collection['series_days']} closed UTC days: days with "
            "no partition at all (gap days, never recoverable), and days whose row "
            f"count is under {int(LOW_VOLUME_FRACTION * 100)}% of the median "
            "day's (low-volume days, which usually mean a partial collection). "
            "Counted from partition listings and parquet footers, not from post "
            "content."
        ),
        window_days=collection["series_days"],
        days_with_data=len(present),
        gap_days_in_window=len(series) - len(present),
        gap_days_total=collection["gap_days_total"],
        median_rows_per_day=median,
        low_volume_fraction=LOW_VOLUME_FRACTION,
        low_volume_days=low,
        min_objects_per_day=min(r["objects"] for r in present),
        max_objects_per_day=max(r["objects"] for r in present),
    )


def stage_coverage_card(enrichment: dict) -> dict:
    title = "Rows per stage against posts"
    source = "posts/, embeddings/, relevance/, hatespeech/, incitement/: listings and footers"
    stages = {s["stage"]: s for s in enrichment["stages"]}
    base = stages.get("posts", {}).get("rows_last_7d")
    if not base:
        return unavailable("stage_coverage", title, reason=REASON_UNREADABLE, source=source)
    out = []
    for name, st in stages.items():
        if name == "posts":
            continue
        n = st["rows_last_7d"]
        out.append({
            "stage": name,
            "rows": n,
            "rows_per_posts_row": None if n is None else round(n / base, 3),
        })
    return available(
        "stage_coverage", title, source=source,
        basis=(
            f"Rows each stage wrote over the last {enrichment['window_days']} closed "
            "UTC days, divided by rows posts/ wrote over the same days. A stage's dt "
            "is when it ran, not the date of the posts it scored, and posts/ counts "
            "re-collections, so this is a throughput ratio and not the share of "
            "posts a stage has covered. Coverage by distinct post needs a join "
            "that has not been measured."
        ),
        window_days=enrichment["window_days"],
        posts_rows=base,
        stages=out,
    )


def benchmark_card(block: dict) -> dict:
    rows = []
    for r in block["rows"]:
        paper = float(r.get("paper", r["published"]))
        ours = float(r["ours"])
        entry = {"dataset": r["country"], "paper": paper, "ours": ours,
                 "delta": round(ours - paper, 2)}
        if "paper" in r:
            entry["reference_code"] = float(r["published"])
            entry["delta_vs_reference_code"] = round(ours - float(r["published"]), 2)
        rows.append(entry)
    close = sum(1 for r in rows if abs(r["delta"]) <= 0.2)
    return available(
        "benchmark", block["title"], source=block["source"], measured=str(block["measured"]),
        basis=(
            "Macro-F1 of the unsupervised IOHunter-style detector re-implemented "
            f"here, on the {len(rows)} published IO benchmark datasets, against "
            "the number the paper reports. A claim about this copy of the method, "
            "never a Kenya result."
        ),
        metric="Macro-F1",
        datasets=len(rows),
        within_0_2=close,
        tolerance=0.2,
        rows=rows,
        caveat=block["note"],
    )


def relevance_gate_card(block: dict) -> dict:
    return available(
        "relevance_gate", block["title"], source=block["source"],
        measured=str(block["measured"]),
        basis=(
            f"{block['labels']} posts drawn fresh, labelled by a human blind to "
            "both methods, and corpus-weighted. Precision: of the posts the "
            "filter calls Kenyan, the share that are. Recall: of the Kenyan "
            "posts, the share the filter finds."
        ),
        labelled_posts=int(block["labels"]),
        rows=[{"method": r["method"], "precision": float(r["precision"]),
               "recall": float(r["recall"])} for r in block["rows"]],
        caveat=block["note"],
    )


def detector_card(block: dict, runs: pd.DataFrame | None, now: datetime) -> dict:
    """What the detector surfaced on a frozen snapshot, as its own output.

    The live run contributes only whether it listed anything: its counts are
    withheld, so no number on this card describes how much coordination exists.
    """
    rows = [{"method": r["method"], "cases": int(r["cases"]), "political": int(r["political"]),
             "kenya_relevant": int(r["kenya_relevant"])} for r in block["rows"]]
    live = pipeline_card(runs, now)
    return available(
        "detector", block["title"], source=block["source"], measured=str(block["measured"]),
        basis=(
            "One frozen snapshot, not the live run. Each method's communities "
            f"were read blind by a model reader at a relevance floor of "
            f"{block['relevance_floor']}; cases are communities read, and the "
            "counts are how many that reader called political or Kenya-relevant. "
            "This describes what the method surfaced. It is not a rate: there "
            "is no denominator."
        ),
        snapshot=block["snapshot"],
        relevance_floor=float(block["relevance_floor"]),
        rows=rows,
        influence_operations_found=int(block["influence_operations_found"]),
        caveat=block["note"],
        live_listing={
            "status": live["status"],
            "computed_at": live.get("computed_at"),
            "listing_nonempty": live.get("listing_nonempty"),
            "published": False,
        },
    )


def negative_results_card(block: dict) -> dict:
    items = []
    for it in block["items"]:
        entry = {"id": it["id"], "claim": " ".join(it["claim"].split()),
                 "figure": it["figure"], "basis": " ".join(it["basis"].split()),
                 "source": it["source"]}
        if it.get("checked"):
            entry["checked"] = str(it["checked"])
        items.append(entry)
    return available(
        "negative_results", block["title"], source=block["source"],
        measured=str(block["measured"]),
        basis=(
            "Each is a measurement of this project's own method failing, so no "
            "collection artifact can flatter it."
        ),
        items=items,
    )


def build_summary(
    src: Sources, recorded: dict, *, now: datetime | None = None, code_version: str | None = None,
) -> dict:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    today = now.date()

    def guarded(card_id: str, title: str, source: str, build: Callable[[], dict]) -> dict:
        try:
            return build()
        except Exception:
            log.exception("card %s could not be built", card_id)
            return unavailable(card_id, title, reason=REASON_UNREADABLE, source=source)

    runs_cache: dict[str, pd.DataFrame | None] = {}

    def runs() -> pd.DataFrame | None:
        if "v" not in runs_cache:
            runs_cache["v"] = src.daily_runs()
        return runs_cache["v"]

    listings: dict[str, pd.DataFrame] = {}

    def listing(name: str, prefix: str) -> pd.DataFrame:
        if name not in listings:
            listings[name] = src.list_prefix(prefix)
        return listings[name]

    def enrichment() -> dict:
        stages = {
            "posts": f"posts/platform={PLATFORM}/",
            "embeddings": f"embeddings/platform={PLATFORM}/",
            "relevance": f"relevance/platform={PLATFORM}/model={relevance.MODEL}/",
            "hatespeech": f"hatespeech/platform={PLATFORM}/",
            "incitement": f"incitement/platform={PLATFORM}/",
        }
        return enrichment_card({n: listing(n, p) for n, p in stages.items()},
                               src.object_rows, today)

    live = {
        "pipeline": guarded("pipeline", "Daily detector run", "coord2/kind=daily_runs",
                            lambda: pipeline_card(runs(), now)),
        "collection": guarded("collection", "Collection volume", "posts/",
                              lambda: collection_card(
                                  listing("posts", f"posts/platform={PLATFORM}/"),
                                  src.object_rows, today)),
        "enrichment": guarded("enrichment", "Enrichment lag", "enrichment prefixes", enrichment),
    }
    if all(c["status"] != "ok" for c in live.values()):
        raise RuntimeError("no live source could be read; not publishing an empty page")

    def derived(card_id: str, title: str, parent: dict, build: Callable[[dict], dict]) -> dict:
        if parent["status"] != "ok":
            return unavailable(card_id, title, reason=REASON_UNREADABLE, source=parent["source"])
        return guarded(card_id, title, parent["source"], lambda: build(parent))

    derived_cards = {
        "composition": derived("composition", "Collector composition", live["collection"],
                               composition_card),
        "control_arm": derived("control_arm", "Control arm", live["collection"],
                               control_arm_card),
        "data_quality": derived("data_quality", "Data quality", live["collection"],
                                data_quality_card),
        "stage_coverage": derived("stage_coverage", "Rows per stage against posts",
                                  live["enrichment"], stage_coverage_card),
    }
    cards = {
        "benchmark": benchmark_card(recorded["benchmark_reproduction"]),
        "relevance_gate": relevance_gate_card(recorded["relevance_error"]),
        "detector": guarded("detector", "What the detector surfaced", "recorded",
                            lambda: detector_card(recorded["adjudication"], runs(), now)),
        "negative_results": negative_results_card(recorded["negative_results"]),
        **live,
        **derived_cards,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(),
        "code_version": code_version,
        "platform": PLATFORM,
        "scope": (
            "Aggregate descriptions of this project's own method and pipeline. "
            "No prevalence figure and nothing per account: account-level views "
            "live in the explorer, not in this summary."
        ),
        "cards": cards,
        "withheld": [dict(w) for w in WITHHELD],
    }
    assert_public(summary)
    return summary


def r2_sources(
    con: duckdb.DuckDBPyConnection, daily_root: str | None = None,
) -> Sources:
    """The real reads. `daily_root` points the daily-run read at a local
    directory laid out like a `v2_daily_run --out-dir`, for a trial before the
    cloud host has written anything; everything else always comes from R2."""
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )

    def list_prefix(prefix: str) -> pd.DataFrame:
        objects = []
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=db.BUCKET, Prefix=prefix
        ):
            objects += page.get("Contents", [])
        return parse_listing(objects)

    def object_rows(keys: list[str]) -> dict[str, int]:
        if not keys:
            return {}
        uris = [f"r2://{db.BUCKET}/{k}" for k in keys]
        got = con.execute(
            "SELECT file_name, sum(num_rows)::BIGINT FROM parquet_file_metadata(?) GROUP BY 1",
            [uris],
        ).fetchall()
        by_uri = dict(got)
        missing = [u for u in uris if u not in by_uri]
        if missing:
            raise ValueError(f"{len(missing)} object(s) returned no footer")
        return {k: int(by_uri[u]) for k, u in zip(keys, uris)}

    root = daily_root.rstrip("/") if daily_root else f"r2://{db.BUCKET}"

    def daily_runs() -> pd.DataFrame | None:
        glob = f"{root}/coord2/platform={PLATFORM}/kind=daily_runs/dt=*/run=*.parquet"
        try:
            return con.sql(f"SELECT * FROM read_parquet('{glob}', union_by_name=true)").df()
        except duckdb.IOException as exc:
            if "No files found" in str(exc):
                return None
            raise

    return Sources(list_prefix=list_prefix, object_rows=object_rows, daily_runs=daily_runs)


def publish(
    summary: dict, *, bucket: str = DASHBOARD_BUCKET, client=None, key: str = PUBLIC_KEY,
    archive: bool = True,
) -> list[str]:
    """Write the blob, plus a dated archive copy, to the publish bucket."""
    assert_public(summary)
    if bucket == db.BUCKET:
        raise ValueError(f"{bucket!r} is the raw data bucket; the publish bucket is separate")
    if client is None:
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_DASHBOARD_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_DASHBOARD_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
    body = json.dumps(summary, separators=(",", ":"), ensure_ascii=False).encode()
    meta = {"generated_at": summary["generated_at"]}
    keys = [key]
    client.put_object(Bucket=bucket, Key=key, Body=body,
                      ContentType="application/json", Metadata=meta)
    if archive:
        stamp = datetime.fromisoformat(summary["generated_at"])
        arch = f"public/archive/dt={stamp:%Y-%m-%d}/run={stamp:%Y%m%dT%H%M%SZ}.json"
        client.put_object(Bucket=bucket, Key=arch, Body=body,
                          ContentType="application/json", Metadata=meta)
        keys.append(arch)
    log.info("published %s (%d bytes) to %s", ", ".join(keys), len(body), bucket)
    return keys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the public kenya.tomfr.net summary.")
    ap.add_argument("--out-dir", type=Path,
                    help="write summary.json here and publish nothing")
    ap.add_argument("--daily-root", type=str,
                    help="read coord2/kind=daily_runs from this directory (a v2_daily_run "
                         "--out-dir) instead of R2")
    ap.add_argument("--no-archive", action="store_true", help="skip the dated archive copy")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = db.connect()
    con.execute("SET enable_progress_bar = false")
    summary = build_summary(
        r2_sources(con, args.daily_root), figures.load_recorded(),
        code_version=os.getenv("GIT_SHA") or None,
    )
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        target = args.out_dir / "summary.json"
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"dashboard: wrote {target}")
        return
    keys = publish(summary, archive=not args.no_archive)
    print("dashboard: published " + ", ".join(keys))


if __name__ == "__main__":
    main()
