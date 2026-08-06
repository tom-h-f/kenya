"""Dashboard producer: aggregate the R2 dataset into compact, public-safe JSON
summaries for the kenya.tomfr.net dashboard to read live.

The website is a Cloudflare Worker; it must not run DuckDB or touch raw Parquet.
So the heavy aggregation lives here (reusing the same readers the notebooks use)
and the result is written to a single stable R2 key `dashboard/summary.json`
(plus a timestamped archive). The Worker binds the bucket and serves that file.

    kma-dashboard --once      # build + write once
    kma-dashboard --loop      # forever, ~60 min cadence

Scheduling with no new infra: this runs as its own lightweight service on the
analysis image (`docker compose --profile dashboard up -d dashboard`, beside
enrich on tf1). It reads only PERSISTED artifacts from R2 - no torch, and no
coordination rebuild.

Why it reads the persisted coordination run rather than recomputing it: a
rebuild here would publish a different clustering from the one the collector's
targeting acts on, computed minutes apart from the same code, and two
disagreeing "N clusters" numbers is exactly the failure the census-tuning work
existed to stop. It is also the heaviest job in the repo and would contend with
`enrich` on a 3.7GB host.

PUBLIC-SAFE BY CONSTRUCTION: every field here is an aggregate (counts,
histograms, distributions, cluster keyword names). No account handles, author
ids, post ids or example post text are emitted. `tests/test_dashboard.py`
asserts this against the built payload rather than trusting the claim.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import duckdb
import pandas as pd

from kma import db
from kma.db import BUCKET, connect

log = logging.getLogger("kma.dashboard")

SUMMARY_KEY = "dashboard/summary.json"
SCHEMA_VERSION = 1
DASHBOARD_INTERVAL_S = int(os.getenv("DASHBOARD_INTERVAL_S", "3600"))

# The Worker binds THIS bucket, which holds only the JSON blobs below. An R2
# binding cannot be scoped to a key prefix, so binding the data bucket would put
# the entire raw corpus - handles, post text - one line of Worker code from the
# public internet. Defaults to the data bucket so an unset var degrades to the
# old single-bucket behaviour rather than crashing.
DASHBOARD_BUCKET = os.getenv("R2_DASHBOARD_BUCKET", BUCKET)

# The persisted edge partition matching the run of record. `coordination_run`
# clusters on `co.DEFAULT_EDGE_METHOD` ("bonferroni"), which `EDGE_METHODS`
# writes under `method=svn_bonf`. Reading any other partition would draw a
# network the published cluster count was not derived from.
PERSISTED_METHOD = "svn_bonf"

NET_MIN_SIZE = 4       # ignore coordinating pairs/triples - show real groups
NET_TOP_CLUSTERS = 50  # clusters to draw
NET_MAX_EDGES = 4000   # payload/perf cap (kept by weight)


# `collection_start` costs a listing of the whole posts prefix (31s cold,
# measured 2026-08-05) to produce a value that only moves when a backfill
# reaches further back than we have ever collected. Recomputing it hourly is the
# single largest cost in the build, so it is cached for a day.
_START_TTL_S = 86_400
_start_cache: tuple[float, str | None] | None = None


@contextmanager
def _timed(timings: dict, name: str):
    """Record a section's wall time. The cadence is chosen from these numbers,
    so they are logged rather than guessed at."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter() - t0, 2)
        log.info("section %s: %.2fs", name, timings[name])


# The exact 97.5th percentile of the standard normal, not the 1.96 shorthand.
# At n in the thousands the rounding moves an interval bound in the 5th decimal,
# which is invisible on a chart but makes the statsmodels equivalence test - the
# only thing establishing this implementation is correct - fail.
_Z_975 = 1.959963984540054


def _wilson(k: int, n: int, z: float = _Z_975) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Not the normal approximation: these rates run to a handful of events in a
    day, where a Wald interval routinely covers negative values and understates
    the width. Matches `statsmodels.stats.proportion.proportion_confint(
    method="wilson")`, which the tests check it against."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _f(row, col, default=0.0) -> float:
    v = row.get(col, default)
    return default if v is None or pd.isna(v) else float(v)


def _collection_start(con: duckdb.DuckDBPyConnection, platform: str = "x") -> str | None:
    """When we first collected anything, cached for `_START_TTL_S`.

    `collected_at`, not `created_at`: the oldest authored post in the corpus is
    from 2010, surfaced by a timeline fetch, and publishing that as the start of
    the series would imply 16 years of observation we do not have."""
    global _start_cache
    now = time.time()
    if _start_cache and now - _start_cache[0] < _START_TTL_S:
        return _start_cache[1]
    row = con.sql(f"SELECT min(collected_at) FROM {db.posts_source(platform)}").fetchone()
    value = row[0].isoformat() if row and row[0] else None
    _start_cache = (now, value)
    return value


def build_meta(con: duckdb.DuckDBPyConnection, platform: str = "x") -> dict:
    """The methodology block. The spec requires these rules to be visible on the
    page rather than buried, so they travel with the data instead of being
    hardcoded in the site - a scoping change here cannot leave a stale caveat
    rendered next to a fresh number."""
    return {
        "code_version": os.getenv("GIT_SHA") or None,
        "scope": "baseline",
        "leak_corrected": db.leak_corrected(),
        "leak_note": (
            "Posts promoted to account timelines before 2026-08-01 landed in a "
            "baseline partition; 41,595 posts across 407 accounts, 6.7% of "
            "baseline. Corrected by first-seen type."
        ),
        "collection_start": _collection_start(con, platform),
        "search_horizon_days": 14,
        "horizon_note": (
            "X search reaches back 14 days, so the series begins when collection "
            "began and cannot be backfilled earlier."
        ),
        "excluded_types": list(db.TARGETED_TYPES),
        "coverage_note": (
            "We sample X rather than censusing it. Absence of evidence is not "
            "evidence of absence; recall is bounded, precision is not affected."
        ),
        "model_limits": {
            "hate_classifier": (
                "afro-xlmr, triage threshold p_hate >= 0.28. Blind to the coded "
                "register: on 14 known-coded posts its mean p_hate dropped."
            ),
            "coordination": (
                "Detection is triage, not a verdict. We do not publish that any "
                "account is a bot."
            ),
        },
    }


def _multiplex(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-channel persisted edges into one multiplex edge list.

    Each layer is scaled to unit total weight before summing, mirroring
    `co.aggregate_layers(normalise="mass")`. Without it the ranking that picks
    which NET_MAX_EDGES to draw is decided entirely by the larger layer: measured
    2026-08-05, co_retweet carries 18,784 edges against co_reply's 28."""
    if edges.empty:
        return edges.assign(channels=None, n_channels=0)
    scaled = edges.copy()
    mass = scaled.groupby("channel")["weight"].transform("sum")
    scaled["w"] = scaled["weight"] / mass.where(mass > 0, 1.0)
    agg = scaled.groupby(["src", "dst"], as_index=False).agg(
        weight=("w", "sum"),
        min_gap=("min_gap", "min"),
        channels=("channel", lambda s: sorted(set(s))),
    )
    agg["n_channels"] = agg["channels"].str.len()
    return agg


def _cluster_rows(members: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster from the persisted per-member rows."""
    first = ["size", "n_channels", "internal_edge_share"]
    first += [c for c in ("name", "label", "channels") if c in members.columns]
    return (
        members.groupby("cluster_id", as_index=False)
        .agg({c: "first" for c in first})
        .assign(corr=lambda d: d["n_channels"] >= 2)
    )


def build_coordination(con: duckdb.DuckDBPyConnection, platform: str = "x") -> dict:
    """Statistic C, read from the persisted run of record.

    Corroborated (>= 2 channels) and candidate clusters are kept in separate
    keys and never summed. That is not presentation preference: publishing one
    blended count would claim N validated networks when only the corroborated
    subset is defensible."""
    members = db.coordination_run_latest(con, "clusters", platform).df()
    edges = db.coordination_run_latest(
        con, "edges", platform, method=PERSISTED_METHOD
    ).df()
    metrics = db.coordination_metrics(con, platform).df()

    run: dict = {}
    if len(metrics):
        last = metrics[metrics["computed_at"] == metrics["computed_at"].max()]
        r = last.iloc[0]
        run = {
            "computed_at": str(r.get("computed_at")),
            "code_version": r.get("code_version"),
            "channels": sorted(set(last["channel"].dropna())),
            "method": r.get("method"),
            "resolution": _f(r, "resolution"),
            "min_size": int(_f(r, "min_size")),
            "lookback_days": int(_f(r, "lookback_days")),
        }

    if members.empty:
        return {
            "run": run,
            "headline": {
                "clusters_total": 0, "accounts_total": 0,
                "clusters_corroborated": 0, "accounts_corroborated": 0,
                "edges_total": 0, "edges_multichannel": 0,
                "shown_accounts": 0, "shown_clusters": 0,
            },
            "corroborated": {"clusters": []},
            "candidates": {"n": 0, "size_histogram": []},
            "network": {"nodes": [], "edges": [], "clusters": []},
            "share_series": _share_series(metrics),
        }

    clusters = _cluster_rows(members)
    agg = _multiplex(edges)
    corr_ids = set(clusters.loc[clusters["corr"], "cluster_id"])
    handles = db.curated_handles(platform=platform)

    headline = {
        "clusters_total": int(len(clusters)),
        "accounts_total": int(members["author_id"].nunique()),
        "clusters_corroborated": len(corr_ids),
        "accounts_corroborated": int(members["cluster_id"].isin(corr_ids).sum()),
        "edges_total": int(len(agg)),
        "edges_multichannel": int((agg["n_channels"] >= 2).sum()) if len(agg) else 0,
    }

    drawable = clusters[clusters["size"] >= NET_MIN_SIZE].sort_values(
        ["corr", "size"], ascending=[False, False]
    ).head(NET_TOP_CLUSTERS)
    keep = set(drawable["cluster_id"])
    mem = members[members["cluster_id"].isin(keep)]
    in_net = set(mem["author_id"])
    e = agg[agg["src"].isin(in_net) & agg["dst"].isin(in_net)] if len(agg) else agg
    e = e.sort_values("weight", ascending=False).head(NET_MAX_EDGES)

    candidates = clusters[~clusters["corr"]]
    out = {
        "run": run,
        "headline": headline,
        "corroborated": {
            "clusters": _annotate(clusters[clusters["corr"]], None, handles),
        },
        "candidates": {
            "n": int(len(candidates)),
            "size_histogram": [
                {"size": int(s), "n": int(n)}
                for s, n in candidates["size"].value_counts().sort_index().items()
            ],
        },
        "share_series": _share_series(metrics),
    }

    used = sorted(set(e["src"]) | set(e["dst"])) if len(e) else []
    if not used:
        out["headline"] |= {"shown_accounts": 0, "shown_clusters": 0}
        out["network"] = {"nodes": [], "edges": [], "clusters": []}
        return out

    node_idx = {a: i for i, a in enumerate(used)}
    cid_idx = {c: i for i, c in enumerate(sorted(keep))}
    a2c = dict(zip(mem["author_id"], mem["cluster_id"]))

    # Precompute a static layout so the browser only pans and zooms: no force
    # simulation, and therefore no graph library on the client.
    import igraph as ig

    g = ig.Graph(
        n=len(used),
        edges=[(node_idx[s], node_idx[d]) for s, d in zip(e["src"], e["dst"])],
    )
    layout = g.layout_drl() if g.ecount() else g.layout_circle()
    coords = layout.coords
    deg = g.degree()

    out["network"] = {
        "nodes": [
            {
                "id": i,
                "x": round(coords[i][0], 2),
                "y": round(coords[i][1], 2),
                "c": cid_idx.get(a2c.get(a), -1),
                "d": int(deg[i]),
                "corr": bool(a2c.get(a) in corr_ids),
            }
            for a, i in node_idx.items()
        ],
        "edges": [
            {"s": node_idx[s], "t": node_idx[d], "w": round(float(w), 4), "m": int(nc)}
            for s, d, w, nc in zip(e["src"], e["dst"], e["weight"], e["n_channels"])
        ],
        "clusters": _annotate(drawable, cid_idx, handles),
    }
    out["headline"] |= {
        "shown_accounts": len(node_idx),
        "shown_clusters": len(cid_idx),
    }
    return out


def _safe_name(name, handles: set[str]) -> str | None:
    """Drop a cluster name that carries an account handle.

    Names are c-TF-IDF terms over member post text. `semantic._CLEAN` already
    strips `@mention` and `#hashtag` forms before tokenising, so the residual
    risk is a handle written as bare prose, which survives tokenisation and
    would put an identifiable account on a public page. Cheap to exclude, and
    the loss is one keyword on one cluster."""
    if not isinstance(name, str) or not name.strip():
        return None
    tokens = name.lower().split()
    if handles and any(t in handles for t in tokens):
        return None
    return name


def _annotate(clusters: pd.DataFrame, cid_idx: dict | None, handles: set[str]) -> list[dict]:
    """Cluster cards. `cid_idx` remaps to the opaque drawing index when the
    cluster is in the network; without it the raw per-run cluster_id is used,
    which is fine because it identifies nothing outside this run."""
    out = []
    for _, r in clusters.iterrows():
        cid = r["cluster_id"]
        out.append({
            "id": cid_idx[cid] if cid_idx is not None else int(cid),
            "size": int(_f(r, "size")),
            "n_channels": int(_f(r, "n_channels")),
            "channels": list(r.get("channels") or []),
            "internal_edge_share": round(_f(r, "internal_edge_share"), 3),
            "name": _safe_name(r.get("name"), handles),
            "corr": bool(r.get("corr", False)),
        })
    return out


def _share_series(metrics: pd.DataFrame) -> list[dict]:
    """Detection counters over time, carrying `code_version` on every point.

    The site segments on it rather than drawing one continuous line: these
    counters are not comparable across a methodology change, and smoothing over
    one is the exact error that made the census-tuning Q2 figure wrong."""
    if metrics.empty:
        return []
    per_run = (
        metrics.groupby("computed_at", as_index=False)
        .agg({
            "code_version": "first",
            "n_clusters": "first",
            "n_accounts": "first",
            "n_corroborated_clusters": "first",
            "n_corroborated_accounts": "first",
        })
        .sort_values("computed_at")
    )
    return [
        {
            "computed_at": str(r["computed_at"]),
            "code_version": r["code_version"],
            "n_clusters": int(_f(r, "n_clusters")),
            "n_accounts": int(_f(r, "n_accounts")),
            "n_corroborated_clusters": int(_f(r, "n_corroborated_clusters")),
            "n_corroborated_accounts": int(_f(r, "n_corroborated_accounts")),
        }
        for _, r in per_run.iterrows()
    ]


def build_summary(con: duckdb.DuckDBPyConnection, platform: str = "x") -> dict:
    """The public blob. `toxicity`, `share_of_voice` and `targeting` are null
    until their phases land; the site reserves layout for them rather than
    shifting when they appear."""
    timings: dict = {}
    with _timed(timings, "meta"):
        meta = build_meta(con, platform)
    with _timed(timings, "coordination"):
        coordination = build_coordination(con, platform)
    return {
        "schema_version": SCHEMA_VERSION,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform,
        "meta": meta,
        "coordination": coordination,
        "toxicity": None,
        "share_of_voice": None,
        "targeting": None,
        "build": {"timings_s": timings},
    }


def _r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def persist_summary(
    summary: dict, key: str = SUMMARY_KEY, bucket: str = DASHBOARD_BUCKET
) -> str:
    """Write to the stable key plus a timestamped archive. `computed_at` also
    goes on the object metadata so the Worker's health route can check freshness
    with a HEAD instead of pulling the body."""
    body = json.dumps(summary, default=str, separators=(",", ":")).encode()
    now = datetime.now(timezone.utc)
    kind = key.rsplit("/", 1)[-1].removesuffix(".json")
    client = _r2_client()
    meta = {"computed_at": summary.get("computed_at", now.isoformat())}
    client.put_object(
        Bucket=bucket, Key=key, Body=body,
        ContentType="application/json", Metadata=meta,
    )
    archive = f"dashboard/archive/kind={kind}/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.json"
    client.put_object(
        Bucket=bucket, Key=archive, Body=body,
        ContentType="application/json", Metadata=meta,
    )
    log.info("wrote %s (%d bytes) + %s", key, len(body), archive)
    return key


def run_once(con: duckdb.DuckDBPyConnection | None = None, platform: str = "x") -> str:
    con = con or connect()
    return persist_summary(build_summary(con, platform=platform))


def run_loop(interval_s: int = DASHBOARD_INTERVAL_S, platform: str = "x") -> None:
    """Forever: rebuild, then sleep. A per-cycle failure is logged and retried so
    a transient R2 hiccup never kills the container."""
    while True:
        try:
            run_once(platform=platform)
        except Exception:
            log.exception("dashboard refresh failed")
        time.sleep(interval_s)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description="Build the kenya.tomfr.net summary.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--loop", action="store_true", help="run forever")
    mode.add_argument("--once", action="store_true", help="build + write once")
    ap.add_argument("--platform", default="x")
    ap.add_argument("--dry-run", action="store_true", help="build, print, write nothing")
    args = ap.parse_args()
    if args.loop:
        run_loop(platform=args.platform)
    elif args.dry_run:
        print(json.dumps(build_summary(connect(), platform=args.platform), default=str))
    else:
        print(f"dashboard: wrote {run_once(platform=args.platform)}")


if __name__ == "__main__":
    main()
