"""Explorer producer: persisted daily output and window posts in, one bundle out.

    uv run python -m kma.explorer_export --out-dir /tmp/explorer   # build, validate, write here only
    uv run python -m kma.explorer_export                            # build, validate, publish

Feeds the explorer pages of kenya.tomfr.net. The contract is
`docs/plans/explorer-contract/schema.json`; the design is
`docs/plans/2026-10-10-explorer.md`. Nothing here recomputes a network: the
fused edges and the trace rows the daily run persisted (`kind=daily_edges`,
`kind=daily_traces`) are read, and the window's posts are read once into a
DuckDB table for everything that needs text, tags, links or counts.

## Order of work, and why a failure publishes nothing

1. Read the persisted daily tables and the window's posts. Any unreadable
   source raises here, before a byte is written anywhere.
2. Build every file in memory, apply the denylist, stage the files under a
   directory as they would sit in the bucket, and write `manifest.json` last.
3. `assert_explorer_safe` over the staged tree: the contract validator (schema,
   cross-file checks, forbidden field names, denylist), the language lint over
   the strings this module writes, and the id checks no schema can express.
4. Only then, and only without `--out-dir`, upload `runs/{run}/...`, then the
   manifest, then prune runs past the retention.

## Where it writes

`R2_DASHBOARD_BUCKET` with the dashboard token, under `explorer/v1/`. Reads of
the data bucket use the read key. The raw data bucket is refused as a target.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import logging
import os
import re
import resource
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from kma import bench, canary, coord2, coord2_communities, dashboard, db, explorer_build as eb, v2_daily

log = logging.getLogger("kma.explorer")

PREFIX = "explorer/v1"
RETENTION = 30
PLATFORM = "x"
DENYLIST_PATH = db.MONOREPO_ROOT / "explorer" / "denylist.txt"
CONTRACT_DIR = db.MONOREPO_ROOT / "docs" / "plans" / "explorer-contract"
RUN_FILE_CACHE = "public, max-age=31536000, immutable"
MANIFEST_CACHE = "public, max-age=60, s-maxage=300"
UPLOAD_THREADS = 8
TEXT_POSTS_PER_USER = 100
TEXT_MIN_WORDS = coord2.MIN_TEXT_WORDS

UNAVAILABLE = (
    {"view": "topic_index", "reason": "Topics are not part of this release."},
    {"view": "time_coverage", "reason": "Coverage by stage is not part of this release."},
)

_RUN_ID = re.compile(r"^\d{8}T\d{6}Z$")
DATA_KEYED_PARENTS = frozenset({"tags"})


class ExplorerUnsafe(ValueError):
    """The staged bundle failed a check; nothing was published."""


def load_contract_module(name: str):
    path = CONTRACT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_explorer_contract_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_denylist(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def to_json_bytes(doc: Any) -> bytes:
    def default(value: Any):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        raise TypeError(f"{type(value).__name__} is not JSON serialisable")

    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
                      default=default).encode()


# --------------------------------------------------------------------------
# Reading.
# --------------------------------------------------------------------------


def daily_uri(root: str, kind: str, run_id: str) -> str:
    stamp = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return f"{root}/{v2_daily.daily_key(kind, stamp)}"


def read_run_table(con: duckdb.DuckDBPyConnection, root: str, kind: str, run_id: str) -> pd.DataFrame:
    return con.sql(f"SELECT * FROM read_parquet('{daily_uri(root, kind, run_id)}')").df()


def read_runs(con: duckdb.DuckDBPyConnection, root: str) -> pd.DataFrame:
    glob = f"{root}/coord2/platform={PLATFORM}/kind=daily_runs/dt=*/run=*.parquet"
    return con.sql(f"SELECT * FROM read_parquet('{glob}', union_by_name=true)").df()


def pick_run(runs: pd.DataFrame, run_id: str | None) -> dict:
    if runs.empty:
        raise RuntimeError("no daily run has been persisted")
    if run_id:
        hit = runs[runs["run_id"] == run_id]
        if hit.empty:
            raise RuntimeError(f"no daily run {run_id}")
        return hit.iloc[0].to_dict()
    return runs.sort_values("computed_at").iloc[-1].to_dict()


def read_daily(con: duckdb.DuckDBPyConnection, root: str, run_id: str | None) -> eb.Daily:
    runs = read_runs(con, root)
    run = pick_run(runs, run_id)
    rid = str(run["run_id"])
    try:
        edges = read_run_table(con, root, "edges", rid)
    except duckdb.Error as exc:
        raise RuntimeError(
            f"run {rid} has no daily_edges: it predates edge persistence, so the explorer "
            "cannot be built from it without rebuilding the networks") from exc
    try:
        traces = read_run_table(con, root, "traces", rid)
    except duckdb.Error:
        traces = pd.DataFrame({c: pd.Series(dtype=str) for c in ("user_id", "trace", "entity")})
        log.warning("run %s has no daily_traces; evidence will be empty", rid)
    return eb.Daily(
        run=run,
        communities=read_run_table(con, root, "communities", rid),
        members=read_run_table(con, root, "members", rid).astype({"user_id": str}),
        edges=edges.astype({"source": str, "target": str}),
        traces=traces.astype({"user_id": str, "entity": str}),
        runs=runs,
    )


def load_window(con: duckdb.DuckDBPyConnection, window_id: str) -> tuple[eb.Corpus, int]:
    """The window's distinct posts into `_win`. Raises if the manifest or posts are unreadable."""
    manifest = bench.load(window_id, con=con)
    posts = manifest[manifest["prefix"] == "posts"]
    if posts.empty:
        raise RuntimeError(f"window {window_id} pins no posts objects")
    post_rows = int(pd.to_numeric(posts["rows"], errors="coerce").sum())
    source = bench.pinned_source(manifest, "posts")
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE _win AS
            SELECT platform_post_id AS post_id, author_id, author_handle, created_at, text, url,
                   in_reply_to_id, coalesce(is_repost, repost_of_id IS NOT NULL) AS is_repost,
                   repost_of_id, like_count, reply_count, repost_count, hashtags, urls, collected_at
            FROM {source}
            QUALIFY row_number() OVER (PARTITION BY platform_post_id
                                       ORDER BY collected_at DESC) = 1"""
    )
    n = con.execute("SELECT count(*) FROM _win").fetchone()[0]
    if not n:
        raise RuntimeError(f"window {window_id} holds no readable posts")
    log.info("window %s: %d distinct posts (%d rows)", window_id, n, post_rows)
    return manifest, post_rows


def read_profiles(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Latest profile snapshot of every member. Display name and bio, never location."""
    return con.execute(
        f"""SELECT platform_user_id AS id, handle, display_name, bio,
                   followers_count AS followers, following_count AS following,
                   tweet_count AS statuses, created_at, collected_at AS profile_as_of
            FROM {db.authors_source(PLATFORM)}
            WHERE platform_user_id IN (SELECT id FROM _mem)
            QUALIFY row_number() OVER (PARTITION BY platform_user_id
                                       ORDER BY collected_at DESC) = 1"""
    ).df()


def read_kenya(con: duckdb.DuckDBPyConnection, ids: list[str]) -> pd.DataFrame:
    return coord2_communities.model_kenya_share(
        con, "SELECT author_id AS user_id, post_id FROM _win", ids)


def read_trend(con: duckdb.DuckDBPyConnection) -> pd.DataFrame | None:
    glob = f"r2://{db.BUCKET}/trend_candidates/platform={PLATFORM}/dt=*/run=*.parquet"
    try:
        return con.sql(
            f"""SELECT run_id, tag, recent_rate, prior_rate, emergence, posts, censused
                FROM read_parquet('{glob}', union_by_name=true)
                QUALIFY run_id = max(run_id) OVER ()"""
        ).df()
    except duckdb.Error:
        log.exception("trend_candidates unreadable")
        return None


def read_control_arm(con: duckdb.DuckDBPyConnection) -> dict | None:
    glob = f"r2://{db.BUCKET}/control_runs/platform={PLATFORM}/dt=*/run=*.parquet"
    try:
        row = con.sql(
            f"""SELECT count(*) AS runs, coalesce(sum(posts), 0) AS posts,
                       count(*) FILTER (truncated) AS truncated_runs,
                       CAST(min(window_start) AS DATE) AS first, CAST(max(window_start) AS DATE) AS last
                FROM read_parquet('{glob}', union_by_name=true)"""
        ).fetchone()
    except duckdb.Error:
        log.exception("control_runs unreadable")
        return None
    if not row or not row[0]:
        return None
    return {"runs": int(row[0]), "posts": int(row[1]), "truncated_runs": int(row[2]),
            "first": row[3].isoformat(), "last": row[4].isoformat()}


def collection_series(src: dashboard.Sources, today: date, days: int = dashboard.COLLECTION_DAYS) -> dict:
    """Rows per UTC collection day and posts/ type partition, and the gap days."""
    posts = src.list_prefix(f"posts/platform={PLATFORM}/")
    observed = sorted(set(posts["dt"].dropna()))
    if not observed:
        raise RuntimeError("posts/ lists no partitions")
    first = date.fromisoformat(observed[0])
    seen = set(observed)
    span = [first + timedelta(days=i) for i in range((today - timedelta(days=1) - first).days + 1)]
    gaps = [d.isoformat() for d in span if d.isoformat() not in seen]
    window = dashboard._closed_days(today, days)
    wanted = {d.isoformat() for d in window}
    inside = posts[posts["dt"].isin(wanted)]
    counts = src.object_rows(list(inside["key"]))
    inside = inside.assign(rows=inside["key"].map(counts))
    if inside["rows"].isna().any():
        raise ValueError("object_rows did not answer every posts key")
    series = []
    for d in window:
        day = inside[inside["dt"] == d.isoformat()]
        by_type = day.groupby(day["type"].fillna("unknown"))["rows"].sum()
        series.append({"dt": d.isoformat(), "rows": int(day["rows"].sum()),
                       "by_type": {str(k): int(v) for k, v in by_type.items()}})
    return {"series": series, "gap_days": gaps}


# --------------------------------------------------------------------------
# Text-similarity evidence, from embeddings.
# --------------------------------------------------------------------------


def text_evidence_reader(
    con: duckdb.DuckDBPyConnection, manifest: pd.DataFrame, start: date, threshold: float,
    min_overlap: float,
) -> eb.TextEvidence:
    """Recompute the post pairs behind text-similarity edges.

    The persisted trace is pair-level, so the posts are found again: for each
    edge, every pair of the two accounts' posts whose embedding cosine reaches
    the run's threshold and whose cleaned text overlaps by the run's floor.
    Only the accounts on text edges are read, and only embeddings in objects
    dated inside the window, since a post collected in the window cannot have
    been embedded before it."""
    emb = manifest[manifest["prefix"] == "embeddings"]
    emb = emb[emb["dt"].astype(str) >= start.isoformat()] if "dt" in emb.columns else emb
    paths = sorted(emb["path"].dropna().tolist())

    def read(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[int, list[dict]]]:
        if not pairs or not paths:
            return {}
        users = sorted({u for p in pairs for u in p})
        con.register("_text_users", pd.DataFrame({"id": users}))
        try:
            posts = con.execute(
                f"""SELECT author_id, post_id, text, created_at FROM (
                      SELECT w.author_id, w.post_id, w.text, w.created_at,
                             row_number() OVER (PARTITION BY w.author_id
                                                ORDER BY w.created_at DESC) AS rk
                      FROM _win w JOIN _text_users u ON u.id = w.author_id
                      WHERE NOT coalesce(w.is_repost, false) AND w.text IS NOT NULL)
                    WHERE rk <= {TEXT_POSTS_PER_USER}"""
            ).df()
        finally:
            con.unregister("_text_users")
        posts = posts[posts["text"].map(lambda t: len(coord2.clean_text(t).split()) >= TEXT_MIN_WORDS)]
        con.register("_text_posts", posts[["post_id"]].rename(columns={"post_id": "id"}))
        listed = ", ".join(f"'{p}'" for p in paths)
        try:
            vecs = con.execute(
                f"""SELECT platform_post_id AS id, embedding FROM
                      read_parquet([{listed}], union_by_name=true)
                    WHERE platform_post_id IN (SELECT id FROM _text_posts)
                    QUALIFY row_number() OVER (PARTITION BY platform_post_id
                                               ORDER BY embedded_at DESC) = 1"""
            ).df()
        finally:
            con.unregister("_text_posts")
        if vecs.empty:
            return {}
        matrix = np.vstack(vecs["embedding"].to_numpy()).astype(np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        where = dict(zip(vecs["id"], range(len(vecs))))
        posts = posts[posts["post_id"].isin(where)]
        by_user = {u: g for u, g in posts.groupby("author_id")}
        cleaned = {pid: coord2.clean_text(t) for pid, t in zip(posts["post_id"], posts["text"])}
        out: dict[tuple[str, str], tuple[int, list[dict]]] = {}
        for a, b in pairs:
            ga, gb = by_user.get(a), by_user.get(b)
            if ga is None or gb is None:
                continue
            ia = [where[p] for p in ga["post_id"]]
            ib = [where[p] for p in gb["post_id"]]
            sims = matrix[ia] @ matrix[ib].T
            hits = np.argwhere(sims >= threshold)
            found = []
            for x, y in hits:
                pa_, pb_ = ga["post_id"].iloc[x], gb["post_id"].iloc[y]
                if coord2.text_overlap(cleaned[pa_], cleaned[pb_]) >= min_overlap:
                    found.append((float(sims[x, y]), pa_, pb_))
            if found:
                found.sort(key=lambda f: (-f[0], f[1], f[2]))
                out[(a, b)] = (len(found), [
                    {"trace": "text_similarity", "post_a": pa_, "post_b": pb_,
                     "similarity": round(s, 4)} for s, pa_, pb_ in found[:eb.EVIDENCE_ITEMS]])
        return out

    return read


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------


@dataclass
class Inputs:
    daily: eb.Daily
    profiles_for: Callable[[], pd.DataFrame]
    kenya_for: Callable[[list[str]], pd.DataFrame]
    corpus: eb.Corpus
    post_rows: int | None
    trend: pd.DataFrame | None
    collection: dict | None
    control_arm: dict | None
    previous: eb.Previous | None
    text_evidence: eb.TextEvidence | None
    denylist: set[str]
    now: datetime


def apply_denylist(daily: eb.Daily, corpus: eb.Corpus, deny: set[str]) -> set[str]:
    """Remove denied ids from every table before building. Returns the ids of
    posts they authored, which evidence must not name either."""
    if not deny:
        return set()
    daily.members = daily.members[~daily.members["user_id"].isin(deny)].reset_index(drop=True)
    daily.edges = daily.edges[~(daily.edges["source"].isin(deny) | daily.edges["target"].isin(deny))
                              ].reset_index(drop=True)
    daily.traces = daily.traces[~daily.traces["user_id"].isin(deny)].reset_index(drop=True)
    corpus.con.register("_deny", pd.DataFrame({"id": sorted(deny)}))
    try:
        posts = {r[0] for r in corpus.con.execute(
            "SELECT post_id FROM _win WHERE author_id IN (SELECT id FROM _deny)").fetchall()}
        corpus.con.execute("DELETE FROM _win WHERE author_id IN (SELECT id FROM _deny)")
    finally:
        corpus.con.unregister("_deny")
    return posts


def assemble(inp: Inputs) -> eb.Bundle:
    """Every file of the bundle, keyed by path under `runs/{run}/`."""
    daily = inp.daily
    corpus = inp.corpus
    deny = inp.denylist
    hidden_posts = apply_denylist(daily, corpus, deny)
    stamp = eb.make_stamp(daily.run, inp.now)
    bundle = eb.Bundle()

    kept_cids = set(daily.communities.loc[daily.communities["kept"], "community"].astype(int))
    members = daily.members[daily.members["community"].isin(kept_cids)]
    members = members[~members["user_id"].map(canary.is_canary)].reset_index(drop=True)
    corpus.set_members(members)

    all_members = daily.members[~daily.members["user_id"].map(canary.is_canary)]
    lineages = eb.current_lineages(daily, all_members, inp.previous, deny)
    rows = eb.build_rows(daily, members, corpus, lineages)

    profiles = inp.profiles_for()
    kenya = inp.kenya_for(list(members["user_id"]))
    fallback = corpus.per_author().set_index("id")["handle"].to_dict()
    handles: dict[str, str] = {}
    if not profiles.empty:
        handles.update({i: h for i, h in zip(profiles["id"], profiles["handle"]) if isinstance(h, str) and h})
    missing = 0
    for uid in members["user_id"]:
        if uid not in handles:
            h = fallback.get(uid)
            handles[uid] = h if isinstance(h, str) and h else uid
            missing += 1
    bundle.stats["members_without_profile_handle"] = missing

    cid_of = dict(zip(members["user_id"], members["community"].astype(int)))
    kept_edges = eb.edge_tables(daily.edges, cid_of)

    files: dict[str, Any] = {}
    graphs, index = eb.build_graphs(stamp, members, kept_edges, handles, rows)
    files.update(graphs)
    for cid, r in rows.items():
        if r["kept"]:
            local = kept_edges[kept_edges["intra"] & (kept_edges["cs"] == cid)]
            r["edges"] = int(len(local))

    evidence, extra, ev_stats = eb.build_evidence(
        stamp, index, daily.traces, corpus, inp.text_evidence, hidden_posts)
    files.update(evidence)
    bundle.stats.update(ev_stats)
    bundle.extra_authors |= extra

    cards, amplified = eb.build_accounts(
        stamp, members, kept_edges, daily.edges, handles, profiles, kenya, corpus, lineages)
    files.update(cards)
    bundle.extra_authors |= amplified

    files.update(eb.build_community_details(stamp, members, rows, handles, kenya, corpus, index))
    files["communities.json"] = {
        "kind": "community_index", "stamp": stamp,
        "communities": [rows[c] for c in sorted(rows)]}
    files["search/accounts.json"] = eb.build_search(stamp, members, handles, profiles)

    tag_files, tag_authors = eb.build_hashtags(stamp, corpus, inp.trend)
    files.update(tag_files)
    bundle.extra_authors |= tag_authors
    files["links/index.json"] = eb.build_links(stamp, corpus)

    totals = corpus.totals()
    files["overview.json"] = eb.build_overview(stamp, daily, rows, members, kept_edges, totals, inp.post_rows)
    files["time/pipeline.json"] = eb.build_pipeline(stamp, daily.runs)

    if inp.collection is not None:
        files["time/collection.json"] = {
            "kind": "time_collection", "stamp": stamp, **inp.collection,
            **({"control_arm": inp.control_arm} if inp.control_arm else {})}
    else:
        bundle.unavailable.append({"view": "time_collection",
                                   "reason": dashboard.REASON_UNREADABLE})

    cur = eb.communities_of(daily.members, daily.communities, deny)
    files["time/changes.json"] = eb.build_changes(stamp, cur, inp.previous, deny, lineages)

    bundle.files = files
    bundle.unavailable = [dict(u) for u in UNAVAILABLE] + bundle.unavailable
    bundle.members = set(members["user_id"])
    bundle.stats.update({
        "accounts": len(members), "communities_kept": len(kept_cids),
        "communities": len(rows), "edges_fused_kept": len(kept_edges),
        "edges_drawn": int(kept_edges["drawn"].sum()),
        "nodes_drawn": len(files["graph/all.json"]["nodes"]["id"]),
        "edges_drawn_all_graph": len(files["graph/all.json"]["edges"]["s"]),
    })
    return bundle


# --------------------------------------------------------------------------
# Staging, safety, sizes.
# --------------------------------------------------------------------------


def stage(bundle: eb.Bundle, run: dict, root: Path, previous_runs: list[dict],
          generated_at: datetime) -> dict:
    """Write the run's files under `root/explorer/v1/` and the manifest last. Returns the manifest."""
    run_id = str(run["run_id"])
    base = root / PREFIX / "runs" / run_id
    meta: dict[str, dict] = {}
    for rel, doc in sorted(bundle.files.items()):
        data = to_json_bytes(doc)
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        meta[rel] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    end = str(run["window_end"])[:10]
    this = {"run_id": run_id, "window_id": str(run["window_id"]), "window_end": end,
            "computed_at": eb.iso(run["computed_at"])}
    runs = [this] + [r for r in previous_runs if r["run_id"] != run_id]
    manifest = {
        "kind": "manifest", "schema_version": eb.SCHEMA_VERSION, "latest_run": run_id,
        "generated_at": eb.iso(generated_at), "method_url": eb.METHOD_URL,
        "corrections_url": eb.CORRECTIONS_URL, "runs": runs[:RETENTION], "files": meta,
        "unavailable": bundle.unavailable,
    }
    (root / PREFIX).mkdir(parents=True, exist_ok=True)
    (root / PREFIX / "manifest.json").write_bytes(to_json_bytes(manifest))
    return manifest


def assert_explorer_safe(root: Path, bundle: eb.Bundle, denylist: set[str]) -> None:
    """Refuse a staged bundle that breaks the contract or a guardrail.

    The contract validator covers schema, layout, manifest hashes, forbidden
    field names, dropped communities and the denylist. The language lint covers
    the strings this producer writes (user content is skipped). The id checks
    cover what no schema can: that every account named is a member of a kept
    community or an author deliberately quoted as evidence."""
    problems: list[str] = []
    validator = load_contract_module("validate_bundle")
    for line in validator.validate_bundle(root, denylist):
        if re.search(r"/tags/[^/]+$", line) and "forbidden field name" in line:
            continue
        problems.append(line)

    linter = load_contract_module("lint_language")
    allow = linter.load_allowlist(CONTRACT_DIR / "lint-allowlist.json")
    problems += linter.lint_paths([root / PREFIX], allow, include_user_content=False)

    allowed = bundle.members | bundle.extra_authors
    for rel, doc in bundle.files.items():
        problems += _id_problems(rel, doc, bundle.members, allowed)
    if denylist:
        blob_hits = [rel for rel, doc in bundle.files.items()
                     if any(d in json.dumps(doc) for d in denylist)]
        problems += [f"{rel}: denylisted id present" for rel in blob_hits]
    if problems:
        raise ExplorerUnsafe(f"{len(problems)} problem(s), first: " + "; ".join(problems[:8]))


def _id_problems(rel: str, doc: Any, members: set[str], allowed: set[str]) -> list[str]:
    out: list[str] = []

    def check(ids, only_members: bool, what: str) -> None:
        pool = members if only_members else allowed
        bad = [i for i in ids if i not in pool]
        if bad:
            out.append(f"{rel}: {what} names {len(bad)} account(s) outside the allowed set, e.g. {bad[0]}")

    kind = doc.get("kind")
    if kind in ("graph",):
        check(doc["nodes"]["id"], True, "graph nodes")
    elif kind == "community_detail":
        check(doc["members"]["id"], True, "members")
        check([a["id"] for a in doc["amplified"]], False, "amplified")
    elif kind == "search_accounts":
        check(doc["id"], True, "search")
    elif kind == "account_shard":
        check(list(doc["accounts"]), True, "account cards")
        for card in doc["accounts"].values():
            check([n["id"] for n in card["neighbours"]], True, "neighbours")
            check([a["id"] for a in card["amplifies"]], False, "amplifies")
            check([p["author_id"] for p in card["recent_posts"]], True, "recent posts")
    elif kind == "evidence":
        check([p["author_id"] for p in doc["posts"].values()], False, "evidence posts")
    elif kind == "hashtag_shard":
        for t in doc["tags"].values():
            check([p["author_id"] for p in t["examples"]], True, "tag examples")
    return out


def size_report(root: Path) -> dict:
    """Raw and gzip bytes per top-level group and in total."""
    groups: dict[str, dict] = defaultdict(lambda: {"files": 0, "raw": 0, "gzip": 0})
    for path in sorted((root / PREFIX).rglob("*.json")):
        rel = path.relative_to(root / PREFIX)
        if rel.parts[0] != "runs":
            continue
        sub = Path(*rel.parts[2:])
        group = sub.parts[0] if len(sub.parts) > 1 else sub.stem
        data = path.read_bytes()
        g = groups[group]
        g["files"] += 1
        g["raw"] += len(data)
        g["gzip"] += len(gzip.compress(data, compresslevel=6))
    total = {k: sum(g[k] for g in groups.values()) for k in ("files", "raw", "gzip")}
    return {"groups": dict(groups), "total": total}


# --------------------------------------------------------------------------
# Publishing.
# --------------------------------------------------------------------------


def r2_client(dashboard_key: bool = True):
    import boto3

    prefix = "R2_DASHBOARD_" if dashboard_key else "R2_"
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ[f"{prefix}ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ[f"{prefix}SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def read_published_manifest(client, bucket: str) -> dict | None:
    try:
        body = client.get_object(Bucket=bucket, Key=f"{PREFIX}/manifest.json")["Body"].read()
    except client.exceptions.NoSuchKey:
        return None
    return json.loads(body)


def read_published_lineage(client, bucket: str, run_id: str) -> dict[int, str] | None:
    try:
        body = client.get_object(Bucket=bucket, Key=f"{PREFIX}/runs/{run_id}/communities.json")["Body"].read()
    except client.exceptions.NoSuchKey:
        return None
    return {int(c["cid"]): c["lineage"] for c in json.loads(body)["communities"]}


def publish(root: Path, manifest: dict, *, bucket: str, client, run_id: str) -> dict:
    """Upload the run, then the manifest, then prune old runs. The manifest is
    the only object a reader follows, so a failure before it leaves the
    previous run live and the half-written one unreferenced."""
    if bucket == db.BUCKET:
        raise ValueError(f"{bucket!r} is the raw data bucket; the publish bucket is separate")
    if not _RUN_ID.match(run_id):
        raise ValueError(f"bad run id {run_id!r}")
    base = root / PREFIX / "runs" / run_id
    files = sorted(p for p in base.rglob("*") if p.is_file())

    def put(path: Path) -> None:
        key = f"{PREFIX}/runs/{run_id}/{path.relative_to(base).as_posix()}"
        client.put_object(Bucket=bucket, Key=key, Body=path.read_bytes(),
                          ContentType="application/json", CacheControl=RUN_FILE_CACHE)

    with ThreadPoolExecutor(UPLOAD_THREADS) as pool:
        list(pool.map(put, files))
    client.put_object(Bucket=bucket, Key=f"{PREFIX}/manifest.json",
                      Body=(root / PREFIX / "manifest.json").read_bytes(),
                      ContentType="application/json", CacheControl=MANIFEST_CACHE)
    pruned = prune(client, bucket, keep={r["run_id"] for r in manifest["runs"]})
    log.info("published %d files for %s, pruned %d object(s)", len(files), run_id, pruned)
    return {"uploaded": len(files), "pruned": pruned}


def prune(client, bucket: str, keep: set[str]) -> int:
    """Delete `runs/{id}/` for runs outside the manifest. Touches nothing else."""
    base = f"{PREFIX}/runs/"
    deleted = 0
    runs: set[str] = set()
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=base, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            runs.add(entry["Prefix"][len(base):].rstrip("/"))
    for run in sorted(runs):
        if not _RUN_ID.match(run) or run in keep:
            continue
        keys = []
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{base}{run}/"):
            keys += [{"Key": o["Key"]} for o in page.get("Contents", [])]
        for i in range(0, len(keys), 1000):
            client.delete_objects(Bucket=bucket, Delete={"Objects": keys[i:i + 1000]})
        deleted += len(keys)
    return deleted


# --------------------------------------------------------------------------
# Run.
# --------------------------------------------------------------------------


def run(
    *, out_dir: Path | None, daily_root: str | None, run_id: str | None, denylist: set[str],
    duckdb_memory: str | None, text_evidence: bool, bucket: str = dashboard.DASHBOARD_BUCKET,
    client=None, con: duckdb.DuckDBPyConnection | None = None,
) -> dict:
    t0 = time.monotonic()
    timings: dict[str, float] = {}
    if out_dir is not None and (out_dir / PREFIX).exists():
        raise RuntimeError(f"{out_dir / PREFIX} already exists; use an empty --out-dir")
    con = con or db.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute("SET TimeZone = 'UTC'")
    if duckdb_memory:
        con.execute(f"SET memory_limit = '{duckdb_memory}'")
    con.execute(f"SET temp_directory = '{tempfile.gettempdir()}/kma-explorer-duckdb'")
    root = daily_root.rstrip("/") if daily_root else f"r2://{db.BUCKET}"
    now = datetime.now(timezone.utc).replace(microsecond=0)

    def timed(name: str, fn):
        t = time.monotonic()
        out = fn()
        timings[name] = round(time.monotonic() - t, 1)
        return out

    daily = timed("daily_s", lambda: read_daily(con, root, run_id))
    run_row = daily.run
    log.info("run %s window %s", run_row["run_id"], run_row["window_id"])
    end = date.fromisoformat(str(run_row["window_end"])[:10])
    start = end - timedelta(days=int(run_row["window_days"]) - 1)
    manifest_df, post_rows = timed("window_s", lambda: load_window(con, str(run_row["window_id"])))
    corpus = eb.Corpus(con, start, end)

    published = None
    if out_dir is None:
        client = client or r2_client()
        published = read_published_manifest(client, bucket)
    previous_runs = list(published["runs"]) if published else []
    previous = None
    if published:
        prev_id = next((r["run_id"] for r in previous_runs if r["run_id"] != str(run_row["run_id"])), None)
        if prev_id:
            try:
                previous = eb.Previous(
                    run_id=prev_id,
                    communities=read_run_table(con, root, "communities", prev_id),
                    members=read_run_table(con, root, "members", prev_id).astype({"user_id": str}),
                    lineage=read_published_lineage(client, bucket, prev_id),
                )
            except duckdb.Error:
                log.exception("previous run %s unreadable; no change view", prev_id)

    src = dashboard.r2_sources(con)
    collection = None
    try:
        today = datetime.fromisoformat(str(eb.iso(run_row["computed_at"]))[:-1] + "+00:00").date()
        collection = timed("collection_s", lambda: collection_series(src, today))
    except Exception:
        log.exception("collection volume unreadable")
    control_arm = timed("control_s", lambda: read_control_arm(con))
    trend = timed("trend_s", lambda: read_trend(con))

    text_reader = None
    if text_evidence:
        text_reader = text_evidence_reader(
            con, manifest_df, start, float(run_row["text_threshold"]),
            float(run_row["text_overlap"]))

    inp = Inputs(
        daily=daily, profiles_for=lambda: timed("profiles_s", lambda: read_profiles(con)),
        kenya_for=lambda ids: timed("kenya_s", lambda: read_kenya(con, ids)),
        corpus=corpus, post_rows=post_rows, trend=trend, collection=collection,
        control_arm=control_arm, previous=previous, text_evidence=text_reader,
        denylist=denylist, now=now,
    )
    bundle = timed("assemble_s", lambda: assemble(inp))

    stage_root = out_dir or Path(tempfile.mkdtemp(prefix="kma-explorer-"))
    manifest = timed("stage_s", lambda: stage(bundle, run_row, stage_root, previous_runs, now))
    timed("safety_s", lambda: assert_explorer_safe(stage_root, bundle, denylist))
    sizes = size_report(stage_root)

    result = {
        "run_id": str(run_row["run_id"]), "window_id": str(run_row["window_id"]),
        "stats": bundle.stats, "sizes": sizes, "seconds": timings,
        "wall_s": round(time.monotonic() - t0, 1), "peak_rss_mb": peak_rss_mb(),
        "staged_at": str(stage_root),
    }
    if out_dir is None:
        timed("publish_s", lambda: publish(
            stage_root, manifest, bucket=bucket, client=client, run_id=str(run_row["run_id"])))
        result["wall_s"] = round(time.monotonic() - t0, 1)
        result["seconds"] = timings
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the kenya.tomfr.net explorer bundle.")
    ap.add_argument("--out-dir", type=Path,
                    help="stage the bundle here and publish nothing; must not hold explorer/v1 yet")
    ap.add_argument("--daily-root", type=str,
                    help="read coord2/kind=daily_* from this directory (a v2_daily_run --out-dir)")
    ap.add_argument("--run-id", type=str, help="build this daily run, default the newest")
    ap.add_argument("--denylist", type=Path, default=DENYLIST_PATH)
    ap.add_argument("--duckdb-memory", type=str, default="2GB")
    ap.add_argument("--no-text-evidence", action="store_true",
                    help="skip the embedding read behind text-similarity evidence")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = run(
        out_dir=args.out_dir, daily_root=args.daily_root, run_id=args.run_id,
        denylist=read_denylist(args.denylist), duckdb_memory=args.duckdb_memory,
        text_evidence=not args.no_text_evidence,
    )
    print(json.dumps(out, default=str, indent=2))


if __name__ == "__main__":
    main()
