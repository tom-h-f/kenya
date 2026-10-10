"""Explorer bundle builders: persisted daily tables and window posts in, JSON out.

Everything here is a function of its arguments. `kma.explorer_export` reads R2
and publishes; this module never touches R2. The shape of every file is the
contract in `docs/plans/explorer-contract/schema.json`.

Window posts live in a DuckDB table `_win` (one row per distinct post, the
latest collected snapshot), which `Corpus` owns. Aggregates over it are SQL;
per-account and per-edge assembly is dict and numpy work over the result.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from kma import coord2, coord2_communities, relevance

log = logging.getLogger("kma.explorer")

SCHEMA_VERSION = 1
CAVEAT_ID = "caveat-v1"
METHOD_URL = "/methodology"
CORRECTIONS_URL = "/corrections"

ACCOUNT_SHARDS = 64
HASHTAG_SHARDS = 16
TOP_TAGS = 1000
TOP_DOMAINS = 300
TOP_URLS = 500
NEIGHBOURS = 25
AMPLIFIED = 20
AMPLIFIES = 10
EVIDENCE_ITEMS = 5
RECENT_POSTS = 10
TAG_EXAMPLES = 5
MIN_SHARED_DRAWN = 2
JACCARD = 0.3
FAST_RETWEET_SECONDS = coord2.FAST_RETWEET_SECONDS
TEXT_MAX = 1200

BIT_RETWEET = 1
TRACE_OF_BIT = {1: "co_retweet", 2: "co_url", 4: "hashtag_sequence", 8: "fast_retweet",
                16: "text_similarity"}

EDGE_RULE = (
    "Edges drawn where two accounts retweeted at least 2 of the same posts, or are "
    "linked by any other trace. Single-shared-retweet links are opt-in."
)
EDGE_RULE_THIN = (
    "Links where two accounts retweeted exactly one post in common and nothing else "
    "linked them. Opt-in: most links in the network are of this kind."
)
EDGE_RULE_ALL = EDGE_RULE + " Between kept communities and within them."


def account_shard(user_id: str) -> int:
    return int(hashlib.sha1(user_id.encode()).hexdigest()[:8], 16) % ACCOUNT_SHARDS


def hashtag_shard(tag: str) -> int:
    return int(hashlib.sha1(tag.lower().encode()).hexdigest()[0], 16)


def lineage_id(member_ids: Iterable[str]) -> str:
    return hashlib.sha1("\n".join(sorted(map(str, member_ids))).encode()).hexdigest()[:10]


def iso(value: Any) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def num(value: Any, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def day(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def count(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


@dataclass(frozen=True)
class Previous:
    """The run before this one, as persisted: enough to match communities."""

    run_id: str
    communities: pd.DataFrame
    members: pd.DataFrame
    lineage: dict[int, str] | None = None


@dataclass
class Daily:
    run: dict
    communities: pd.DataFrame
    members: pd.DataFrame
    edges: pd.DataFrame
    traces: pd.DataFrame
    runs: pd.DataFrame


@dataclass
class Bundle:
    files: dict[str, Any] = field(default_factory=dict)
    unavailable: list[dict] = field(default_factory=list)
    members: set[str] = field(default_factory=set)
    extra_authors: set[str] = field(default_factory=set)
    stats: dict = field(default_factory=dict)


class Corpus:
    """The window's distinct posts in a DuckDB temp table, and the queries over it.

    `members` is the id, community, score table of kept-community members the
    per-account queries join to. Build it with `set_members` once.
    """

    COLUMNS = (
        "post_id", "author_id", "author_handle", "created_at", "text", "url",
        "in_reply_to_id", "is_repost", "repost_of_id", "like_count", "reply_count",
        "repost_count", "hashtags", "urls", "collected_at",
    )

    def __init__(self, con: duckdb.DuckDBPyConnection, start: date, end: date):
        self.con = con
        self.start = start
        self.end = end
        self.until = end + timedelta(days=1)
        self._tags_ready = False
        self._urls_ready = False

    def _in_window(self, col: str = "created_at") -> str:
        return (f"{col} >= TIMESTAMPTZ '{self.start.isoformat()}' "
                f"AND {col} < TIMESTAMPTZ '{self.until.isoformat()}'")

    def set_members(self, members: pd.DataFrame) -> None:
        frame = members[["user_id", "community"]].rename(columns={"user_id": "id", "community": "cid"})
        self.con.register("_mem_df", frame.astype({"id": str, "cid": "int64"}))
        self.con.execute("CREATE OR REPLACE TEMP TABLE _mem AS SELECT * FROM _mem_df")
        self.con.unregister("_mem_df")

    def totals(self) -> dict[str, int]:
        row = self.con.execute(
            "SELECT count(*), count(DISTINCT author_id) FROM _win").fetchone()
        return {"distinct_posts": int(row[0]), "distinct_authors": int(row[1])}

    def per_author(self) -> pd.DataFrame:
        """Posts and reposts per member over the whole window corpus."""
        return self.con.execute(
            """SELECT w.author_id AS id, count(*) AS posts,
                      count(*) FILTER (WHERE coalesce(w.is_repost, false)) AS reposts,
                      max(w.author_handle) AS handle
               FROM _win w JOIN _mem m ON m.id = w.author_id GROUP BY 1"""
        ).df()

    def daily(self) -> pd.DataFrame:
        """Members' posts by author, UTC created_at day inside the window, and kind."""
        got = self.con.execute(
            f"""SELECT w.author_id AS id, CAST(w.created_at AT TIME ZONE 'UTC' AS DATE) AS dt,
                       CASE WHEN coalesce(w.is_repost, false) THEN 'reposts'
                            WHEN w.in_reply_to_id IS NOT NULL THEN 'replies'
                            ELSE 'originals' END AS kind,
                       count(*) AS n
                FROM _win w JOIN _mem m ON m.id = w.author_id
                WHERE {self._in_window('w.created_at')} GROUP BY 1, 2, 3"""
        ).df()
        got["dt"] = got["dt"].map(day)
        return got

    def _ensure_tags(self) -> None:
        if self._tags_ready:
            return
        self.con.execute(
            f"""CREATE OR REPLACE TEMP TABLE _tag AS
                SELECT DISTINCT post_id, author_id,
                       CAST(created_at AT TIME ZONE 'UTC' AS DATE) AS dt, tag
                FROM (SELECT post_id, author_id, created_at,
                             lower(trim(ltrim(unnest(hashtags), '#'))) AS tag
                      FROM _win WHERE hashtags IS NOT NULL AND len(hashtags) > 0
                        AND {self._in_window()})
                WHERE tag <> '' AND length(tag) <= 100"""
        )
        self._tags_ready = True

    def _ensure_urls(self) -> None:
        if self._urls_ready:
            return
        self.con.execute(
            f"""CREATE OR REPLACE TEMP TABLE _url AS
                SELECT DISTINCT post_id, author_id, url,
                       lower(regexp_extract(url, '^[a-zA-Z]+://(?:www\\.)?([^/:?#]+)', 1)) AS domain
                FROM (SELECT post_id, author_id, unnest(urls) AS url
                      FROM _win WHERE urls IS NOT NULL AND len(urls) > 0
                        AND {self._in_window()})
                WHERE url IS NOT NULL AND length(url) <= 500"""
        )
        self._urls_ready = True

    def tag_counts(self) -> pd.DataFrame:
        self._ensure_tags()
        return self.con.execute(
            """SELECT tag, count(DISTINCT post_id) AS posts, count(DISTINCT author_id) AS accounts,
                      min(dt) AS first_seen
               FROM _tag GROUP BY 1 ORDER BY posts DESC, tag LIMIT ?""", [TOP_TAGS]
        ).df()

    def tag_daily(self, tags: list[str]) -> pd.DataFrame:
        self._ensure_tags()
        self.con.register("_top_tags", pd.DataFrame({"tag": tags}))
        try:
            return self.con.execute(
                """SELECT t.tag, t.dt, count(DISTINCT t.post_id) AS posts,
                          count(DISTINCT t.author_id) AS accounts
                   FROM _tag t JOIN _top_tags s ON s.tag = t.tag GROUP BY 1, 2"""
            ).df()
        finally:
            self.con.unregister("_top_tags")

    def tag_communities(self, tags: list[str]) -> pd.DataFrame:
        self._ensure_tags()
        self.con.register("_top_tags", pd.DataFrame({"tag": tags}))
        try:
            return self.con.execute(
                """SELECT t.tag, m.cid, count(DISTINCT t.post_id) AS posts
                   FROM _tag t JOIN _top_tags s ON s.tag = t.tag
                   LEFT JOIN _mem m ON m.id = t.author_id GROUP BY 1, 2"""
            ).df()
        finally:
            self.con.unregister("_top_tags")

    def tag_examples(self, tags: list[str]) -> pd.DataFrame:
        """Latest original posts by kept members using the tag, post ids only."""
        self._ensure_tags()
        self.con.register("_top_tags", pd.DataFrame({"tag": tags}))
        try:
            return self.con.execute(
                f"""SELECT tag, post_id FROM (
                      SELECT t.tag, t.post_id,
                             row_number() OVER (PARTITION BY t.tag ORDER BY w.created_at DESC,
                                                t.post_id) AS rk
                      FROM _tag t JOIN _top_tags s ON s.tag = t.tag
                      JOIN _win w ON w.post_id = t.post_id
                      JOIN _mem m ON m.id = w.author_id
                      WHERE NOT coalesce(w.is_repost, false)) WHERE rk <= {TAG_EXAMPLES}"""
            ).df()
        finally:
            self.con.unregister("_top_tags")

    def member_tags(self) -> pd.DataFrame:
        self._ensure_tags()
        return self.con.execute(
            """SELECT t.author_id AS id, m.cid, t.tag, count(DISTINCT t.post_id) AS posts
               FROM _tag t JOIN _mem m ON m.id = t.author_id GROUP BY 1, 2, 3"""
        ).df()

    def member_domains(self) -> pd.DataFrame:
        self._ensure_urls()
        return self.con.execute(
            """SELECT u.author_id AS id, m.cid, u.domain, count(DISTINCT u.post_id) AS posts
               FROM _url u JOIN _mem m ON m.id = u.author_id
               WHERE u.domain <> '' GROUP BY 1, 2, 3"""
        ).df()

    def domain_counts(self) -> pd.DataFrame:
        self._ensure_urls()
        return self.con.execute(
            """SELECT domain, count(DISTINCT post_id) AS posts, count(DISTINCT author_id) AS accounts
               FROM _url WHERE domain <> '' GROUP BY 1 ORDER BY posts DESC, domain LIMIT ?""",
            [TOP_DOMAINS],
        ).df()

    def url_counts(self) -> pd.DataFrame:
        self._ensure_urls()
        return self.con.execute(
            """SELECT url, count(DISTINCT post_id) AS posts, count(DISTINCT author_id) AS accounts
               FROM _url GROUP BY 1 ORDER BY posts DESC, url LIMIT ?""", [TOP_URLS]
        ).df()

    def url_communities(self, urls: list[str]) -> pd.DataFrame:
        self._ensure_urls()
        self.con.register("_top_urls", pd.DataFrame({"url": urls}))
        try:
            return self.con.execute(
                """SELECT u.url, m.cid, count(DISTINCT u.post_id) AS posts
                   FROM _url u JOIN _top_urls s ON s.url = u.url
                   JOIN _mem m ON m.id = u.author_id GROUP BY 1, 2"""
            ).df()
        finally:
            self.con.unregister("_top_urls")

    def amplified(self) -> pd.DataFrame:
        """Members' reposts whose original is in the corpus: reposter, original author, count."""
        return self.con.execute(
            """SELECT r.author_id AS id, o.author_id AS orig_id, max(o.author_handle) AS orig_handle,
                      count(*) AS reposts
               FROM _win r JOIN _mem m ON m.id = r.author_id
               JOIN _win o ON o.post_id = r.repost_of_id
               WHERE coalesce(r.is_repost, false) AND o.author_id <> r.author_id
               GROUP BY 1, 2"""
        ).df()

    def reposts_known(self) -> pd.DataFrame:
        """Per member: reposts, and those whose original author is known."""
        return self.con.execute(
            """SELECT r.author_id AS id, count(*) AS reposts, count(o.post_id) AS known
               FROM _win r JOIN _mem m ON m.id = r.author_id
               LEFT JOIN _win o ON o.post_id = r.repost_of_id
               WHERE coalesce(r.is_repost, false) GROUP BY 1"""
        ).df()

    def recent(self) -> pd.DataFrame:
        """Each member's latest non-repost posts created inside the window."""
        return self.con.execute(
            f"""SELECT * FROM (
                  SELECT w.*, row_number() OVER (PARTITION BY w.author_id
                                                 ORDER BY w.created_at DESC, w.post_id) AS rk
                  FROM _win w JOIN _mem m ON m.id = w.author_id
                  WHERE NOT coalesce(w.is_repost, false) AND {self._in_window('w.created_at')})
                WHERE rk <= {RECENT_POSTS}"""
        ).df()

    def posts(self, ids: Iterable[str]) -> pd.DataFrame:
        need = pd.DataFrame({"id": sorted({str(i) for i in ids})})
        self.con.register("_need", need)
        try:
            return self.con.execute(
                f"SELECT w.* FROM _win w JOIN _need n ON n.id = w.post_id"
            ).df()
        finally:
            self.con.unregister("_need")

    def fast_retweets(self) -> pd.DataFrame:
        """Member retweets that landed within the fast window of an original in the corpus."""
        return self.con.execute(
            f"""SELECT r.author_id AS id, r.repost_of_id AS post_id, o.author_id AS entity,
                       date_diff('second', o.created_at, r.created_at) AS gap
                FROM _win r JOIN _mem m ON m.id = r.author_id
                JOIN _win o ON o.post_id = r.repost_of_id
                WHERE r.repost_of_id IS NOT NULL AND o.author_id IS NOT NULL
                  AND date_diff('second', o.created_at, r.created_at)
                      BETWEEN 0 AND {FAST_RETWEET_SECONDS}"""
        ).df()


def post_ref(row: pd.Series) -> dict:
    text = row.get("text")
    text = "" if text is None or (not isinstance(text, str) and pd.isna(text)) else text
    url = row.get("url")
    if url is None or (not isinstance(url, str) and pd.isna(url)):
        url = f"https://x.com/i/web/status/{row['post_id']}"
    handle = row.get("author_handle")
    return {
        "id": str(row["post_id"]),
        "author_id": str(row["author_id"]),
        "author_handle": "" if handle is None or pd.isna(handle) else str(handle),
        "created_at": iso(row["created_at"]),
        "text": text[:TEXT_MAX],
        "url": str(url),
        "like_count": count(row.get("like_count")),
        "repost_count": count(row.get("repost_count")),
        "reply_count": count(row.get("reply_count")),
        "counts_as_of": iso(row.get("collected_at")),
    }


def params_of(run: dict) -> dict:
    def pick(key: str, default: Any) -> Any:
        value = run.get(key)
        return default if value is None or pd.isna(value) else value

    return {
        "text_threshold": float(pick("text_threshold", 0.0)),
        "text_overlap": float(pick("text_overlap", 0.0)),
        "min_entities": int(pick("min_entities", coord2.MIN_ENTITIES_PER_USER)),
        "min_kenya": float(pick("min_kenya", coord2_communities.MIN_KENYA)),
        "relevance_model": relevance.MODEL,
        "leiden_resolution": coord2_communities.RESOLUTION,
        "leiden_seed": coord2_communities.SEED,
        "min_community_size": coord2_communities.MIN_SIZE,
        "listing_budget": coord2_communities.BUDGET,
        "listing_per_group": coord2_communities.PER_GROUP,
        "edge_percentile": {k: v for k, v in coord2.EDGE_PERCENTILE.items()},
    }


def make_stamp(run: dict, generated_at: datetime) -> dict:
    end = date.fromisoformat(str(run["window_end"])[:10])
    days = int(run["window_days"])
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run["run_id"]),
        "window_id": str(run["window_id"]),
        "window_start": (end - timedelta(days=days - 1)).isoformat(),
        "window_end": end.isoformat(),
        "window_days": days,
        "computed_at": iso(run["computed_at"]),
        "generated_at": iso(generated_at),
        "params": params_of(run),
    }


def layout(n: int, pairs: np.ndarray, seed: int = 0, algorithm: str = "drl") -> np.ndarray:
    """Seeded 2-D positions scaled into [-1, 1], one row per node."""
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.zeros((1, 2))
    import igraph as ig

    graph = ig.Graph(n=n, edges=[(int(a), int(b)) for a, b in pairs])
    random.seed(seed)
    np.random.seed(seed)
    if n < 4 or len(pairs) == 0:
        coords = np.array(graph.layout_circle().coords)
    elif algorithm == "drl":
        coords = np.array(graph.layout_drl().coords)
    else:
        coords = np.array(graph.layout_fruchterman_reingold().coords)
    coords = coords - coords.mean(axis=0)
    scale = float(np.abs(coords).max())
    return coords / scale if scale > 0 else coords


def community_label(cid: int) -> str:
    return f"Community {cid + 1:02d}"


def _ids(series: pd.Series) -> list[str]:
    return [str(v) for v in series]


def build_rows(daily: Daily, members: pd.DataFrame, corpus: Corpus, lineages: dict[int, str]) -> dict[int, dict]:
    """The CommunityRow of every community, kept or dropped."""
    tags = corpus.member_tags()
    domains = corpus.member_domains()
    top_tags: dict[int, list[dict]] = {}
    top_domains: dict[int, list[dict]] = {}
    if not tags.empty:
        grouped = tags.groupby(["cid", "tag"], as_index=False).agg(
            posts=("posts", "sum"), accounts=("id", "nunique"))
        for cid, part in grouped.groupby("cid"):
            part = part.sort_values(["posts", "tag"], ascending=[False, True]).head(10)
            top_tags[int(cid)] = [{"tag": r.tag, "posts": int(r.posts), "accounts": int(r.accounts)}
                                  for r in part.itertuples()]
    if not domains.empty:
        grouped = domains.groupby(["cid", "domain"], as_index=False).agg(
            posts=("posts", "sum"), accounts=("id", "nunique"))
        for cid, part in grouped.groupby("cid"):
            part = part.sort_values(["posts", "domain"], ascending=[False, True]).head(5)
            top_domains[int(cid)] = [{"domain": r.domain, "posts": int(r.posts), "accounts": int(r.accounts)}
                                     for r in part.itertuples()]

    sizes = members.groupby("community").size().to_dict()
    share_cols = [c for c in daily.communities.columns if c.startswith("share_")]
    rows: dict[int, dict] = {}
    for rec in daily.communities.to_dict("records"):
        cid = int(rec["community"])
        kept = bool(rec["kept"])
        share = {c[len("share_"):]: num(rec[c]) for c in share_cols
                 if rec.get(c) is not None and not pd.isna(rec[c])}
        row = {
            "cid": cid,
            "lineage": lineages[cid],
            "label": community_label(cid),
            "size": int(sizes.get(cid, 0)) if kept else int(rec["size"]),
            "edges": int(rec["edges"]),
            "eigenvalue": num(rec["eigenvalue"]),
            "trace_share": share,
            "kenya_model": num(rec.get("kenya_model")),
            "relevance_coverage": num(rec.get("relevance_coverage")),
            "relevance_scored": bool(rec["relevance_scored"]),
            "kept": kept,
            "listed": int(rec.get("listed", 0) or 0),
        }
        if kept:
            row["top_hashtags"] = top_tags.get(cid, [])
            row["top_domains"] = top_domains.get(cid, [])
        rows[cid] = row
    return rows


def edge_tables(edges: pd.DataFrame, cid_of: dict[str, int]) -> pd.DataFrame:
    """Fused edges between kept members, with community ids and the drawn flag."""
    cs = edges["source"].map(cid_of)
    ct = edges["target"].map(cid_of)
    both = cs.notna() & ct.notna()
    kept = edges[both].copy()
    kept["cs"] = cs[both].astype("int64")
    kept["ct"] = ct[both].astype("int64")
    if kept["shared_retweets"].isna().any() and (kept["mask"] & BIT_RETWEET).astype(bool)[
            kept["shared_retweets"].isna()].any():
        raise ValueError("co_retweet edges without shared_retweets: the daily run has no trace rows")
    kept["shared"] = kept["shared_retweets"].fillna(0).astype("int64")
    other = (kept["mask"] & ~BIT_RETWEET) != 0
    kept["drawn"] = other | (kept["shared"] >= MIN_SHARED_DRAWN)
    kept["intra"] = kept["cs"] == kept["ct"]
    return kept.reset_index(drop=True)


def _graph(stamp: dict, scope: str, cid: int | None, rule: str, layout_note: str,
           ids: list[str], handles: list[str], cids: list[int], xy: np.ndarray,
           edges: pd.DataFrame) -> dict:
    index = {i: k for k, i in enumerate(ids)}
    s = [index[i] for i in edges["source"]]
    t = [index[i] for i in edges["target"]]
    degree = np.bincount(np.array(s + t, dtype=np.int64), minlength=len(ids)) if s else np.zeros(len(ids), int)
    weights = [0.0 if pd.isna(w) else round(float(w), 4) for w in edges["weight_max"]]
    doc = {
        "kind": "graph", "stamp": stamp, "scope": scope,
        "edge_rule": rule, "layout": layout_note,
        "nodes": {
            "id": ids, "handle": handles, "cid": [int(c) for c in cids],
            "x": [round(float(v), 4) for v in xy[:, 0]],
            "y": [round(float(v), 4) for v in xy[:, 1]],
            "degree": [int(d) for d in degree],
        },
        "edges": {
            "s": s, "t": t, "w": weights,
            "mask": [int(m) for m in edges["mask"]],
            "shared": [int(v) for v in edges["shared"]],
        },
    }
    if cid is not None:
        doc["cid"] = int(cid)
    return doc


def build_graphs(stamp: dict, members: pd.DataFrame, kept_edges: pd.DataFrame,
                 handles: dict[str, str], rows: dict[int, dict]) -> tuple[dict[str, dict], dict]:
    """graph/meta, graph/all, and per community graph and thin graph.

    Returns the files and `index`: per community, the drawn and thin edge frames in file
    order, which evidence keys point into."""
    files: dict[str, dict] = {}
    index: dict[int, dict] = {}
    cid_of = dict(zip(members["user_id"], members["community"].astype(int)))

    drawn = kept_edges[kept_edges["drawn"]]
    ids = sorted({*drawn["source"], *drawn["target"]}, key=lambda i: (cid_of[i], i))
    pos = {i: k for k, i in enumerate(ids)}
    pairs = np.array([[pos[a], pos[b]] for a, b in zip(drawn["source"], drawn["target"])],
                     dtype=np.int64).reshape(-1, 2)
    xy = layout(len(ids), pairs, seed=0)
    files["graph/all.json"] = _graph(
        stamp, "all", None, EDGE_RULE_ALL, "igraph drl seed 0", ids,
        [handles[i] for i in ids], [cid_of[i] for i in ids], xy, drawn)

    cross = kept_edges[~kept_edges["intra"]]
    pair_counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for a, b, sh in zip(cross["cs"], cross["ct"], cross["shared"]):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        pair_counts[key][0] += 1
        pair_counts[key][1] += int(sh >= MIN_SHARED_DRAWN)
    kept_cids = sorted(c for c, r in rows.items() if r["kept"])
    meta_index = {c: k for k, c in enumerate(kept_cids)}
    meta_pairs = np.array([[meta_index[a], meta_index[b]] for (a, b) in pair_counts],
                          dtype=np.int64).reshape(-1, 2)
    meta_xy = layout(len(kept_cids), meta_pairs, seed=0, algorithm="fr")
    files["graph/meta.json"] = {
        "kind": "graph_meta", "stamp": stamp,
        "nodes": [{"cid": c, "size": rows[c]["size"], "x": round(float(meta_xy[k, 0]), 4),
                   "y": round(float(meta_xy[k, 1]), 4)} for k, c in enumerate(kept_cids)],
        "edges": [{"a": a, "b": b, "edges": n, "edges_shared_ge_2": g}
                  for (a, b), (n, g) in sorted(pair_counts.items())],
    }

    intra = kept_edges[kept_edges["intra"]]
    for cid in kept_cids:
        part = members[members["community"] == cid].sort_values(
            ["within_score", "user_id"], ascending=[False, True])
        cids_ids = _ids(part["user_id"])
        local = intra[intra["cs"] == cid]
        p = {i: k for k, i in enumerate(cids_ids)}
        all_pairs = np.array([[p[a], p[b]] for a, b in zip(local["source"], local["target"])],
                             dtype=np.int64).reshape(-1, 2)
        cxy = layout(len(cids_ids), all_pairs, seed=0)
        main = local[local["drawn"]]
        thin = local[~local["drawn"]]
        hs = [handles[i] for i in cids_ids]
        files[f"graph/c{cid}.json"] = _graph(
            stamp, "community", cid, EDGE_RULE, "igraph drl seed 0", cids_ids, hs,
            [cid] * len(cids_ids), cxy, main)
        files[f"graph/c{cid}.thin.json"] = _graph(
            stamp, "community_thin", cid, EDGE_RULE_THIN, "igraph drl seed 0", cids_ids, hs,
            [cid] * len(cids_ids), cxy, thin)
        index[cid] = {"drawn": main.reset_index(drop=True), "thin": thin.reset_index(drop=True)}
    return files, index


def entity_sets(traces: pd.DataFrame) -> dict[str, dict[str, set]]:
    out: dict[str, dict[str, set]] = {}
    if traces.empty:
        return out
    for name, part in traces.groupby("trace"):
        sets: dict[str, set] = defaultdict(set)
        for user, entity in zip(part["user_id"], part["entity"]):
            sets[user].add(entity)
        out[str(name)] = sets
    return out


TextEvidence = Callable[[list[tuple[str, str]]], dict[tuple[str, str], tuple[int, list[dict]]]]


def build_evidence(
    stamp: dict, index: dict[int, dict], traces: pd.DataFrame, corpus: Corpus,
    text_evidence: TextEvidence | None, hidden: set[str] | frozenset[str] = frozenset(),
) -> tuple[dict[str, dict], set[str], dict]:
    """Per community evidence file. Returns the files, the non-member authors
    whose posts it quotes, and counters. `hidden` is post ids evidence must not
    name (posts by denylisted accounts)."""
    sets = entity_sets(traces)
    fast = corpus.fast_retweets()
    fast_by_user: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for r in fast.itertuples():
        fast_by_user[r.id][r.entity][r.post_id] = int(r.gap)

    text_pairs = [(a, b) for part in index.values() for frame in (part["drawn"], part["thin"])
                  for a, b, m in zip(frame["source"], frame["target"], frame["mask"]) if m & 16]
    text = text_evidence(text_pairs) if text_evidence and text_pairs else {}

    def items_for(a: str, b: str, mask: int) -> tuple[int, list[dict]]:
        total = 0
        items: list[dict] = []
        if mask & 1:
            common = sorted((sets.get("co_retweet", {}).get(a, set())
                             & sets.get("co_retweet", {}).get(b, set())) - hidden)
            total += len(common)
            items += [{"trace": "co_retweet", "post": p} for p in common[:EVIDENCE_ITEMS]]
        if mask & 2:
            common = sorted(sets.get("co_url", {}).get(a, set()) & sets.get("co_url", {}).get(b, set()))
            total += len(common)
            items += [{"trace": "co_url", "url": u} for u in common[:EVIDENCE_ITEMS]]
        if mask & 4:
            common = sorted(sets.get("hashtag_sequence", {}).get(a, set())
                            & sets.get("hashtag_sequence", {}).get(b, set()))
            total += len(common)
            items += [{"trace": "hashtag_sequence", "tags": s.split("|")} for s in common[:EVIDENCE_ITEMS]]
        if mask & 8:
            both = set(fast_by_user.get(a, {})) & set(fast_by_user.get(b, {}))
            found: list[tuple[str, int]] = []
            for author in sorted(both):
                pa_, pb_ = fast_by_user[a][author], fast_by_user[b][author]
                shared = sorted((set(pa_) & set(pb_)) - hidden)
                found += [(p, max(pa_[p], pb_[p])) for p in shared]
                if not shared:
                    found += [(p, g) for p, g in sorted(pa_.items()) if p not in hidden][:1]
            total += len(found)
            items += [{"trace": "fast_retweet", "post": p, "gap_seconds": g}
                      for p, g in found[:EVIDENCE_ITEMS]]
        if mask & 16:
            n, got = text.get((a, b), (0, []))
            total += n
            items += got[:EVIDENCE_ITEMS]
        return total, items

    per_cid: dict[int, dict[str, dict]] = {}
    needed: set[str] = set()
    for cid, part in index.items():
        evid: dict[str, dict] = {}
        for prefix, frame in (("e", part["drawn"]), ("t", part["thin"])):
            for i, (a, b, m) in enumerate(zip(frame["source"], frame["target"], frame["mask"])):
                total, items = items_for(a, b, int(m))
                if not items and total == 0:
                    continue
                evid[f"{prefix}{i}"] = {"total": int(total), "items": items}
                for it in items:
                    for key in ("post", "post_a", "post_b"):
                        if key in it:
                            needed.add(it[key])
        per_cid[cid] = evid

    posts = corpus.posts(needed)
    refs = {str(r["post_id"]): post_ref(r) for _, r in posts.iterrows()}
    files: dict[str, dict] = {}
    extra: set[str] = set()
    for cid, evid in per_cid.items():
        ids = {it[k] for e in evid.values() for it in e["items"] for k in ("post", "post_a", "post_b") if k in it}
        mine = {i: refs[i] for i in sorted(ids) if i in refs}
        extra |= {r["author_id"] for r in mine.values()}
        files[f"evidence/c{cid}.json"] = {
            "kind": "evidence", "stamp": stamp, "cid": cid, "edges": evid, "posts": mine}
    return files, extra, {"evidence_posts": len(refs), "evidence_edges": sum(len(v) for v in per_cid.values())}


def build_accounts(
    stamp: dict, members: pd.DataFrame, kept_edges: pd.DataFrame, all_edges: pd.DataFrame,
    handles: dict[str, str], profiles: pd.DataFrame, kenya: pd.DataFrame, corpus: Corpus,
    lineages: dict[int, str],
) -> tuple[dict[str, dict], set[str]]:
    """Account shards, and the non-member authors the cards name as amplified."""
    cid_of = dict(zip(members["user_id"], members["community"].astype(int)))
    per = corpus.per_author().set_index("id")
    daily = corpus.daily()
    tags = corpus.member_tags()
    amplified = corpus.amplified()
    recent = corpus.recent()

    prof = profiles.drop_duplicates("id").set_index("id") if not profiles.empty else profiles
    kenya_idx = kenya if kenya is not None and not kenya.empty else None

    by_daily: dict[str, list[dict]] = defaultdict(list)
    for r in daily.groupby(["id", "dt"], as_index=False)["n"].sum().itertuples():
        by_daily[r.id].append({"dt": r.dt, "n": int(r.n)})
    for v in by_daily.values():
        v.sort(key=lambda d: d["dt"])
    by_tags: dict[str, list[dict]] = defaultdict(list)
    if not tags.empty:
        top = tags.sort_values(["id", "posts", "tag"], ascending=[True, False, True])
        for r in top.groupby("id").head(10).itertuples():
            by_tags[r.id].append({"tag": r.tag, "posts": int(r.posts), "accounts": 1})
    by_amp: dict[str, list[dict]] = defaultdict(list)
    extra: set[str] = set()
    if not amplified.empty:
        top = amplified.sort_values(["id", "reposts", "orig_id"], ascending=[True, False, True])
        for r in top.groupby("id").head(AMPLIFIES).itertuples():
            by_amp[r.id].append({"id": str(r.orig_id), "handle": "" if pd.isna(r.orig_handle) else str(r.orig_handle),
                                 "reposts": int(r.reposts)})
            extra.add(str(r.orig_id))
    by_recent: dict[str, list[dict]] = defaultdict(list)
    for _, r in recent.sort_values(["author_id", "rk"]).iterrows():
        by_recent[r["author_id"]].append(post_ref(r))

    both = pd.concat([
        all_edges[["source", "target", "mask", "weight_max", "shared_retweets"]],
        all_edges.rename(columns={"source": "target", "target": "source"})[
            ["source", "target", "mask", "weight_max", "shared_retweets"]],
    ], ignore_index=True)
    totals = both.groupby("source").size().to_dict()

    ke = kept_edges
    nb = pd.concat([
        pd.DataFrame({"id": ke["source"], "other": ke["target"], "mask": ke["mask"], "w": ke["weight_max"],
                      "shared": ke["shared"], "drawn": ke["drawn"], "ocid": ke["ct"]}),
        pd.DataFrame({"id": ke["target"], "other": ke["source"], "mask": ke["mask"], "w": ke["weight_max"],
                      "shared": ke["shared"], "drawn": ke["drawn"], "ocid": ke["cs"]}),
    ], ignore_index=True)
    nb = nb[nb["drawn"]].sort_values(["id", "shared", "w", "other"], ascending=[True, False, False, True])
    by_nb: dict[str, list[dict]] = defaultdict(list)
    for r in nb.groupby("id").head(NEIGHBOURS).itertuples():
        by_nb[r.id].append({"id": r.other, "handle": handles[r.other], "cid": int(r.ocid),
                            "mask": int(r.mask), "shared": int(r.shared),
                            "w": 0.0 if pd.isna(r.w) else round(float(r.w), 4)})

    shards: dict[int, dict[str, dict]] = defaultdict(dict)
    for rec in members.itertuples():
        uid = str(rec.user_id)
        cid = int(rec.community)
        p = prof.loc[uid] if uid in prof.index else None
        profile = {
            "display_name": None, "bio": None, "followers": None, "following": None,
            "statuses": None, "created_at": None, "profile_as_of": None,
        }
        if p is not None:
            profile = {
                "display_name": None if pd.isna(p["display_name"]) else str(p["display_name"]),
                "bio": None if pd.isna(p["bio"]) else str(p["bio"]),
                "followers": count(p["followers"]), "following": count(p["following"]),
                "statuses": count(p["statuses"]), "created_at": iso(p["created_at"]),
                "profile_as_of": iso(p["profile_as_of"]),
            }
        k = kenya_idx.loc[uid] if kenya_idx is not None and uid in kenya_idx.index else None
        card = {
            "id": uid, "handle": handles[uid], "cid": cid, "lineage": lineages[cid],
            "within_score": num(rec.within_score),
            "centrality": num(rec.centrality),
            "kenya_model": None if k is None else num(k["kenya_model"]),
            "posts_scored": 0 if k is None else int(k["scored"]),
            "posts_in_window": int(per["posts"].get(uid, 0)),
            "reposts_in_window": int(per["reposts"].get(uid, 0)),
            "profile": profile,
            "neighbours": by_nb.get(uid, []),
            "neighbours_total": int(totals.get(uid, 0)),
            "hashtags": by_tags.get(uid, []),
            "amplifies": by_amp.get(uid, []),
            "daily_posts": by_daily.get(uid, []),
            "recent_posts": by_recent.get(uid, []),
        }
        shards[account_shard(uid)][uid] = card
    files = {f"accounts/{s}.json": {"kind": "account_shard", "stamp": stamp, "shard": s,
                                     "accounts": dict(sorted(cards.items()))}
             for s, cards in sorted(shards.items())}
    return files, extra


def build_search(stamp: dict, members: pd.DataFrame, handles: dict[str, str],
                 profiles: pd.DataFrame) -> dict:
    names = {}
    if not profiles.empty:
        names = dict(zip(profiles["id"], profiles["display_name"]))
    rows = sorted(((handles[str(u)], str(u), int(c)) for u, c in zip(members["user_id"], members["community"])),
                  key=lambda r: (r[0].lower(), r[1]))
    return {
        "kind": "search_accounts", "stamp": stamp,
        "id": [r[1] for r in rows], "handle": [r[0] for r in rows],
        "display_name": [None if pd.isna(names.get(r[1])) else names.get(r[1]) for r in rows],
        "cid": [r[2] for r in rows],
    }


def build_community_details(
    stamp: dict, members: pd.DataFrame, rows: dict[int, dict], handles: dict[str, str],
    kenya: pd.DataFrame, corpus: Corpus, graph_index: dict[int, dict],
) -> dict[str, dict]:
    per = corpus.per_author().set_index("id")
    daily = corpus.daily()
    amplified = corpus.amplified()
    known = corpus.reposts_known().set_index("id")
    days = [(date.fromisoformat(stamp["window_start"]) + timedelta(days=i)).isoformat()
            for i in range(stamp["window_days"])]
    cid_of = dict(zip(members["user_id"], members["community"].astype(int)))
    daily = daily.assign(cid=daily["id"].map(cid_of))
    amplified = amplified.assign(cid=amplified["id"].map(cid_of))

    files: dict[str, dict] = {}
    for cid, row in rows.items():
        if not row["kept"]:
            continue
        part = members[members["community"] == cid].sort_values(
            ["within_score", "user_id"], ascending=[False, True])
        ids = _ids(part["user_id"])
        kn = [None if kenya is None or i not in kenya.index else num(kenya.loc[i, "kenya_model"])
              for i in ids]
        d = daily[daily["cid"] == cid].groupby(["dt", "kind"])["n"].sum().unstack(fill_value=0)
        series = []
        for key in days:
            r = d.loc[key] if key in d.index else None
            series.append({"dt": key,
                           "originals": int(r.get("originals", 0)) if r is not None else 0,
                           "replies": int(r.get("replies", 0)) if r is not None else 0,
                           "reposts": int(r.get("reposts", 0)) if r is not None else 0})
        a = amplified[amplified["cid"] == cid]
        amp = []
        if not a.empty:
            agg = a.groupby("orig_id").agg(reposts=("reposts", "sum"), members=("id", "nunique"),
                                           handle=("orig_handle", "max")).reset_index()
            agg = agg.sort_values(["reposts", "orig_id"], ascending=[False, True]).head(AMPLIFIED)
            amp = [{"id": str(r.orig_id), "handle": "" if pd.isna(r.handle) else str(r.handle),
                    "reposts": int(r.reposts), "members": int(r.members)} for r in agg.itertuples()]
        k = known.loc[known.index.intersection(ids)]
        total_rt = int(k["reposts"].sum()) if len(k) else 0
        files[f"communities/{cid}.json"] = {
            "kind": "community_detail", "stamp": stamp, "community": row,
            "members": {
                "id": ids, "handle": [handles[i] for i in ids],
                "within_score": [num(v) for v in part["within_score"]],
                "centrality": [num(v) for v in part["centrality"]],
                "posts_in_window": [int(per["posts"].get(i, 0)) for i in ids],
                "kenya_model": kn,
            },
            "daily_posts": series,
            "amplified": amp,
            "amplified_coverage": num(k["known"].sum() / total_rt) if total_rt else None,
            "graph_available": True,
        }
    return files


def build_hashtags(stamp: dict, corpus: Corpus, trend: pd.DataFrame | None
                   ) -> tuple[dict[str, dict], set[str]]:
    counts = corpus.tag_counts()
    tags = list(counts["tag"])
    emerging, emerging_run = [], None
    emerging_tags: set[str] = set()
    if trend is not None and not trend.empty:
        emerging_run = str(trend["run_id"].iloc[0])
        for r in trend.sort_values("emergence", ascending=False).itertuples():
            emerging.append({"tag": str(r.tag).lower(), "recent_rate": float(r.recent_rate),
                             "prior_rate": float(r.prior_rate), "emergence": float(r.emergence),
                             "posts": int(r.posts), "censused": bool(r.censused)})
        emerging_tags = {e["tag"] for e in emerging}
    index = {
        "kind": "hashtag_index", "stamp": stamp,
        "basis": ("Distinct posts in the collected corpus created inside the window, reposts "
                  "included, by lower-cased tag. Counts of rows in a targeted collection, not "
                  "prevalence and not a share of discourse."),
        "tags": [{"tag": r.tag, "posts": int(r.posts), "accounts": int(r.accounts),
                  "shard": hashtag_shard(r.tag), "first_seen": day(r.first_seen),
                  **({"emerging": True} if r.tag in emerging_tags else {})}
                 for r in counts.itertuples()],
        "emerging": emerging,
        "emerging_run": emerging_run,
    }
    files = {"hashtags/index.json": index}
    if not tags:
        return files, set()

    daily = corpus.tag_daily(tags)
    comm = corpus.tag_communities(tags)
    ex = corpus.tag_examples(tags)
    posts = corpus.posts(ex["post_id"]) if not ex.empty else pd.DataFrame()
    refs = {str(r["post_id"]): post_ref(r) for _, r in posts.iterrows()}
    daily_by: dict[str, list] = defaultdict(list)
    for r in daily.sort_values(["tag", "dt"]).itertuples():
        daily_by[r.tag].append({"dt": day(r.dt), "posts": int(r.posts), "accounts": int(r.accounts)})
    comm_by: dict[str, list] = defaultdict(list)
    for r in comm.sort_values(["tag", "posts"], ascending=[True, False]).itertuples():
        comm_by[r.tag].append({"cid": None if pd.isna(r.cid) else int(r.cid), "posts": int(r.posts)})
    ex_by: dict[str, list] = defaultdict(list)
    for r in ex.itertuples():
        if r.post_id in refs:
            ex_by[r.tag].append(refs[r.post_id])
    shards: dict[int, dict] = defaultdict(dict)
    for tag in tags:
        shards[hashtag_shard(tag)][tag] = {
            "tag": tag, "daily": daily_by.get(tag, []), "communities": comm_by.get(tag, []),
            "examples": ex_by.get(tag, []),
        }
    for s in range(HASHTAG_SHARDS):
        if s in shards:
            files[f"hashtags/s{s}.json"] = {"kind": "hashtag_shard", "stamp": stamp, "shard": s,
                                             "tags": shards[s]}
    extra = {r["author_id"] for r in refs.values()}
    return files, extra


def build_links(stamp: dict, corpus: Corpus) -> dict:
    domains = corpus.domain_counts()
    urls = corpus.url_counts()
    comm = corpus.url_communities(list(urls["url"])) if not urls.empty else pd.DataFrame()
    top: dict[str, list[int]] = {}
    if not comm.empty:
        for url, part in comm.groupby("url"):
            part = part.sort_values(["posts", "cid"], ascending=[False, True]).head(3)
            top[url] = [int(c) for c in part["cid"]]
    return {
        "kind": "link_index", "stamp": stamp,
        "domains": [{"domain": r.domain, "posts": int(r.posts), "accounts": int(r.accounts)}
                    for r in domains.itertuples()],
        "urls": [{"url": r.url, "posts": int(r.posts), "accounts": int(r.accounts),
                  **({"top_cids": top[r.url]} if r.url in top else {})} for r in urls.itertuples()],
    }


def build_overview(stamp: dict, daily: Daily, rows: dict[int, dict], members: pd.DataFrame,
                   kept_edges: pd.DataFrame, corpus_totals: dict, post_rows: int | None) -> dict:
    run = daily.run
    kept_rows = [r for r in rows.values() if r["kept"]]
    retweet_intra = kept_edges[kept_edges["intra"] & ((kept_edges["mask"] & BIT_RETWEET) != 0)]
    trace_edges = {t: int(run[f"edges_{t}"]) for t in coord2.TRACES
                   if f"edges_{t}" in run and run[f"edges_{t}"] is not None and not pd.isna(run[f"edges_{t}"])}
    return {
        "kind": "overview", "stamp": stamp, "caveat_id": CAVEAT_ID,
        "surfaced": {
            "accounts_in_graph": int(run["accounts_in_graph"]),
            "edges_in_graph": int(run["edges_in_graph"]),
            "communities": len(rows),
            "communities_kept": len(kept_rows),
            "communities_unscored": int(sum(not r["relevance_scored"] for r in rows.values())),
            "accounts_in_kept_communities": int(len(members)),
        },
        "trace_edges": trace_edges,
        "edge_basis": {
            "co_retweet_edges_in_kept": int(len(retweet_intra)),
            "with_shared_ge_2": int((retweet_intra["shared"] >= 2).sum()),
            "with_shared_ge_3": int((retweet_intra["shared"] >= 3).sum()),
        },
        "window_corpus": {
            "post_rows": int(post_rows or 0),
            "distinct_posts": corpus_totals["distinct_posts"],
            "distinct_authors": corpus_totals["distinct_authors"],
            "embedding_coverage": None,
            "relevance_coverage": None,
        },
    }


def build_pipeline(stamp: dict, runs: pd.DataFrame) -> dict:
    out = []
    for rec in runs.sort_values("computed_at", ascending=False).to_dict("records"):
        stages = {k: round(float(v), 1) for k, v in rec.items()
                  if k.endswith("_s") and v is not None and not pd.isna(v)}
        edges = {k[len("edges_"):]: int(v) for k, v in rec.items()
                 if k.startswith("edges_") and k != "edges_in_graph" and v is not None and not pd.isna(v)}
        out.append({
            "run_id": str(rec["run_id"]), "window_id": str(rec["window_id"]),
            "computed_at": iso(rec["computed_at"]), "window_days": int(rec["window_days"]),
            "accounts_in_graph": int(rec["accounts_in_graph"]), "edges_in_graph": int(rec["edges_in_graph"]),
            "communities": int(rec["communities"]), "communities_kept": int(rec["communities_kept"]),
            "stage_seconds": stages, "edges": edges, "params": params_of(rec),
        })
    return {"kind": "time_pipeline", "stamp": stamp, "runs": out}


def match_communities(cur: dict[int, set[str]], prev: dict[int, set[str]],
                      prev_lineage: dict[int, str], min_jaccard: float = JACCARD
                      ) -> tuple[list[dict], dict[int, str]]:
    """Carry community identity across two runs by membership overlap.

    Returns the change rows and the lineage of every current community."""
    owner_cur = {u: c for c, ms in cur.items() for u in ms}
    inter: Counter = Counter()
    for p, ms in prev.items():
        for u in ms:
            c = owner_cur.get(u)
            if c is not None:
                inter[(c, p)] += 1
    jac = {k: n / (len(cur[k[0]]) + len(prev[k[1]]) - n) for k, n in inter.items()}

    cont_of_cur: dict[int, int] = {}
    cont_of_prev: dict[int, int] = {}
    for (c, p), j in sorted(jac.items(), key=lambda kv: (-kv[1], kv[0])):
        if j < min_jaccard or c in cont_of_cur or p in cont_of_prev:
            continue
        cont_of_cur[c] = p
        cont_of_prev[p] = c

    lineage: dict[int, str] = {}
    matches: list[dict] = []

    def row(status: str, c: int | None, p: int | None, j: float | None) -> dict:
        cm = cur.get(c, set()) if c is not None else set()
        pm = prev.get(p, set()) if p is not None else set()
        return {
            "status": status, "lineage": "", "cid": c, "previous_cid": p,
            "size": len(cm) if c is not None else None,
            "previous_size": len(pm) if p is not None else None,
            "jaccard": None if j is None else round(j, 4),
            "joined": len(cm - pm), "left": len(pm - cm),
        }

    split_parents: set[int] = set()
    for c in sorted(cur):
        if c in cont_of_cur:
            p = cont_of_cur[c]
            lineage[c] = prev_lineage[p]
            r = row("continuing", c, p, jac[(c, p)])
            r["lineage"] = lineage[c]
            matches.append(r)
            continue
        parent = max(((inter[(c, p)], -p) for p in prev if (c, p) in inter), default=None)
        if parent is not None and parent[0] / len(cur[c]) >= min_jaccard:
            p = -parent[1]
            split_parents.add(p)
            lineage[c] = lineage_id(cur[c])
            r = row("split", c, p, jac.get((c, p)))
            r["lineage"] = lineage[c]
            matches.append(r)
        else:
            lineage[c] = lineage_id(cur[c])
            r = row("new", c, None, None)
            r["lineage"] = lineage[c]
            matches.append(r)

    for p in sorted(prev):
        if p in cont_of_prev or p in split_parents:
            continue
        into = max(((inter[(c, p)], -c) for c in cur if (c, p) in inter), default=None)
        if into is not None and into[0] / len(prev[p]) >= min_jaccard:
            c = -into[1]
            r = row("merged", c, p, jac.get((c, p)))
        else:
            r = row("vanished", None, p, None)
        r["lineage"] = prev_lineage[p]
        matches.append(r)
    return matches, lineage


def communities_of(members: pd.DataFrame, communities: pd.DataFrame, deny: set[str]) -> dict[int, set[str]]:
    kept = set(communities.loc[communities["kept"], "community"].astype(int))
    out: dict[int, set[str]] = defaultdict(set)
    for u, c in zip(members["user_id"].astype(str), members["community"].astype(int)):
        if c in kept and u not in deny:
            out[c].add(u)
    return dict(out)


def build_changes(stamp: dict, cur: dict[int, set[str]], previous: Previous | None,
                  deny: set[str], lineage_of_cur: dict[int, str]) -> dict:
    if previous is None:
        return {"kind": "changes", "stamp": stamp, "previous_run": None, "min_jaccard": JACCARD, "matches": []}
    prev = communities_of(previous.members, previous.communities, deny)
    matches, _ = match_communities(cur, prev, previous_lineage(previous, prev))
    for m in matches:
        if m["cid"] is not None:
            m["lineage"] = lineage_of_cur[m["cid"]]
    return {"kind": "changes", "stamp": stamp, "previous_run": previous.run_id,
            "min_jaccard": JACCARD, "matches": matches}


def previous_lineage(previous: Previous | None, prev: dict[int, set[str]]) -> dict[int, str]:
    got = previous.lineage if previous is not None and previous.lineage else {}
    return {c: got.get(c) or lineage_id(ms) for c, ms in prev.items()}


def current_lineages(daily: Daily, members_all: pd.DataFrame, previous: Previous | None,
                     deny: set[str]) -> dict[int, str]:
    """Lineage of every community in this run, kept or dropped."""
    cur_all = {int(c): set(part["user_id"].astype(str)) - deny
               for c, part in members_all.groupby("community")}
    lineage = {c: lineage_id(ms) for c, ms in cur_all.items()}
    if previous is not None:
        kept = communities_of(daily.members, daily.communities, deny)
        prev = communities_of(previous.members, previous.communities, deny)
        _, carried = match_communities(kept, prev, previous_lineage(previous, prev))
        lineage.update(carried)
    return lineage
