"""Parquet slices of the explorer bundle, for the in-browser SQL console.

Five tables under `slices/`, written with DuckDB so the files are plain zstd
parquet that DuckDB-WASM range-reads. They hold what the JSON files already
publish, flattened: members of kept communities, the fused edges among them, their
window posts, hashtag use per day, and the community rows. No per-account flag
columns exist in any of them (the contract validator checks the column names).

`edges.source` and `edges.target` are row positions in `accounts`, which is sorted
by id, so a query joins them with `row_number()` or reads them as plain integers.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from kma.explorer_build import TOP_TAGS, Corpus, community_label

log = logging.getLogger("kma.explorer")

CAP_BYTES = 40 * 1024 * 1024
TAG_ROWS_TOP = 5 * TOP_TAGS
NAMES = ("accounts", "edges", "posts", "hashtags", "communities")


@dataclass(frozen=True)
class Slice:
    data: bytes
    rows: int


class SlicesTooLarge(RuntimeError):
    pass


def _copy(con: duckdb.DuckDBPyConnection, relation: str, path: Path) -> int:
    con.execute(f"COPY ({relation}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    return int(con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0])


def build_slices(
    corpus: Corpus, members: pd.DataFrame, kept_edges: pd.DataFrame, handles: dict[str, str],
    profiles: pd.DataFrame, rows: dict[int, dict], per_author: pd.DataFrame, *, cap: int = CAP_BYTES,
) -> dict[str, Slice]:
    con = corpus.con
    ordered = members.assign(user_id=members["user_id"].astype(str)).sort_values("user_id"
                                                                                 ).reset_index(drop=True)
    names = {}
    if not profiles.empty:
        names = dict(zip(profiles["id"], profiles["display_name"]))
    per = per_author.set_index("id") if "id" in per_author.columns else per_author
    accounts = pd.DataFrame({
        "id": ordered["user_id"],
        "handle": [handles[u] for u in ordered["user_id"]],
        "display_name": [None if pd.isna(names.get(u)) else names.get(u) for u in ordered["user_id"]],
        "cid": ordered["community"].astype("int32"),
        "within_score": ordered["within_score"].astype("float64"),
        "centrality": ordered["centrality"].astype("float64"),
        "posts_in_window": [int(per["posts"].get(u, 0)) for u in ordered["user_id"]],
        "reposts_in_window": [int(per["reposts"].get(u, 0)) for u in ordered["user_id"]],
    })
    position = {u: i for i, u in enumerate(ordered["user_id"])}
    edges = pd.DataFrame({
        "source": kept_edges["source"].map(position).astype("int32"),
        "target": kept_edges["target"].map(position).astype("int32"),
        "mask": kept_edges["mask"].astype("int8"),
        "weight": kept_edges["weight_max"].astype("float64"),
        "shared_retweets": kept_edges["shared"].astype("int32"),
        "drawn": kept_edges["drawn"].astype(bool),
    })
    communities = pd.DataFrame([{
        "cid": cid, "label": community_label(cid), "size": int(r["size"]), "kept": bool(r["kept"]),
        "template": (r.get("template") or {}).get("label") if r["kept"] else None,
    } for cid, r in sorted(rows.items())])

    has_topic = bool(con.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE table_name = '_topic'").fetchone()[0])
    topic_cols = ("t.topic AS topic, t.method AS topic_method" if has_topic
                  else "CAST(NULL AS BIGINT) AS topic, CAST(NULL AS VARCHAR) AS topic_method")
    topic_join = "LEFT JOIN _topic t ON t.post_id = w.post_id" if has_topic else ""
    corpus._ensure_tags()

    out: dict[str, Slice] = {}
    with tempfile.TemporaryDirectory(prefix="kma-slices-") as tmp:
        root = Path(tmp)
        con.register("_sl_accounts", accounts)
        con.register("_sl_edges", edges)
        con.register("_sl_communities", communities)
        try:
            plan = {
                "accounts": "SELECT * FROM _sl_accounts",
                "edges": "SELECT * FROM _sl_edges",
                "communities": "SELECT * FROM _sl_communities ORDER BY cid",
                "hashtags": f"""SELECT tag, CAST(dt AS VARCHAR) AS dt, count(DISTINCT post_id) AS posts,
                                       count(DISTINCT author_id) AS accounts
                                FROM _tag WHERE tag IN (
                                    SELECT tag FROM _tag GROUP BY 1
                                    ORDER BY count(DISTINCT post_id) DESC, tag LIMIT {TAG_ROWS_TOP})
                                GROUP BY 1, 2 ORDER BY 1, 2""",
            }
            for name, relation in plan.items():
                path = root / f"{name}.parquet"
                rows_written = _copy(con, relation, path)
                out[name] = Slice(path.read_bytes(), rows_written)

            def posts(with_reposts: bool) -> Slice:
                where = "" if with_reposts else "WHERE NOT coalesce(w.is_repost, false)"
                path = root / "posts.parquet"
                n = _copy(con, f"""
                    SELECT w.post_id AS id, w.author_id,
                           w.created_at,
                           CASE WHEN coalesce(w.is_repost, false) THEN 'repost'
                                WHEN w.in_reply_to_id IS NOT NULL THEN 'reply'
                                ELSE 'original' END AS kind,
                           w.text, w.like_count, w.repost_count, w.reply_count,
                           coalesce(list_transform(w.hashtags,
                               x -> lower(trim(ltrim(x, '#')))), []) AS hashtags,
                           {topic_cols}
                    FROM _win w JOIN _mem m ON m.id = w.author_id {topic_join} {where}
                    ORDER BY w.author_id, w.created_at, w.post_id""", path)
                return Slice(path.read_bytes(), n)

            out["posts"] = posts(True)
            if sum(len(s.data) for s in out.values()) > cap:
                log.warning("slices over %d bytes with reposts; dropping repost rows from posts", cap)
                out["posts"] = posts(False)
        finally:
            for name in ("_sl_accounts", "_sl_edges", "_sl_communities"):
                con.unregister(name)
    total = sum(len(s.data) for s in out.values())
    if total > cap:
        raise SlicesTooLarge(f"slices are {total} bytes, over the {cap} byte cap")
    return {f"slices/{name}.parquet": out[name] for name in NAMES}
