"""Topic and coverage files of the explorer bundle.

Reads nothing from R2 itself except through the two `load_*` functions; the
builders work on a DuckDB connection that already holds the window's posts in
`_win` and the member table `_mem` (see `explorer_build.Corpus`) and, for topics,
`_topic`, which `load_topic_rows` fills.

Topic ids are only meaningful within one fit, and the `topics/` prefix holds one
fit plus nearest-neighbour rows (`kma.topics_assign`) placed against it. A row of
an older fit, or a nearest-neighbour row made against an older fit, does not
describe the newest fit's topics and is not read. A post that has both a fit row
and a nearest-neighbour row takes the newer, as `semantic.load_topics` does.

Nothing here is per account: a topic is described by its terms, its volume by
day, which kept communities' members posted in it, and example posts by members.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import duckdb
import pandas as pd

from kma import db, semantic
from kma.explorer_build import Corpus, day, iso, post_ref

log = logging.getLogger("kma.explorer")

PLATFORM = "x"
TOP_TOPICS = 300
EXAMPLES = 5
TOPIC_TAGS = 10
TERM_SAMPLE = 1500
TERMS = 8
NAME_WORDS = 3
KNN_PREFIX = "method=knn,"

STAGES = ("embeddings", "relevance", "hatespeech", "incitement", "topics", "labels")


class TopicsUnavailable(RuntimeError):
    """The topics table cannot be read or holds nothing usable for this window."""


def _glob(platform: str = PLATFORM) -> str:
    return f"r2://{db.BUCKET}/topics/platform={platform}/model=*/dt=*/run=*.parquet"


def load_topic_rows(corpus: Corpus, source: str | None = None) -> dict:
    """Fill the temp table `_topic` (post_id, topic, method, assigned_at) with the
    posts created inside the window that carry a topic row of the newest fit, and return what the
    fit is: when, how many posts and topics, how many placed by nearest
    neighbours. `source` is a relation expression for tests."""
    con = corpus.con
    src = source or (
        f"read_parquet('{_glob()}', union_by_name=true, filename=true, hive_partitioning=false)")
    try:
        head = con.execute(
            f"""SELECT max(assigned_at) FILTER (WHERE params NOT LIKE '{KNN_PREFIX}%')
                FROM {src}""").fetchone()
    except duckdb.Error as exc:
        raise TopicsUnavailable(f"topics/ is unreadable: {exc}") from exc
    fitted_at = head[0]
    if fitted_at is None:
        raise TopicsUnavailable("topics/ holds no fit run")
    fit_run = con.execute(
        f"""SELECT min(regexp_extract(filename, 'run=([^/]+)\\.parquet$', 1)) FROM {src}
            WHERE params NOT LIKE '{KNN_PREFIX}%' AND assigned_at = ?""", [fitted_at]).fetchone()[0]

    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE _topic AS
            SELECT platform_post_id AS post_id, topic,
                   CASE WHEN params LIKE '{KNN_PREFIX}%' THEN 'knn' ELSE 'fit' END AS method,
                   assigned_at
            FROM {src}
            WHERE ((params NOT LIKE '{KNN_PREFIX}%' AND assigned_at = ?)
                   OR (params LIKE '{KNN_PREFIX}%' AND params LIKE '%fit_run=' || ? || ',%'))
              AND platform_post_id IN (
                  SELECT post_id FROM _win WHERE {corpus._in_window('created_at')})
            QUALIFY row_number() OVER (PARTITION BY platform_post_id ORDER BY assigned_at DESC) = 1""",
        [fitted_at, fit_run])

    fit = con.execute(
        f"""SELECT count(*), count(DISTINCT topic) FILTER (WHERE topic >= 0),
                   count(*) FILTER (WHERE topic < 0), any_value(params)
            FROM {src} WHERE params NOT LIKE '{KNN_PREFIX}%' AND assigned_at = ?""",
        [fitted_at]).fetchone()
    knn = con.execute(
        f"""SELECT count(*), max(params) FROM {src}
            WHERE params LIKE '{KNN_PREFIX}%' AND params LIKE '%fit_run={fit_run},%'""").fetchone()
    return {
        "fitted_at": iso(fitted_at), "fit_posts": int(fit[0]), "topics": int(fit[1]),
        "noise_share": round(int(fit[2]) / int(fit[0]), 4) if fit[0] else 0.0,
        "params": str(fit[3]), "knn_posts": int(knn[0]), "knn_params": knn[1],
    }


def terms_for(con: duckdb.DuckDBPyConnection, tids: list[int]) -> dict[int, list[str]]:
    """c-TF-IDF terms (`semantic.topic_summary`) of the topics' window posts, at
    most `TERM_SAMPLE` posts per topic, drawn in post-id hash order so a rerun
    reads the same ones."""
    if not tids:
        return {}
    con.register("_tids", pd.DataFrame({"topic": tids}))
    try:
        frame = con.execute(
            f"""SELECT topic, text FROM (
                  SELECT t.topic, w.text,
                         row_number() OVER (PARTITION BY t.topic ORDER BY hash(w.post_id)) AS rk
                  FROM _topic t JOIN _win w ON w.post_id = t.post_id JOIN _tids s ON s.topic = t.topic
                  WHERE w.text IS NOT NULL AND NOT coalesce(w.is_repost, false))
                WHERE rk <= {TERM_SAMPLE}""").df()
    finally:
        con.unregister("_tids")
    if frame.empty:
        return {}
    try:
        summary = semantic.topic_summary(frame, top_terms=TERMS)
    except ValueError:
        log.warning("topics: no term appears in two topics, so none are labelled")
        return {}
    return {int(r.topic): [t for t in str(r.terms).split(", ") if t] for r in summary.itertuples()}


def build_topics(stamp: dict, corpus: Corpus, fit: dict, *, top: int = TOP_TOPICS
                 ) -> dict[str, dict]:
    """`topics/index.json` and `topics/{tid}.json` for the topics with most window
    posts. Raises `TopicsUnavailable` when no window post carries a topic."""
    con = corpus.con
    totals = con.execute(
        """SELECT count(*), count(*) FILTER (WHERE t.topic < 0)
           FROM _topic t JOIN _win w ON w.post_id = t.post_id""").fetchone()
    if not totals[0]:
        raise TopicsUnavailable("no post in this window carries a topic row of the newest fit")
    window_posts, noise_posts = int(totals[0]), int(totals[1])

    per = con.execute(
        f"""SELECT t.topic AS tid, count(*) AS posts,
                   count(*) FILTER (WHERE t.method = 'fit') AS fit_posts,
                   count(*) FILTER (WHERE t.method = 'knn') AS knn_posts,
                   count(DISTINCT w.author_id) AS accounts
            FROM _topic t JOIN _win w ON w.post_id = t.post_id
            WHERE t.topic >= 0
            GROUP BY 1 ORDER BY posts DESC, tid LIMIT {top}""").df()
    tids = [int(t) for t in per["tid"]]
    listed_total = int(con.execute(
        f"""SELECT count(DISTINCT t.topic) FROM _topic t JOIN _win w ON w.post_id = t.post_id
            WHERE t.topic >= 0""").fetchone()[0])
    if not tids:
        raise TopicsUnavailable("no topic has a post created inside this window")

    con.register("_top_tids", pd.DataFrame({"tid": tids}))
    try:
        days = [corpus.start + timedelta(days=i)
                for i in range((corpus.end - corpus.start).days + 1)]
        position = {day(d): i for i, d in enumerate(days)}
        daily_rows = con.execute(
            f"""SELECT t.topic AS tid, CAST(w.created_at AT TIME ZONE 'UTC' AS DATE) AS dt,
                       count(*) AS n
                FROM _topic t JOIN _win w ON w.post_id = t.post_id JOIN _top_tids s ON s.tid = t.topic
                GROUP BY 1, 2""").df()
        comm = con.execute(
            """SELECT t.topic AS tid, m.cid, count(*) AS posts
               FROM _topic t JOIN _win w ON w.post_id = t.post_id
               JOIN _mem m ON m.id = w.author_id JOIN _top_tids s ON s.tid = t.topic
               GROUP BY 1, 2""").df()
        comm_total = con.execute(
            """SELECT m.cid, count(*) AS posts
               FROM _topic t JOIN _win w ON w.post_id = t.post_id JOIN _mem m ON m.id = w.author_id
               GROUP BY 1""").df().set_index("cid")["posts"].to_dict()
        examples = con.execute(
            f"""SELECT tid, post_id FROM (
                  SELECT t.topic AS tid, t.post_id,
                         row_number() OVER (PARTITION BY t.topic
                                            ORDER BY w.created_at DESC, t.post_id) AS rk
                  FROM _topic t JOIN _win w ON w.post_id = t.post_id
                  JOIN _mem m ON m.id = w.author_id JOIN _top_tids s ON s.tid = t.topic
                  WHERE NOT coalesce(w.is_repost, false) AND w.text IS NOT NULL)
                WHERE rk <= {EXAMPLES}""").df()
        corpus._ensure_tags()
        tags = con.execute(
            f"""SELECT tid, tag, posts, accounts FROM (
                  SELECT t.topic AS tid, g.tag, count(DISTINCT g.post_id) AS posts,
                         count(DISTINCT g.author_id) AS accounts,
                         row_number() OVER (PARTITION BY t.topic
                                            ORDER BY count(DISTINCT g.post_id) DESC, g.tag) AS rk
                  FROM _topic t JOIN _tag g ON g.post_id = t.post_id JOIN _top_tids s ON s.tid = t.topic
                  GROUP BY t.topic, g.tag) WHERE rk <= {TOPIC_TAGS}""").df()
    finally:
        con.unregister("_top_tids")

    terms = terms_for(con, tids)
    ex_posts = corpus.posts(examples["post_id"]) if not examples.empty else pd.DataFrame()
    refs = {str(r["post_id"]): post_ref(r) for _, r in ex_posts.iterrows()}

    daily_by: dict[int, list[int]] = {t: [0] * len(days) for t in tids}
    for r in daily_rows.itertuples():
        daily_by[int(r.tid)][position[day(r.dt)]] = int(r.n)

    topics, files = [], {}
    for r in per.itertuples():
        tid = int(r.tid)
        t_terms = terms.get(tid, [])
        entry = {"tid": tid, "terms": t_terms, "posts_in_window": int(r.posts),
                 "fit_posts_in_window": int(r.fit_posts), "knn_posts_in_window": int(r.knn_posts),
                 "accounts_in_window": int(r.accounts), "daily": daily_by[tid]}
        name = " ".join(t_terms[:NAME_WORDS])
        if name:
            entry["name"] = name
        topics.append(entry)
        detail = {
            "kind": "topic_detail", "stamp": stamp, "tid": tid, "terms": t_terms,
            "posts_in_window": int(r.posts), "fit_posts_in_window": int(r.fit_posts),
            "knn_posts_in_window": int(r.knn_posts), "daily": daily_by[tid],
            "examples": [refs[str(p)] for p in examples.loc[examples["tid"] == tid, "post_id"]
                         if str(p) in refs],
            "communities": [
                {"cid": int(c.cid), "posts": int(c.posts), "community_posts": int(comm_total[c.cid])}
                for c in comm[comm["tid"] == tid].sort_values(["posts", "cid"], ascending=[False, True]
                                                               ).itertuples()],
            "hashtags": [{"tag": g.tag, "posts": int(g.posts), "accounts": int(g.accounts)}
                         for g in tags[tags["tid"] == tid].itertuples()],
        }
        if name:
            detail["name"] = name
        files[f"topics/{tid}.json"] = detail

    assignment = (f"nearest fitted posts (kNN): {fit['knn_params']}" if fit.get("knn_params")
                  else "none: only posts placed by the fit carry a topic")
    files["topics/index.json"] = {
        "kind": "topic_index", "stamp": stamp,
        "fit": {"fitted_at": fit["fitted_at"], "fit_posts": fit["fit_posts"],
                "topics": fit["topics"], "noise_share": fit["noise_share"], "params": fit["params"],
                "assignment": assignment, "knn_posts": fit["knn_posts"],
                "window_posts": window_posts, "window_noise_posts": noise_posts,
                "listed_topics": len(topics)},
        "topics": topics,
    }
    if listed_total > len(topics):
        log.info("topics: %d of %d listed (top by window posts)", len(topics), listed_total)
    return files


def read_stage(con: duckdb.DuckDBPyConnection, source: str) -> tuple[int, str | None]:
    """Distinct non-repost window posts with text that `source` has a row for, and
    the latest day any such row is partitioned under."""
    row = con.execute(
        f"""SELECT count(DISTINCT d.post_id), max(s.dt)
            FROM _denominator d JOIN (SELECT platform_post_id, dt FROM {source}) s
              ON s.platform_post_id = d.post_id""").fetchone()
    return int(row[0]), None if row[1] is None else str(row[1])[:10]


def stage_sources() -> dict[str, str]:
    return {
        "embeddings": db.embeddings_source(PLATFORM),
        "relevance": db.relevance_source(PLATFORM),
        "hatespeech": db.hatespeech_source(PLATFORM),
        "incitement": db.incitement_source(PLATFORM),
        "labels": db.labels_source(PLATFORM),
    }


COVERAGE_BASIS = (
    "Distinct posts in the collected window that are not reposts and have text. A stage's "
    "covered posts are those it holds a row for; a stage that scores only some posts "
    "(eligibility rules) cannot reach 100%. Counts of rows, not a rate of anything."
)


def build_coverage(stamp: dict, corpus: Corpus, *, sources: dict[str, str] | None = None
                   ) -> tuple[dict, list[str]]:
    """`time/coverage.json` and the stages that could not be read (left out, not zeroed)."""
    con = corpus.con
    con.execute(
        """CREATE OR REPLACE TEMP TABLE _denominator AS
           SELECT DISTINCT post_id FROM _win
           WHERE NOT coalesce(is_repost, false) AND text IS NOT NULL AND length(trim(text)) > 0""")
    denominator = int(con.execute("SELECT count(*) FROM _denominator").fetchone()[0])
    stages, failed = [], []
    for stage, source in (sources if sources is not None else stage_sources()).items():
        try:
            covered, latest = read_stage(con, source)
        except duckdb.Error:
            log.exception("coverage: stage %s unreadable", stage)
            failed.append(stage)
            continue
        stages.append({"stage": stage, "covered_posts": covered, "latest_dt": latest})
    if con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = '_topic'").fetchone()[0]:
        row = con.execute(
            """SELECT count(DISTINCT d.post_id), max(CAST(t.assigned_at AS DATE))
               FROM _denominator d JOIN _topic t ON t.post_id = d.post_id""").fetchone()
        stages.append({"stage": "topics", "covered_posts": int(row[0]),
                       "latest_dt": None if row[1] is None else str(row[1])})
    order = {s: i for i, s in enumerate(STAGES)}
    stages.sort(key=lambda s: order[s["stage"]])
    return {"kind": "time_coverage", "stamp": stamp, "denominator_posts": denominator,
            "basis": COVERAGE_BASIS, "stages": stages}, failed
