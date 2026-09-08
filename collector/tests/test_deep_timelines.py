"""Offline tests for the deep-timeline pass.

No R2: `posts_view`, `authors_view` and `coord2_scores_view` are registered
in-memory tables and storage is a stub, so the same SQL that runs against the
parquet globs is exercised locally.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import duckdb
import pyarrow as pa
import pytest

from kenya_monitor.collectors.base import Author, Post
from kenya_monitor.collectors.x import MAX_AGE_DAYS, XCollector
from kenya_monitor.deep_timelines import (
    MIN_ENTITIES,
    SOURCE_SCORES,
    SOURCE_SUSPICION,
    TARGET_TYPE,
    DeepEntry,
    DeepTarget,
    candidate_accounts,
    collect_deep_timelines,
    deepened_expr,
    is_due,
    load_state,
    not_due_ids,
    save_state,
    timeline_summary,
)

NOW = datetime.now(timezone.utc)

# `dt` mirrors the hive partition key every R2 read exposes; the activity stage
# prunes on it when a lookback is set, so a fixture without it cannot catch a
# wrong pruning predicate.
_POSTS_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("platform_post_id", pa.string()),
        ("author_id", pa.string()),
        ("author_handle", pa.string()),
        ("text", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("collected_at", pa.timestamp("us", tz="UTC")),
        ("repost_of_id", pa.string()),
        ("dt", pa.date32()),
    ]
)

# `filename` is part of the real view (`storage.coord2_scores_view` reads with
# filename=true) because provenance has to record WHICH scores run a target
# came from - each object under kind=scores is a complete ranking of its own.
_SCORES_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("centrality", pa.float64()),
        ("kenya_share", pa.float64()),
        ("computed_at", pa.timestamp("us", tz="UTC")),
        ("filename", pa.string()),
    ]
)


def _posts(rows: list[dict]) -> pa.Table:
    defaults = {
        "platform": "x",
        "platform_post_id": "p0",
        "author_id": "a0",
        "author_handle": "h0",
        "text": "hello",
        "created_at": NOW,
        "collected_at": NOW,
        "repost_of_id": None,
    }
    rows = [{**defaults, **r} for r in rows]
    for row in rows:
        row.setdefault("dt", row["collected_at"].date())
    return pa.Table.from_pylist(rows, schema=_POSTS_SCHEMA)


def _scores(rows: list[dict]) -> pa.Table:
    defaults = {
        "user_id": "a0",
        "centrality": 0.1,
        "kenya_share": 0.5,
        "computed_at": NOW,
        "filename": "r2://b/coord2/platform=x/kind=scores/dt=2026-09-08/run=A.parquet",
    }
    return pa.Table.from_pylist([{**defaults, **r} for r in rows], schema=_SCORES_SCHEMA)


def _con(posts: list[dict], scores: list[dict] | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("posts_tbl", _posts(posts))
    if scores is not None:
        con.register("scores_tbl", _scores(scores))
    return con


def _authored(author: str, n: int, *, reposts: list[str] | None = None, **kw) -> list[dict]:
    """`n` distinct posts by one author, optionally each reposting an object."""
    reposts = reposts or [None] * n
    return [
        {
            "platform_post_id": f"{author}-p{i}",
            "author_id": author,
            "repost_of_id": reposts[i],
            **kw,
        }
        for i in range(n)
    ]


def _select(con, **kw) -> list[DeepTarget]:
    kw.setdefault("limit", 10)
    return candidate_accounts(con, "posts_tbl", scores_view="scores_tbl", **kw)


# --- targeting ---------------------------------------------------------------


def test_strata_put_the_thinnest_accounts_first():
    """The rule the whole task turns on. Cost is ~constant per account (~10
    paginated requests at depth 200); benefit is not. An account holding one
    post has a degenerate one-hot vector and is discarded by v2's floor, so 200
    fetched posts make it scorable; an account holding 200 can at best double
    its history."""
    posts = (
        _authored("thin", 1)
        + _authored("mid", 50)
        + _authored("fat", 400)
    )
    con = _con(
        posts,
        [
            # Centrality deliberately INVERTED against need: the fat account is
            # the highest-ranked target, and must still come last.
            {"user_id": "thin", "centrality": 0.01},
            {"user_id": "mid", "centrality": 0.50},
            {"user_id": "fat", "centrality": 0.99},
        ],
    )
    got = _select(con, depth=200)

    assert [t.user_id for t in got] == ["thin", "mid", "fat"]
    assert [t.stratum for t in got] == [0, 1, 2]


def test_centrality_orders_within_a_stratum():
    """Inside a stratum the ranking is pure centrality - the only ordering here
    we actually believe. No blended `centrality / log(1 + n_posts)` score,
    because that weights two incommensurable quantities by a constant nobody
    can defend."""
    posts = _authored("lo", 1) + _authored("hi", 1) + _authored("mid", 1)
    con = _con(
        posts,
        [
            {"user_id": "lo", "centrality": 0.1},
            {"user_id": "hi", "centrality": 0.9},
            {"user_id": "mid", "centrality": 0.5},
        ],
    )
    got = _select(con)

    assert [t.user_id for t in got] == ["hi", "mid", "lo"]
    assert {t.stratum for t in got} == {0}


def test_the_saturation_boundary_moves_with_depth():
    """Stratum 2 is defined by `depth`, not by a fixed number: an account is
    saturated relative to the pass being run, not in the abstract."""
    con = _con(_authored("a", 30), [{"user_id": "a", "centrality": 0.5}])

    assert _select(con, depth=200)[0].stratum == 1
    assert _select(con, depth=20)[0].stratum == 2


def test_limit_bounds_the_pass_to_the_head_of_the_ranking():
    posts = _authored("thin", 1) + _authored("mid", 50) + _authored("fat", 400)
    con = _con(
        posts,
        [
            {"user_id": "thin", "centrality": 0.1},
            {"user_id": "mid", "centrality": 0.2},
            {"user_id": "fat", "centrality": 0.3},
        ],
    )
    assert [t.user_id for t in _select(con, limit=1, depth=200)] == ["thin"]
    assert [t.user_id for t in _select(con, limit=2, depth=200)] == ["thin", "mid"]


def test_targets_holding_nothing_are_kept():
    """LEFT JOIN, not inner. Census-discovered accounts are known only as
    retweeter ids - 168,356 of them against 73,876 with posts, measured
    2026-08-02 - and they are the accounts depth buys the most for, so an inner
    join would drop exactly the head of the ranking."""
    con = _con(_authored("known", 5), [{"user_id": "ghost"}, {"user_id": "known"}])
    got = _select(con, depth=200)

    assert got[0].user_id == "ghost"
    assert got[0].held_posts == 0
    assert got[0].stratum == 0


def test_held_posts_ignore_recollection_snapshots():
    """Re-collecting the same post must not promote an account out of stratum 0.
    The count is DISTINCT on post id, which is also why the query needs no
    latest-snapshot window: `author_id` and `repost_of_id` cannot change for a
    given post id."""
    rows = _authored("a", 1)
    resnapshot = [{**r, "collected_at": NOW - timedelta(hours=3)} for r in rows]
    con = _con(rows + resnapshot, [{"user_id": "a"}])
    got = _select(con)

    assert got[0].held_posts == 1
    assert got[0].stratum == 0


def test_entity_count_is_reported_beside_the_post_count():
    """The known imprecision, made visible rather than hidden. Five retweets of
    ONE object are five posts and one co_retweet entity, so this account clears
    the post floor while being below v2's real floor in the trace that carries
    47.4% of our edges."""
    posts = _authored("narrow", 5, reposts=["OBJ"] * 5) + _authored(
        "broad", 5, reposts=[f"O{i}" for i in range(5)]
    )
    con = _con(posts, [{"user_id": "narrow"}, {"user_id": "broad"}])
    by_id = {t.user_id: t for t in _select(con, depth=200)}

    assert by_id["narrow"].held_posts == 5
    assert by_id["narrow"].held_entities == 1
    assert by_id["broad"].held_entities == 5
    # Both sit in stratum 1 on posts: the strata do not see the difference,
    # which is why the entity count is carried through to the ledger and to
    # `deep_timelines/` for whoever reads the pass later.
    assert by_id["narrow"].stratum == by_id["broad"].stratum == 1


def test_only_the_latest_scores_run_is_targeted():
    """Each object under `kind=scores` is a complete top-N ranking of its own,
    so unioning two runs produces an order that belongs to neither.
    `computed_at` discriminates rather than the `dt=` partition, because two
    runs can share a day."""
    old, new = NOW - timedelta(days=2), NOW
    con = _con(
        _authored("stale", 1) + _authored("current", 1),
        [
            {"user_id": "stale", "computed_at": old, "centrality": 0.99},
            {"user_id": "current", "computed_at": new, "centrality": 0.01},
        ],
    )
    stats: dict = {}
    got = _select(con, stats=stats)

    assert [t.user_id for t in got] == ["current"]
    assert stats["targets"] == 1
    assert stats["source"] == SOURCE_SCORES


def test_provenance_records_which_scores_run_a_target_came_from():
    con = _con(_authored("a", 1), [{"user_id": "a", "filename": "r2://b/run=Z.parquet"}])
    assert _select(con)[0].source_run == "r2://b/run=Z.parquet"


def test_min_kenya_share_is_an_opt_in():
    """Off by default. Relevance is v2's documented credibility problem - the
    top 500 average 0.404 Kenya share and the gate's recall is 0.508, so every
    such figure is a floor - but a share cut is a second arbitrary threshold and
    must be chosen deliberately, not inherited."""
    posts = _authored("kenyan", 1) + _authored("offtopic", 1)
    scores = [
        {"user_id": "kenyan", "kenya_share": 0.8, "centrality": 0.1},
        {"user_id": "offtopic", "kenya_share": 0.0, "centrality": 0.9},
    ]
    con = _con(posts, scores)

    assert [t.user_id for t in _select(con)] == ["offtopic", "kenyan"]
    assert [t.user_id for t in _select(con, min_kenya_share=0.15)] == ["kenyan"]


def test_falls_back_to_suspicion_when_no_scores_exist():
    """So the command works before a v2 pass has ever run. A weaker target set
    and not a substitute: suspicion is a bot-likeness heuristic and says nothing
    about acting together, which is the signal depth is collected for."""
    con = duckdb.connect()
    con.register("posts_tbl", _posts(_authored("a1", 1) + _authored("a2", 1)))
    con.register(
        "authors_tbl",
        pa.table(
            {
                "platform": ["x", "x"],
                "platform_user_id": ["a1", "a2"],
                "handle": ["suspect", "realnews"],
                "bio": ["", "Kenya elections reporter"],
                "followers_count": [10, 500_000],
                "following_count": [2000, 400],
                "tweet_count": [500, 10_000],
                "created_at": [NOW, NOW],
                "profile_image_url": [
                    "https://x.com/default_profile.png",
                    "https://x.com/pic.jpg",
                ],
                "collected_at": [NOW, NOW],
            }
        ),
    )
    stats: dict = {}
    got = candidate_accounts(
        con, "posts_tbl", limit=10, authors_view="authors_tbl", stats=stats
    )

    assert stats["source"] == SOURCE_SUSPICION
    assert [t.user_id for t in got] == ["a1", "a2"]
    assert {t.rank_metric for t in got} == {"suspicion"}
    # The fallback keys on user_id, not handle, so the ledger and the
    # deepened-accounts record join onto coord2 scores either way.
    assert all(t.user_id and not t.user_id.startswith("@") for t in got)


def test_no_target_source_at_all_is_an_error_not_an_empty_pass():
    """An empty selection and an unavailable target set are different states,
    and silently returning nothing is how a pass looks healthy while doing
    nothing."""
    con = _con(_authored("a", 1))
    with pytest.raises(ValueError, match="suspicion fallback"):
        candidate_accounts(con, "posts_tbl", limit=10)


def test_stats_report_both_denominators():
    """They differ by three orders of magnitude on the real corpus - 500 v2
    targets against 331,138 authors - so quoting the corpus-wide figure as the
    outcome of a bounded pass misdescribes the command."""
    posts = (
        _authored("t1", 1)
        + _authored("t2", 5, reposts=[f"O{i}" for i in range(5)])
        + _authored("bystander", 1)
        + _authored("other", 3, reposts=["A", "B", "C"])
    )
    con = _con(posts, [{"user_id": "t1"}, {"user_id": "t2"}])
    stats: dict = {}
    _select(con, depth=200, stats=stats)

    assert stats["authors"] == 4
    assert stats["authors_clearing_floor"] == 2  # t2 and other
    assert stats["authors_clearing_entity_floor"] == 2
    assert stats["targets"] == 2
    assert stats["targets_below_floor"] == 1
    assert stats["targets_clearing_floor"] == 1
    assert stats["selected"] == 2
    assert stats["selected_below_floor"] == 1
    # Pool budget is the binding constraint, so the pass is costed in requests
    # rather than accounts: ~10 pages per account at 20 posts a page.
    assert stats["requests_estimate"] == 20


def test_window_prunes_the_activity_scan():
    old = NOW - timedelta(days=60)
    posts = _authored("a", 3, collected_at=old, dt=old.date()) + _authored("b", 3)
    con = _con(posts, [{"user_id": "a"}, {"user_id": "b"}])

    windowed = {t.user_id: t.held_posts for t in _select(con, depth=200, lookback_days=2)}
    whole = {t.user_id: t.held_posts for t in _select(con, depth=200)}
    assert windowed == {"a": 0, "b": 3}
    assert whole == {"a": 3, "b": 3}


def test_each_aggregate_is_staged_and_the_widest_one_is_released():
    """Staged, not one query. A query whose every stage fits can still OOM when
    DuckDB runs the stages concurrently and their peaks add - measured on pi0
    2026-09-02, three stages of 569s, 261s and 1,248s each fitting alone while
    the fused form OOMed. `_dt_posts` is the widest relation here (one row per
    distinct post, 1,045,318 on the snapshot, against 331,138 authors) and
    nothing reads it after the two aggregates."""
    con = _con(_authored("a", 2, reposts=["X", "Y"]), [{"user_id": "a"}])
    _select(con)

    staged = {
        r[0]
        for r in con.sql(
            "SELECT table_name FROM duckdb_tables() WHERE table_name LIKE '_dt_%'"
        ).fetchall()
    }
    assert {"_dt_targets", "_dt_activity", "_dt_entities", "_dt_candidates"} <= staged
    assert "_dt_posts" not in staged


# --- the two resumability mechanisms ----------------------------------------


def test_r2_partition_excludes_recently_deepened_accounts():
    """The self-healing half: posts in `type=deep_timeline` inside the TTL are
    evidence the account was deepened, so this survives losing the local ledger
    - `census.censused_expr`'s reasoning. Applied in the candidate SQL, BEFORE
    the LIMIT: `census.threaded_expr` records what happens otherwise, a
    deterministic ranking re-picking completed work every pass (495 selected,
    10-35 fetched)."""
    con = _con(_authored("fresh", 1) + _authored("due", 1), [{"user_id": "fresh"}, {"user_id": "due"}])
    con.register(
        "deep_tbl",
        _posts(
            _authored("fresh", 1, collected_at=NOW)
            + _authored("due", 1, collected_at=NOW - timedelta(days=90))
        ),
    )
    got = _select(con, deep_posts_view="deep_tbl", refresh_days=30)

    assert [t.user_id for t in got] == ["due"]


def test_absent_deep_partition_selects_normally():
    """A first run has no `type=deep_timeline` objects at all, and read_parquet
    raises on an empty glob. That must select normally rather than nothing."""
    con = _con(_authored("a", 1), [{"user_id": "a"}])
    assert deepened_expr(con, "read_parquet('nope/*.parquet')", "c.user_id") == "FALSE"
    assert len(_select(con, deep_posts_view="read_parquet('nope/*.parquet')")) == 1


def test_deepened_expr_requires_a_qualified_column():
    """A bare outer column that also exists on the inner side binds to the
    inner table, making the predicate trivially true for every row - so every
    candidate is excluded and the pass stalls completely."""
    con = _con(_authored("a", 1))
    con.register("deep_tbl", _posts(_authored("a", 1)))
    with pytest.raises(ValueError, match="table-qualified"):
        deepened_expr(con, "deep_tbl", "user_id")


def test_ledger_blocks_accounts_the_partition_can_never_learn_about():
    """An account that returned nothing writes no post row, so no anti-join can
    exclude it. Selection is deterministic, so without the ledger it occupies
    the same slot at the head of the ranking on every pass forever."""
    con = _con(_authored("dead", 1) + _authored("alive", 1),
               [{"user_id": "dead", "centrality": 0.9}, {"user_id": "alive", "centrality": 0.1}])

    assert [t.user_id for t in _select(con)] == ["dead", "alive"]
    assert [t.user_id for t in _select(con, blocked=["dead"])] == ["alive"]


# --- ledger ------------------------------------------------------------------


def test_state_roundtrip(tmp_path):
    path = tmp_path / "deep_timeline.json"
    save_state({"u1": DeepEntry(fetched_at=NOW.isoformat(), posts=180)}, path)
    loaded = load_state(path)
    assert loaded["u1"].status == "ok"
    assert loaded["u1"].posts == 180
    assert load_state(tmp_path / "absent.json") == {}


def test_the_ttl_is_what_makes_a_second_pass_extend_coverage():
    """A deepened account is not due again until `refresh_days`, so the next
    pass spends its ~10 requests per account on accounts never deepened rather
    than re-fetching a history already held. At a fixed depth a refresh only
    adds what the account posted since last time."""
    fresh = DeepEntry(fetched_at=(NOW - timedelta(days=5)).isoformat(), status="ok")
    stale = DeepEntry(fetched_at=(NOW - timedelta(days=40)).isoformat(), status="ok")

    assert not is_due(fresh, refresh_days=30)
    assert is_due(stale, refresh_days=30)
    assert is_due(None, refresh_days=30)
    assert not_due_ids({"a": fresh, "b": stale}, refresh_days=30) == ["a"]


def test_empty_is_terminal_but_a_failed_request_retries():
    """A by-id fetch cannot tell suspended from protected from empty; all three
    yield nothing and none change on this project's timescale. A rate-limited
    request says nothing about the account."""
    empty = DeepEntry(fetched_at=NOW.isoformat(), status="no_posts")
    failed_once = DeepEntry(
        fetched_at=(NOW - timedelta(days=40)).isoformat(), status="failed", attempts=1
    )
    failed_out = DeepEntry(
        fetched_at=(NOW - timedelta(days=40)).isoformat(), status="failed", attempts=3
    )

    assert not is_due(empty, refresh_days=30)
    assert not is_due(empty, refresh_days=0)
    assert is_due(failed_once, refresh_days=30, max_attempts=3)
    assert not is_due(failed_out, refresh_days=30, max_attempts=3)
    # A failure is not more urgent than a success: it waits for the TTL too.
    # `follow_crawl.is_due` records what treating it as due-now cost, which was
    # unresolvable accounts occupying the head of the queue on every pass.
    recent_failure = DeepEntry(fetched_at=NOW.isoformat(), status="failed", attempts=1)
    assert not is_due(recent_failure, refresh_days=30)


def test_summary_reports_floor_movement_not_accounts_fetched():
    entries = {
        # Was below the floor, now clears it: the quantity of record.
        "moved": DeepEntry(fetched_at=NOW.isoformat(), posts=180, held_posts_before=1),
        # Already cleared it; depth improved its vector but moved no coverage.
        "already": DeepEntry(fetched_at=NOW.isoformat(), posts=200, held_posts_before=60),
        # Dormant: resolved, one post, still below the floor.
        "dormant": DeepEntry(fetched_at=NOW.isoformat(), posts=1, held_posts_before=0),
        "gone": DeepEntry(fetched_at=NOW.isoformat(), status="no_posts", held_posts_before=1),
    }
    summary = timeline_summary(entries, refresh_days=30)

    assert summary["tracked"] == 4
    assert summary["ok"] == 3
    assert summary["no_posts"] == 1
    assert summary["floor_cleared"] == 1
    assert summary["posts"] == 381
    assert summary["due_now"] == 0
    assert timeline_summary({})["floor_cleared"] == 0


# --- the pass ----------------------------------------------------------------


class _StubStorage:
    """The Storage surface `collect_deep_timelines` touches."""

    def __init__(self, con: duckdb.DuckDBPyConnection | None = None):
        self.con = con or duckdb.connect()
        self.writes: list[tuple[str, int]] = []
        self.written_posts: list[Post] = []
        self.written_authors: list[Author] = []
        self.runs: list[dict] = []

    def posts_view(self, platform: str = "*", target_type: str = "*") -> str:
        return "deep_tbl" if target_type == TARGET_TYPE else "posts_tbl"

    def authors_view(self, platform: str = "*") -> str:
        return "authors_tbl"

    def coord2_scores_view(self, platform: str = "x") -> str:
        return "scores_tbl"

    def write_posts(self, posts, target_type: str, now=None) -> str | None:
        if not posts:
            return None
        self.writes.append((target_type, len(posts)))
        self.written_posts.extend(posts)
        return f"posts/type={target_type}/run=test.parquet"

    def write_authors(self, authors, now=None) -> str | None:
        self.written_authors.extend(authors)
        return "authors/run=test.parquet" if authors else None

    def write_deep_timeline_run(self, rows, platform: str = "x", now=None) -> str | None:
        if not rows:
            return None
        self.runs.extend(rows)
        return "deep_timelines/run=test.parquet"


class _StubCollector:
    """`deep_timeline` per the real contract: yields nothing for an account that
    cannot be read, raises for a failed request."""

    platform = "x"

    def __init__(
        self,
        posts_per_account: int = 3,
        empty: set[str] = frozenset(),
        raising: set[str] = frozenset(),
    ):
        self.posts_per_account = posts_per_account
        self.empty = set(empty)
        self.raising = set(raising)
        self.requested: list[tuple[str, int, bool]] = []
        self._authors: dict[str, Author] = {}

    async def deep_timeline(self, user_id, limit, include_replies=True):
        self.requested.append((user_id, limit, include_replies))
        if user_id in self.raising:
            raise RuntimeError("rate limited")
        if user_id in self.empty:
            return
        self._authors[user_id] = Author(
            platform="x", platform_user_id=user_id, handle=f"h-{user_id}"
        )
        for i in range(min(limit, self.posts_per_account)):
            yield Post(
                platform="x",
                platform_post_id=f"{user_id}-deep-{i}",
                author_id=user_id,
                author_handle=f"h-{user_id}",
                text=f"post {i}",
                created_at=NOW - timedelta(days=90 + i),
                url=f"https://x.com/h-{user_id}/status/{i}",
            )

    def collected_authors(self) -> list[Author]:
        out = list(self._authors.values())
        self._authors = {}
        return out


def _run(coro):
    return asyncio.run(coro)


def _target(uid: str, **kw) -> DeepTarget:
    return DeepTarget(user_id=uid, **kw)


def test_rows_land_in_the_quarantined_targeted_partition(tmp_path):
    """The constraint the whole task hangs on: `timeline` is a BASELINE type,
    and per-account histories there would move the baseline composition further
    than the 2026-08-06 conversation widening did - the change that made the raw
    toxicity series unpublishable."""
    storage, collector = _StubStorage(), _StubCollector()
    _run(
        collect_deep_timelines(
            collector, storage, limit=2, depth=10,
            state_path=tmp_path / "s.json",
            targets=[_target("u1"), _target("u2")],
        )
    )
    assert {t for t, _ in storage.writes} == {TARGET_TYPE}
    assert TARGET_TYPE == "deep_timeline"
    assert "timeline" not in {t for t, _ in storage.writes}


def test_depth_bounds_the_fetch_and_replies_are_on(tmp_path):
    """Depth 200, not the ~3,200 the endpoint can reach: the marginal value of
    page 40 on one account is far below page 1 on another and the pool is the
    binding constraint. Replies on, because plain `user_tweets` omits them
    entirely - measured 2026-08-03, 760 posts from 20 accounts carried 5 rows
    with `in_reply_to_id`."""
    storage, collector = _StubStorage(), _StubCollector(posts_per_account=1000)
    counts = _run(
        collect_deep_timelines(
            collector, storage, limit=1, depth=200,
            state_path=tmp_path / "s.json", targets=[_target("u1")],
        )
    )
    assert collector.requested == [("u1", 200, True)]
    assert counts["posts"] == 200


def test_limit_bounds_the_pass_even_when_handed_more_targets(tmp_path):
    """Never an unbounded drain, whatever the caller passes."""
    storage, collector = _StubStorage(), _StubCollector()
    counts = _run(
        collect_deep_timelines(
            collector, storage, limit=2, depth=10,
            state_path=tmp_path / "s.json",
            targets=[_target(f"u{i}") for i in range(10)],
        )
    )
    assert counts["selected"] == 2
    assert [u for u, _, _ in collector.requested] == ["u0", "u1"]


def test_the_pass_counts_accounts_moved_over_the_floor(tmp_path):
    """Accounts fetched is not the quantity anyone cares about. Counted on what
    came BACK: an account that returned nothing clears no floor, and a dormant
    account that returned one post does not either."""
    storage = _StubStorage()
    collector = _StubCollector(posts_per_account=5, empty={"gone"})

    class _OnePost(_StubCollector):
        async def deep_timeline(self, user_id, limit, include_replies=True):
            n = 1 if user_id == "dormant" else self.posts_per_account
            async for p in _StubCollector.deep_timeline(self, user_id, min(limit, n), include_replies):
                yield p

    counts = _run(
        collect_deep_timelines(
            _OnePost(posts_per_account=5, empty={"gone"}),
            storage,
            limit=4,
            depth=200,
            state_path=tmp_path / "s.json",
            targets=[
                _target("thin", held_posts=0, stratum=0),
                _target("dormant", held_posts=0, stratum=0),
                _target("already", held_posts=60, stratum=1),
                _target("gone", held_posts=1, stratum=0),
            ],
        )
    )
    assert counts["deepened"] == 3
    assert counts["no_posts"] == 1
    # thin only: dormant came back with 1 post and stayed below the floor,
    # already was above it before the pass, gone returned nothing.
    assert counts["floor_cleared"] == 1
    assert collector.posts_per_account == 5  # sanity: stub not mutated


def test_provenance_records_the_pre_treatment_state(tmp_path):
    """The feedback-artefact check needs the treated set AND what it held
    before treatment, keyed on `user_id` so it joins onto coord2 scores. Without
    the pre-treatment counts there is no covariate to condition on."""
    storage = _StubStorage()
    _run(
        collect_deep_timelines(
            _StubCollector(posts_per_account=7),
            storage,
            limit=1,
            depth=200,
            state_path=tmp_path / "s.json",
            targets=[
                _target(
                    "u1", held_posts=1, held_entities=1, stratum=0,
                    rank_value=0.42, rank_metric="centrality",
                    source=SOURCE_SCORES, source_run="r2://b/run=A.parquet",
                )
            ],
        )
    )
    assert len(storage.runs) == 1
    row = storage.runs[0]
    assert row["user_id"] == "u1"
    assert row["held_posts_before"] == 1
    assert row["held_entities_before"] == 1
    assert row["rank_value"] == 0.42
    assert row["source_run"] == "r2://b/run=A.parquet"
    assert row["depth"] == 200
    assert row["posts_written"] == 7
    assert row["status"] == "ok"


def test_an_empty_account_is_recorded_and_never_retried(tmp_path):
    path = tmp_path / "s.json"
    storage = _StubStorage()
    counts = _run(
        collect_deep_timelines(
            _StubCollector(empty={"gone"}, raising={"boom"}),
            storage,
            limit=3,
            depth=10,
            state_path=path,
            targets=[_target("ok1"), _target("gone"), _target("boom")],
        )
    )
    assert (counts["deepened"], counts["no_posts"], counts["failed"]) == (1, 1, 1)

    entries = load_state(path)
    assert entries["gone"].status == "no_posts"
    assert entries["boom"].status == "failed"
    assert entries["ok1"].status == "ok"
    # The empty account is terminal even with the TTL fully lapsed; the failure
    # is not.
    assert not is_due(entries["gone"], refresh_days=0)
    assert is_due(entries["boom"], refresh_days=0)
    # A failed account writes no provenance row: nothing was collected for it.
    assert {r["user_id"] for r in storage.runs} == {"ok1", "gone"}


def test_a_failure_gives_up_after_max_attempts(tmp_path):
    path = tmp_path / "s.json"
    for _ in range(2):
        _run(
            collect_deep_timelines(
                _StubCollector(raising={"boom"}),
                _StubStorage(),
                limit=1,
                depth=10,
                state_path=path,
                refresh_days=0,
                max_attempts=2,
                targets=[_target("boom")],
            )
        )
    entries = load_state(path)
    assert entries["boom"].attempts == 2
    assert not is_due(entries["boom"], refresh_days=0, max_attempts=2)


def test_partial_progress_survives_an_abort_mid_pass(tmp_path):
    """Flushing every `flush_every` accounts is what stops a rate-limit abort
    discarding every request made since the pass started - and each account is
    ~10 requests, not one."""
    path = tmp_path / "s.json"
    storage = _StubStorage()

    class _Aborting(_StubCollector):
        async def deep_timeline(self, user_id, limit, include_replies=True):
            if user_id == "u3":
                raise KeyboardInterrupt
            async for p in _StubCollector.deep_timeline(self, user_id, limit, include_replies):
                yield p

    with pytest.raises(KeyboardInterrupt):
        _run(
            collect_deep_timelines(
                _Aborting(posts_per_account=4),
                storage,
                limit=4,
                depth=10,
                state_path=path,
                flush_every=2,
                targets=[_target(f"u{i}") for i in range(1, 5)],
            )
        )
    assert storage.writes == [(TARGET_TYPE, 8)]
    assert set(load_state(path)) == {"u1", "u2"}
    assert {r["user_id"] for r in storage.runs} == {"u1", "u2"}


def test_authors_are_written_so_deepened_accounts_get_profiles(tmp_path):
    storage = _StubStorage()
    _run(
        collect_deep_timelines(
            _StubCollector(), storage, limit=1, depth=10,
            state_path=tmp_path / "s.json", targets=[_target("u1")],
        )
    )
    assert [a.platform_user_id for a in storage.written_authors] == ["u1"]


def test_an_empty_candidate_set_is_not_an_error(tmp_path):
    counts = _run(
        collect_deep_timelines(
            _StubCollector(), _StubStorage(), limit=10, depth=10,
            state_path=tmp_path / "s.json", targets=[],
        )
    )
    assert counts["selected"] == 0
    assert counts["deepened"] == 0


def test_the_pass_selects_through_the_query_and_resumes(tmp_path):
    """End to end on the real SQL: selection, strata, the ledger TTL and the
    targeted write, with only the network stubbed.

    `posts_tbl` deliberately does not grow with what was collected, and
    `deep_tbl` stays empty - that isolates the ledger from the R2 exclusion,
    which would otherwise mask it."""
    path = tmp_path / "s.json"
    posts = _authored("thin", 1) + _authored("mid", 50) + _authored("fat", 400)
    con = _con(
        posts,
        [
            {"user_id": "thin", "centrality": 0.1},
            {"user_id": "mid", "centrality": 0.2},
            {"user_id": "fat", "centrality": 0.3},
        ],
    )
    con.register("deep_tbl", _posts([]))
    storage = _StubStorage(con)

    first = _StubCollector()
    stats: dict = {}
    counts = _run(
        collect_deep_timelines(
            first, storage, limit=1, depth=200, state_path=path, stats=stats
        )
    )
    assert [u for u, _, _ in first.requested] == ["thin"]
    assert counts["floor_cleared"] == 1
    assert stats["targets_clearing_floor"] == 2
    assert storage.writes == [(TARGET_TYPE, 3)]

    second = _StubCollector()
    _run(
        collect_deep_timelines(
            second, _StubStorage(con), limit=2, depth=200, state_path=path
        )
    )
    assert [u for u, _, _ in second.requested] == ["mid", "fat"]


# --- the collector method ----------------------------------------------------


def _fake_tweet(age_days: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"t{age_days}", date=NOW - timedelta(days=age_days)
    )


class _FakeApi:
    def __init__(self, ages: list[int]):
        self.ages = ages
        self.calls: list[tuple[str, object]] = []

    async def user_by_login(self, handle):
        self.calls.append(("user_by_login", handle))
        return SimpleNamespace(id=999)

    async def user_tweets_and_replies(self, uid, limit=-1):
        self.calls.append(("user_tweets_and_replies", uid))
        for age in self.ages:
            yield _fake_tweet(age)

    async def user_tweets(self, uid, limit=-1):
        self.calls.append(("user_tweets", uid))
        for age in self.ages:
            yield _fake_tweet(age)


def _collector_with(monkeypatch, ages: list[int]) -> tuple[XCollector, _FakeApi]:
    api = _FakeApi(ages)
    collector = XCollector(api)
    monkeypatch.setattr(
        XCollector,
        "_to_post",
        lambda self, tw: Post(
            platform="x", platform_post_id=str(tw.id), author_id="u",
            author_handle="h", text="t", created_at=tw.date, url="",
        ),
    )
    return collector, api


def test_deep_timeline_carries_no_age_cutoff(monkeypatch):
    """The single most important difference from `timeline`, and the one that
    would have silently voided the whole task. `timeline` drops anything older
    than MAX_AGE_DAYS so the baseline partition stays comparable with `search`,
    which cannot reach past 14 days at all. Applied here it would spend ~10
    requests per account and keep only what a 14-day keyword search could
    already have found - on a corpus averaging 3.2 posts per author, almost all
    the history worth fetching is older than the window."""
    ages = [1, MAX_AGE_DAYS + 1, 400]
    collector, api = _collector_with(monkeypatch, ages)

    deep = _run(_drain(collector.deep_timeline("12345", limit=50)))
    assert len(deep) == 3, "an age cutoff here discards the depth being paid for"

    shallow = _run(_drain(collector.timeline("someone", limit=50)))
    assert len(shallow) == 1
    assert ("user_tweets_and_replies", 12345) in api.calls


def test_deep_timeline_addresses_the_account_by_id_not_by_handle(monkeypatch):
    """Saves one request per account (~9% of a ~10-request budget) and removes a
    false terminal outcome for every account renamed since the corpus saw it.
    The v2 scores this targets carry `user_id`, so a handle round-trip would be
    pure loss."""
    collector, api = _collector_with(monkeypatch, [1])
    _run(_drain(collector.deep_timeline("777", limit=5)))

    assert [c for c, _ in api.calls] == ["user_tweets_and_replies"]
    assert "user_by_login" not in [c for c, _ in api.calls]


def test_deep_timeline_defaults_to_replies_and_timeline_does_not(monkeypatch):
    """Reversed defaults, deliberately. `include_replies` changes what a
    timeline MEANS for a prevalence measurement built on the baseline partition,
    so `timeline` leaves it off; a history with no replies carries no reply
    behaviour to trace at all, so this leaves it on."""
    collector, api = _collector_with(monkeypatch, [1])
    _run(_drain(collector.deep_timeline("1", limit=5)))
    _run(_drain(collector.timeline("someone", limit=5)))

    endpoints = [c for c, _ in api.calls]
    assert endpoints.count("user_tweets_and_replies") == 1
    assert endpoints.count("user_tweets") == 1


async def _drain(gen) -> list[Post]:
    return [p async for p in gen]
