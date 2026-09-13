"""Relevance gates on what coordination output is allowed to re-target.

The loop from analysis back into collection is the sharpest instrument here and
the easiest to point at the wrong thing. Two measurements set these gates:

- corroborated clusters are 8.3% Kenya-referencing against 51.5% for
  single-channel ones (2026-08-12), because reciprocal engagement pods are the
  most abundant coordination on the platform and co_reply detects them best
- the burst detector promoted #bbnaija, #bbnaijaxpepsi, #thirstyformore,
  #momsonn and #citizenweekend, whose 709 rows landed in `type=search` - a
  BASELINE partition, i.e. straight into the prevalence denominator
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.adaptive import bursting_hashtags, cluster_accounts

NOW = datetime.now(timezone.utc)


def _con(clusters: list[dict], authors: list[dict], posts: list[dict], hate: list[dict]):
    c = duckdb.connect()
    c.register(
        "clusters_tbl",
        pa.Table.from_pylist(
            clusters,
            schema=pa.schema([
                ("author_id", pa.string()), ("cluster_id", pa.int64()),
                ("n_channels", pa.int64()), ("internal_edge_share", pa.float64()),
                ("computed_at", pa.timestamp("us", tz="UTC")),
            ]),
        ),
    )
    c.register(
        "authors_tbl",
        pa.Table.from_pylist(
            authors,
            schema=pa.schema([
                ("platform", pa.string()), ("platform_user_id", pa.string()),
                ("handle", pa.string()),
                ("collected_at", pa.timestamp("us", tz="UTC")),
            ]),
        ),
    )
    c.register(
        "posts_tbl",
        pa.Table.from_pylist(
            posts,
            schema=pa.schema([
                ("platform", pa.string()), ("platform_post_id", pa.string()),
                ("author_id", pa.string()),
                ("collected_at", pa.timestamp("us", tz="UTC")),
            ]),
        ),
    )
    c.register(
        "hate_tbl",
        pa.Table.from_pylist(
            hate,
            schema=pa.schema([
                ("platform_post_id", pa.string()), ("domain", pa.string()),
                ("scored_at", pa.timestamp("us", tz="UTC")),
            ]),
        ),
    )
    return c


def _cluster(author_id, cluster_id, n_channels):
    return {
        "author_id": author_id, "cluster_id": cluster_id,
        "n_channels": n_channels, "internal_edge_share": 0.8, "computed_at": NOW,
    }


def _author(uid, handle):
    return {
        "platform": "x", "platform_user_id": uid, "handle": handle,
        "collected_at": NOW,
    }


def _posts_for(author_id, domains):
    posts, hate = [], []
    for i, d in enumerate(domains):
        pid = f"{author_id}-p{i}"
        posts.append({
            "platform": "x", "platform_post_id": pid,
            "author_id": author_id, "collected_at": NOW,
        })
        hate.append({"platform_post_id": pid, "domain": d, "scored_at": NOW})
    return posts, hate


def test_off_domain_cluster_is_not_promoted():
    """A corroborated pod whose posts never mention Kenya is coordination, but
    not the coordination this project exists to find."""
    kenyan_p, kenyan_h = _posts_for("1", ["kenya"] * 8 + ["ambiguous"] * 2)
    pod_p, pod_h = _posts_for("2", ["ambiguous"] * 10)
    con = _con(
        [_cluster("1", 0, 2), _cluster("2", 1, 2)],
        [_author("1", "kenyan_acct"), _author("2", "nigerian_pod")],
        kenyan_p + pod_p, kenyan_h + pod_h,
    )

    got = cluster_accounts(
        con, "clusters_tbl", "authors_tbl",
        posts_view="posts_tbl", hatespeech_view="hate_tbl",
        min_channels=2, min_kenya_share=0.15,
    )
    assert got == ["kenyan_acct"]


def _relevance(con, rows: list[dict]):
    con.register(
        "relevance_tbl",
        pa.Table.from_pylist(
            rows or [],
            schema=pa.schema([
                ("platform_post_id", pa.string()), ("p_kenya", pa.float64()),
                ("scored_at", pa.timestamp("us", tz="UTC")),
            ]),
        ),
    )
    return "relevance_tbl"


def test_the_learned_gate_promotes_a_cluster_the_lexicon_misses():
    """The keyword gate's recall is 0.655 and its misses are plain Kenyan
    politics without the anchor vocabulary - the clusters most worth chasing."""
    posts, hate = _posts_for("1", ["ambiguous"] * 10)
    con = _con([_cluster("1", 0, 2)], [_author("1", "oblique_acct")], posts, hate)
    view = _relevance(con, [{"platform_post_id": p["platform_post_id"], "p_kenya": 0.98,
                             "scored_at": NOW} for p in posts])

    assert cluster_accounts(
        con, "clusters_tbl", "authors_tbl", posts_view="posts_tbl",
        hatespeech_view="hate_tbl", min_channels=2, min_kenya_share=0.15,
    ) == [], "the lexicon alone sees nothing Kenyan here"

    assert cluster_accounts(
        con, "clusters_tbl", "authors_tbl", posts_view="posts_tbl",
        hatespeech_view="hate_tbl", min_channels=2, min_kenya_share=0.15,
        relevance_view=view,
    ) == ["oblique_acct"]


def test_the_learned_gate_also_rejects_what_the_lexicon_over_calls():
    """It cuts both ways: Indian 'DCP' posts and French 'dcp' tripped the
    lexicon, and the model drops them."""
    posts, hate = _posts_for("1", ["kenya"] * 10)
    con = _con([_cluster("1", 0, 2)], [_author("1", "false_hit")], posts, hate)
    view = _relevance(con, [{"platform_post_id": p["platform_post_id"], "p_kenya": 0.01,
                             "scored_at": NOW} for p in posts])

    assert cluster_accounts(
        con, "clusters_tbl", "authors_tbl", posts_view="posts_tbl",
        hatespeech_view="hate_tbl", min_channels=2, min_kenya_share=0.15,
        relevance_view=view,
    ) == []


def test_posts_the_scorer_has_not_reached_keep_the_lexicon_call():
    """Scores arrive a pass behind collection, so a mixed cluster is normal and
    must not be decided by the unscored half."""
    posts, hate = _posts_for("1", ["kenya"] * 10)
    con = _con([_cluster("1", 0, 2)], [_author("1", "half_scored")], posts, hate)
    view = _relevance(con, [{"platform_post_id": posts[0]["platform_post_id"],
                             "p_kenya": 0.01, "scored_at": NOW}])

    assert cluster_accounts(
        con, "clusters_tbl", "authors_tbl", posts_view="posts_tbl",
        hatespeech_view="hate_tbl", min_channels=2, min_kenya_share=0.15,
        relevance_view=view,
    ) == ["half_scored"], "nine lexicon-Kenyan posts still carry the cluster"


def test_the_latest_relevance_score_decides():
    posts, hate = _posts_for("1", ["ambiguous"] * 4)
    con = _con([_cluster("1", 0, 2)], [_author("1", "rescored")], posts, hate)
    rows = []
    for p in posts:
        rows.append({"platform_post_id": p["platform_post_id"], "p_kenya": 0.01,
                     "scored_at": NOW - timedelta(days=2)})
        rows.append({"platform_post_id": p["platform_post_id"], "p_kenya": 0.99, "scored_at": NOW})

    assert cluster_accounts(
        con, "clusters_tbl", "authors_tbl", posts_view="posts_tbl",
        hatespeech_view="hate_tbl", min_channels=2, min_kenya_share=0.15,
        relevance_view=_relevance(con, rows),
    ) == ["rescored"]


def test_single_channel_cluster_is_rejected_on_the_floor():
    p, h = _posts_for("1", ["kenya"] * 10)
    con = _con([_cluster("1", 0, 1)], [_author("1", "solo")], p, h)

    assert cluster_accounts(
        con, "clusters_tbl", "authors_tbl",
        posts_view="posts_tbl", hatespeech_view="hate_tbl", min_channels=2,
    ) == []


def test_an_unscored_cluster_is_allowed_through_and_counted(caplog):
    """Failing closed would stop targeting entirely whenever the enrichment
    worker falls behind - the 'subsystem runs but does nothing' failure."""
    con = _con([_cluster("1", 0, 2)], [_author("1", "unscored")], [], [])

    with caplog.at_level(logging.INFO, logger="kenya_monitor"):
        got = cluster_accounts(
            con, "clusters_tbl", "authors_tbl",
            posts_view="posts_tbl", hatespeech_view="hate_tbl", min_channels=2,
        )

    assert got == ["unscored"]
    assert "1 unmeasured" in caplog.text


def test_gate_disables_itself_without_a_hatespeech_view(caplog):
    p, h = _posts_for("1", ["ambiguous"] * 10)
    con = _con([_cluster("1", 0, 2)], [_author("1", "anyone")], p, h)

    with caplog.at_level(logging.WARNING, logger="kenya_monitor"):
        got = cluster_accounts(con, "clusters_tbl", "authors_tbl", min_channels=2)

    assert got == ["anyone"]
    assert "Kenya gate disabled" in caplog.text


# --- keyword promotion -------------------------------------------------------


_POSTS = pa.schema([
    ("platform", pa.string()), ("platform_post_id", pa.string()),
    ("hashtags", pa.list_(pa.string())),
    ("created_at", pa.timestamp("us", tz="UTC")),
    ("collected_at", pa.timestamp("us", tz="UTC")),
    ("dt", pa.date32()),
])
_HATE = pa.schema([
    ("platform_post_id", pa.string()), ("domain", pa.string()),
    ("scored_at", pa.timestamp("us", tz="UTC")),
])


def _tag_con(rows):
    posts, hate = [], []
    for i, (tag, domain) in enumerate(rows):
        pid = f"p{i}"
        posts.append({
            "platform": "x", "platform_post_id": pid, "hashtags": [tag],
            "created_at": NOW - timedelta(hours=1), "collected_at": NOW,
            "dt": NOW.date(),
        })
        hate.append({"platform_post_id": pid, "domain": domain, "scored_at": NOW})
    c = duckdb.connect()
    c.register("posts_tbl", pa.Table.from_pylist(posts, schema=_POSTS))
    c.register("hate_tbl", pa.Table.from_pylist(hate, schema=_HATE))
    return c


def test_globally_trending_tag_is_rejected():
    """#bbnaija cleared the old volume-and-acceleration predicate easily."""
    rows = [("bbnaija", "ambiguous")] * 25 + [("maamuzi2027", "kenya")] * 25
    con = _tag_con(rows)

    got = dict(
        bursting_hashtags(
            con, "posts_tbl", min_count=20, ratio=5.0,
            hatespeech_view="hate_tbl", min_kenya_share=0.10,
        )
    )
    assert "#maamuzi2027" in got
    assert "#bbnaija" not in got


def test_unscored_tag_is_allowed_through():
    con = _tag_con([("newtag", None)] * 25)
    got = dict(
        bursting_hashtags(
            con, "posts_tbl", min_count=20, ratio=5.0, hatespeech_view="hate_tbl"
        )
    )
    assert "#newtag" in got


def test_cluster_accounts_disabled_returns_empty_without_querying():
    """The disabled path must not touch the connection. On pi0 the candidate
    query is minutes of R2 scan, so paying for a result that is then discarded
    is the whole thing being avoided."""

    class Exploding:
        def sql(self, *a, **k):
            raise AssertionError("cluster_accounts queried while disabled")

    assert cluster_accounts(Exploding(), "clusters", "authors", enabled=False) == []
