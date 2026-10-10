"""Relevance gate on what the burst detector is allowed to re-target.

The burst detector promoted #bbnaija, #bbnaijaxpepsi, #thirstyformore,
#momsonn and #citizenweekend, whose 709 rows landed in `type=search` - a
BASELINE partition, i.e. straight into the prevalence denominator. The account
gate is tested with v2 promotion in `test_v2_promotion.py`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.adaptive import bursting_hashtags

NOW = datetime.now(timezone.utc)


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
