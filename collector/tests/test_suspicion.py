from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pyarrow as pa

from kenya_monitor.suspicion import top_suspicious_handles

NOW = datetime.now(timezone.utc)


def test_top_suspicious_handles_ranks_young_high_ratio_first():
    authors = pa.table(
        {
            "platform": pa.array(["x", "x"], type=pa.string()),
            "platform_user_id": pa.array(["a1", "a2"], type=pa.string()),
            "handle": pa.array(["suspect", "realnews"], type=pa.string()),
            "bio": pa.array(["", "Kenya elections reporter"], type=pa.string()),
            "followers_count": pa.array([10, 500_000], type=pa.int64()),
            "following_count": pa.array([2000, 400], type=pa.int64()),
            "tweet_count": pa.array([500, 10_000], type=pa.int64()),
            "listed_count": pa.array([0, 1000], type=pa.int64()),
            "verified": pa.array([False, True], type=pa.bool_()),
            "blue": pa.array([False, False], type=pa.bool_()),
            "created_at": pa.array([NOW, NOW], type=pa.timestamp("us", tz="UTC")),
            "profile_image_url": pa.array(
                ["https://x.com/default_profile.png", "https://x.com/pic.jpg"],
                type=pa.string(),
            ),
            "collected_at": pa.array([NOW, NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    posts = pa.table(
        {
            "platform": pa.array(["x"], type=pa.string()),
            "platform_post_id": pa.array(["p1"], type=pa.string()),
            "author_id": pa.array(["a1"], type=pa.string()),
            "text": pa.array(["spam spam spam"], type=pa.string()),
            "collected_at": pa.array([NOW], type=pa.timestamp("us", tz="UTC")),
        }
    )
    con = duckdb.connect()
    con.register("authors_tbl", authors)
    con.register("posts_tbl", posts)
    handles = top_suspicious_handles(con, "authors_tbl", "posts_tbl", n=1)
    assert handles == ["suspect"]
