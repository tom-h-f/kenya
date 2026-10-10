"""Embed the text-eligible posts of one window that have no vector yet.

    from kma import embed_window, window
    embed_window.embed_missing(con, window.Window.ending("", days=30))

The v2 text trace can only see posts that carry a vector, and the live enrich
worker embeds a fixed number of posts per cycle against about 40,000 new posts
a day, so the trace saw 12 to 24% of what people wrote (measured 2026-10-10).
This is the stage `kma.v2_daily_run` runs before it pins the window, so the
trace is built over a window whose vectors are all there.

"Eligible" is `coord2.text_rows` and nothing else: non-retweets with a user and
at least `MIN_TEXT_WORDS` cleaned words, about 45% of the posts that have
text. Retweets and short texts are never read by the trace, so they are not
encoded here; the enrich worker still embeds them when it reaches them.

The posts are read the way the window pins them (`dt` in `[start, end]`,
projected, latest row per post) and the vectors already in R2 are matched by id
in SQL, so only the missing posts are cleaned and encoded. A window that is
already covered costs two reads and no model load.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import duckdb

from kma import coord2, semantic, window
from kma.db import embeddings_source, posts_source

log = logging.getLogger(__name__)

PLATFORM = "x"


def _window_posts(w: window.Window) -> str:
    return f"""(
        SELECT author_id, platform_post_id, created_at, is_repost, repost_of_id,
               text, collected_at, NULL AS hashtags, NULL AS urls
        FROM {posts_source(PLATFORM)}
        WHERE dt >= DATE '{w.start:%Y-%m-%d}' AND dt <= DATE '{w.end:%Y-%m-%d}'
    )"""


def missing(
    con: duckdb.DuckDBPyConnection,
    w: window.Window,
    *,
    posts: str | None = None,
    embeddings: str | None = None,
):
    """`post_id`, `text` of the window's eligible posts with no vector.

    `posts` and `embeddings` are relation expressions, defaulting to the live
    R2 prefixes; the tests pass registered frames."""
    view = coord2.posts_view(posts or _window_posts(w))
    have = embeddings or embeddings_source(PLATFORM, semantic._slug(semantic.MODEL))
    rows = coord2.text_rows(
        con, view, missing_from=f"SELECT platform_post_id AS post_id FROM {have}"
    )
    return rows[["post_id", "text"]]


def embed_missing(
    con: duckdb.DuckDBPyConnection,
    w: window.Window,
    *,
    batch_size: int = 64,
    chunk: int = 5_000,
    posts: str | None = None,
    embeddings: str | None = None,
    uri: Callable[[str], str] | None = None,
) -> dict:
    """Embed and persist the window's missing vectors. Returns a summary with
    the count written, the run keys and the seconds spent finding and
    encoding."""
    started = time.monotonic()
    rows = missing(con, w, posts=posts, embeddings=embeddings)
    found_s = time.monotonic() - started
    log.info("window %s: %d eligible post(s) have no vector (%.0fs)", w.id, len(rows), found_s)
    keys: list[str] = []
    if len(rows):
        try:
            keys = semantic.embed_texts(
                con, rows["post_id"].astype(str).tolist(), rows["text"].tolist(),
                platform=PLATFORM, batch_size=batch_size, chunk=chunk, uri=uri,
            )
        finally:
            semantic.release_models()
    return {
        "window_id": w.id,
        "embedded": len(rows),
        "runs": keys,
        "find_s": round(found_s, 1),
        "encode_s": round(time.monotonic() - started - found_s, 1),
    }
