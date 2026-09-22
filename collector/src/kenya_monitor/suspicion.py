"""Heuristic suspicion ranking for collector-side target selection.

Mirrors the SQL heuristic in analysis `kma.authenticity` (no sklearn dep).

Every query here is bounded four ways, because both inputs grow without bound
and the collector runs under a 600MB DuckDB cap (`COLLECTOR_MEMORY_LIMIT`):

1. **Windowed.** Posts are pruned on `dt` before anything reads them.
2. **Projected.** Only the columns the score needs are carried; never `SELECT *`,
   which drags `text` and every author field through the operator.
3. **Spillable.** Hash aggregates only - no window functions over the corpus, no
   `count(DISTINCT ...)`.
4. **Staged.** Each aggregate is materialised before the next one reads it.

Point 4 is not belt and braces, it is the fix. Measured on pi0 at the real cap
2026-09-02, every stage fits on its own - post dedup 569s, duplicate-text
aggregate 261s, author dedup 1248s - but the single nested query they used to
form still OOMed, because DuckDB runs them concurrently and their peaks add.
Staged, the same work completes in 1444s. tf1 has enough headroom to survive the
pipelined form; pi0 does not, which is why this has to be measured on the host
that runs it.

Corpus at the time of that measurement: 2,165,609 post rows / 898,170 distinct
ids / 282,675 distinct authors, and 5,005,226 author rows / 1,195,614 distinct
users. Both seed paths come through here - `hate_signal` ranks on this table -
so one unbounded query took out hate seeding and its fallback together.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import duckdb

from kenya_monitor.config import SUSPICION_CACHE_MINUTES, SUSPICION_LOOKBACK_DAYS

WEIGHTS = {
    "ratio": 0.30,
    "age": 0.25,
    "rate": 0.15,
    "dup": 0.15,
    "bio": 0.05,
    "img": 0.05,
    "handle": 0.05,
}

# The scored profile fields, carried through the author dedup and nothing else.
_AUTHOR_COLS = (
    "handle",
    "created_at",
    "followers_count",
    "following_count",
    "tweet_count",
    "bio",
    "profile_image_url",
)

# When each (connection, inputs) last built its table, for the reuse check in
# `materialise`. Keyed on the connection so tests with their own connections
# never see another's table.
_BUILT: dict[tuple, float] = {}

_BEH = "_susp_beh"
_PROF = "_susp_prof"
SUSPICION_TABLE = "_susp_scored"


def _post_columns(con: duckdb.DuckDBPyConnection, posts_view: str) -> set[str]:
    return set(con.sql(f"SELECT * FROM {posts_view} LIMIT 0").columns)


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    try:
        con.sql(f"SELECT 1 FROM {table} LIMIT 0")
    except duckdb.Error:
        return False
    return True


def _struct(cols: tuple[str, ...]) -> str:
    return "struct_pack(" + ", ".join(f"{c} := {c}" for c in cols) + ")"


def _beh_sql(posts_view: str, window: str) -> str:
    """Per-author volume and duplicate-text ratio.

    Text never enters as text. It is hashed at the scan and only the hash is
    carried, because the ratio needs equality classes and nothing else: two
    posts are duplicates if their hashes match. That is an 8-byte grouping key
    against a whole post body, and the post-level relation is the widest thing
    this stage builds.

    The latest-snapshot dedup stays `arg_max(struct, collected_at)`, and the
    struct stays atomic: a post's text can differ between collections, so the
    author and the text have to come from the same winning row. Only its width
    changes, from a post body to a hash, and that is the whole fix - the
    aggregate state held one full body per distinct post, in a temp relation
    that counts against `memory_limit` and cannot spill.

    A NULL text hashes to NULL rather than to a value, so a post whose newest
    snapshot has no text is still dropped after the dedup, not before it.

    Measured on pi0 2026-09-21 against the live corpus, 30-day window, 600MB
    limit, 2 threads, file cache off for both: hashed completed in 1,249s over
    405,168 authors at a 564 MB peak RSS, nothing spilled; the same query
    carrying the text died at 1,232s on the production error - `failed to
    allocate data of size 256.0 KiB (572.1 MiB/572.2 MiB used)` - after
    spilling 148 MB. The arena holding the aggregate states was 106 MB hashed
    against 354 MB carrying text.

    Two aggregate stages instead of `count(DISTINCT ...)` - the inner group
    collapses repeats, the outer counts them - so no per-author hash set is
    built either.
    """
    return f"""
    SELECT author_id,
        sum(n) AS n_posts,
        1.0 - count(*) * 1.0 / sum(n) AS duplicate_text_ratio
    FROM (
        SELECT r.author_id AS author_id, r.t AS t, count(*) AS n
        FROM (
            SELECT platform, platform_post_id,
                   arg_max(struct_pack(
                       author_id := author_id,
                       t := CASE WHEN text IS NULL THEN NULL
                                 ELSE hash(lower(trim(text))) END
                   ), collected_at) AS r
            FROM {posts_view}
            {window}
            GROUP BY platform, platform_post_id
        )
        WHERE r.t IS NOT NULL
        GROUP BY 1, 2
    )
    WHERE author_id IS NOT NULL
    GROUP BY author_id
    """


def _prof_sql(authors_view: str, beh_table: str) -> str:
    """Latest profile per author, restricted to authors seen in the window.

    `arg_max(struct, collected_at)` replaces `QUALIFY row_number() OVER
    (PARTITION BY ... ORDER BY collected_at DESC)`: the same row wins, but as a
    hash aggregate rather than a sort over the corpus. The struct preserves
    whole-row semantics - unpacking column by column would let a NULL field on
    the newest row fall back to an older row's value, flipping `empty_bio`.

    The authors prefix holds every account ever collected (1.2M distinct against
    ~283k that appear in posts); an account that has posted nothing in the window
    cannot be a useful crawl seed anyway.
    """
    return f"""
    SELECT platform_user_id,
           arg_max({_struct(_AUTHOR_COLS)}, collected_at) AS r
    FROM {authors_view}
    WHERE platform_user_id IN (SELECT author_id FROM {beh_table})
    GROUP BY platform, platform_user_id
    """


def _score_sql(prof_table: str = _PROF, beh_table: str = _BEH) -> str:
    """The score itself, over two small materialised inputs."""
    w = WEIGHTS
    return f"""
    SELECT
        p.r.handle AS handle,
        -- Carried so a caller can rank by suspicion and still address accounts
        -- by id. `deep_timelines` needs that: its primary target set comes from
        -- v2 scores keyed on `user_id`, and its fallback ranking comes from
        -- here, so a handle-only fallback would key the ledger and the
        -- deepened-accounts record on a different column than the join that
        -- has to check them for the feedback artefact.
        p.platform_user_id AS user_id,
        greatest(date_diff('day', p.r.created_at, now()), 0) AS account_age_days,
        p.r.followers_count * 1.0 / greatest(p.r.following_count, 1) AS followers_following_ratio,
        p.r.tweet_count * 1.0 / greatest(date_diff('day', p.r.created_at, now()), 1) AS tweet_rate,
        (p.r.bio IS NULL OR p.r.bio = '') AS empty_bio,
        (p.r.profile_image_url ILIKE '%default_profile%') AS default_profile_image,
        len(regexp_replace(p.r.handle, '[^0-9]', '', 'g')) * 1.0
            / greatest(len(p.r.handle), 1) AS handle_digit_ratio,
        COALESCE(b.duplicate_text_ratio, 0.0) AS duplicate_text_ratio,
        {w['ratio']} * (1.0 / (1.0 + p.r.followers_count * 1.0 / greatest(p.r.following_count, 1)))
        + {w['age']} * greatest(0.0, 1.0 - greatest(date_diff('day', p.r.created_at, now()), 0) / 365.0)
        + {w['rate']} * least(1.0, p.r.tweet_count * 1.0
            / greatest(date_diff('day', p.r.created_at, now()), 1) / 50.0)
        + {w['dup']} * COALESCE(b.duplicate_text_ratio, 0.0)
        + {w['bio']} * (p.r.bio IS NULL OR p.r.bio = '')::INT
        + {w['img']} * (p.r.profile_image_url ILIKE '%default_profile%')::INT
        + {w['handle']} * (len(regexp_replace(p.r.handle, '[^0-9]', '', 'g')) * 1.0
            / greatest(len(p.r.handle), 1)) AS suspicion
    FROM {prof_table} p
    LEFT JOIN {beh_table} b ON p.platform_user_id = b.author_id
    WHERE p.r.handle IS NOT NULL AND trim(p.r.handle) != ''
    """


@contextmanager
def _no_file_cache(con: duckdb.DuckDBPyConnection):
    """Scan the corpus without keeping the bytes it read.

    DuckDB caches fetched file blocks inside `memory_limit`, which pays for
    itself when queries revisit the same files and does not here: these two
    stages read a 30-day glob once. Headroom, not the fix - the fix is the hash
    in `_beh_sql`. Measured on pi0 2026-09-21, 30-day window, 600MB limit, 2
    threads: uncached the stage completed in 1,249s at a 564 MB peak RSS, with
    DuckDB's own accounting peaking at 270 MB of its 572 MiB budget and nothing
    spilled. The same run cached was 100-150 MB higher at every sample and was
    stopped at 20 minutes rather than run to an answer, so the cost of the
    cache here is bounded but not exactly known.

    Scoped rather than set on the connection: other steps reread the same
    partitions within a cycle and the cache is worth having for them. Best
    effort, like `Storage._tune` - a setting that does not exist in this DuckDB
    build must not stop the ranking."""
    try:
        con.execute("SET enable_external_file_cache=false")
    except duckdb.Error:
        yield
        return
    try:
        yield
    finally:
        try:
            con.execute("RESET enable_external_file_cache")
        except duckdb.Error:
            pass


def materialise(
    con: duckdb.DuckDBPyConnection,
    authors_view: str,
    posts_view: str,
    lookback_days: int = SUSPICION_LOOKBACK_DAYS,
    table: str = SUSPICION_TABLE,
    max_age_minutes: int = SUSPICION_CACHE_MINUTES,
) -> str:
    """Build the scored table in stages and return its name.

    Callers rank off the returned table rather than nesting the query, so the
    stages cannot be fused back into one pipeline - see the module docstring for
    what that costs.

    Reused within `max_age_minutes` if the same connection already built it from
    the same inputs. A cycle calls both seed paths and each needs this table;
    rebuilding it for the second caller cost 1,508s on pi0 for a result that had
    not changed. Pass 0 to force a rebuild.
    """
    key = (id(con), table, authors_view, posts_view, lookback_days)
    built = _BUILT.get(key)
    if built is not None and max_age_minutes > 0:
        if (time.monotonic() - built) < max_age_minutes * 60 and _table_exists(con, table):
            return table
    cols = _post_columns(con, posts_view)
    # Callers register bare tables in tests, and `dt` only exists on the hive
    # partitioning of the real R2 views. Probed rather than assumed so an
    # unpartitioned view still ranks instead of failing to bind.
    window = (
        f"WHERE dt >= current_date - INTERVAL {int(lookback_days)} DAY"
        if "dt" in cols
        else ""
    )
    with _no_file_cache(con):
        con.execute(f"CREATE OR REPLACE TEMP TABLE {_BEH} AS {_beh_sql(posts_view, window)}")
        con.execute(f"CREATE OR REPLACE TEMP TABLE {_PROF} AS {_prof_sql(authors_view, _BEH)}")
    con.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS {_score_sql(_PROF, _BEH)}")
    # The two inputs are read once, by the score, and a temp table counts
    # against `memory_limit` for as long as it exists. Holding the profile
    # table - which carries a bio per author - for the rest of the cycle takes
    # that budget away from every later step on this connection.
    con.execute(f"DROP TABLE IF EXISTS {_PROF}")
    con.execute(f"DROP TABLE IF EXISTS {_BEH}")
    _BUILT[key] = time.monotonic()
    return table


def top_suspicious_handles(
    con: duckdb.DuckDBPyConnection,
    authors_view: str,
    posts_view: str,
    n: int = 1000,
    lookback_days: int = SUSPICION_LOOKBACK_DAYS,
) -> list[str]:
    """Return handles of the top `n` accounts by heuristic suspicion.

    Ranks only accounts that posted within `lookback_days`."""
    table = materialise(con, authors_view, posts_view, lookback_days)
    rows = con.sql(
        f"SELECT handle FROM {table} ORDER BY suspicion DESC NULLS LAST LIMIT {int(n)}"
    ).fetchall()
    return [r[0] for r in rows]


def top_suspicious_accounts(
    con: duckdb.DuckDBPyConnection,
    authors_view: str,
    posts_view: str,
    n: int = 1000,
    lookback_days: int = SUSPICION_LOOKBACK_DAYS,
) -> list[tuple[str, str]]:
    """(user id, handle) of the top `n` accounts by heuristic suspicion.

    The follow crawl's seed path. Handles alone made it resolve each seed back
    to an id with a scan of the whole authors prefix, for an id this table
    already holds."""
    table = materialise(con, authors_view, posts_view, lookback_days)
    rows = con.sql(
        f"SELECT user_id, handle FROM {table} ORDER BY suspicion DESC NULLS LAST LIMIT {int(n)}"
    ).fetchall()
    return [(str(uid), handle) for uid, handle in rows]
