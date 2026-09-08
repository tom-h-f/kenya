"""Verify X's information-operations archive before anything is built on it.

Step A2 of `docs/plans/2026-09-05-v2-next-steps.md`. Point it at a sample first
and the full file later; both take the same path or glob:

    uv run python -m kma.ioa_verify /vol/ioa_sample.csv
    uv run python -m kma.ioa_verify /vol/ioa_tweets.csv --users /vol/ioa_users.csv

This script never downloads anything. It reads a file that is already on disk.

Four checks, each of which changes the plan if it fails:

1. ID STABILITY. The Internet Archive mirror anonymises accounts under 5,000
   followers by hashing their id, and 97.5% of the 87,287 rows in
   `ioa_users.csv` are hashed. If those hashes were per-row rather than per-user,
   every account would appear exactly once and NO similarity network could be
   built - the whole method would collapse. Measured on a 4,470-row sample of
   the head of the file: 10 hashed users carry 4,468 rows, mean 446.8 tweets
   each, max 1,946. So the hashes are stable pseudonyms and networks are
   buildable. This check re-asserts that on whatever slice it is pointed at,
   because one sample of one campaign is not the whole 113.72 GB.
2. TEXT SURVIVAL. `tweet_text` survived at 100% in that sample. Without it the
   fused network is four traces, not five.
3. ENTITY SURVIVAL. `urls` on 89.6% of sampled rows, `hashtags` on 4.5%. Two
   more traces depend on these, and `hashtag_sequence` needs at least three tags
   in a row, so the share of rows carrying three is reported separately from the
   share carrying any.
4. CAMPAIGN ATTRIBUTION, and this is the open problem. Neither file carries a
   country or campaign column: the mirror is a plain concatenation of X's
   per-campaign exports and campaign identity was lost in the making of it. The
   paper evaluates per country. This check reports the EVIDENCE available for
   segmenting the file - embedded header rows, contiguous runs of
   `account_language`, contiguous `userid` blocks, `account_creation_date`
   spread - and deliberately does not guess a labelling from it.

   The consequence if campaigns cannot be attributed: per-country evaluation is
   impossible, and the fallback is the paper's Task 2, global classification
   over all drivers combined. That keeps a valid benchmark and loses the
   per-campaign comparison. It is not implemented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import duckdb

log = logging.getLogger(__name__)

TWEET_COLUMNS = (
    "tweetid",
    "userid",
    "account_creation_date",
    "account_language",
    "tweet_language",
    "tweet_text",
    "tweet_time",
    "is_retweet",
    "retweet_userid",
    "retweet_tweetid",
    "hashtags",
    "urls",
)

# Shape floors, not quality bars: they separate "the anonymisation left a usable
# corpus" from "it did not". A hashed id appearing once per row is the failure
# these numbers exist to catch.
MIN_MEDIAN_TWEETS_PER_ACCOUNT = 2.0
MAX_SINGLETON_SHARE = 0.5
MIN_TEXT_SHARE = 0.95


@dataclass(frozen=True)
class Check:
    """One verification result. `passed` is None for the campaign-attribution
    check, which reports evidence rather than a verdict."""

    name: str
    passed: bool | None
    note: str
    detail: dict = field(default_factory=dict)


def connect() -> duckdb.DuckDBPyConnection:
    """A local DuckDB with file order preserved, which the campaign-attribution
    check depends on: contiguity is only meaningful in the file's own order."""
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=true")
    return con


def source(path: str, *, limit: int | None = None) -> str:
    """The archive CSV as a relation.

    `all_varchar` because the mirror carries embedded header rows mid-file,
    which break type sniffing outright.

    `ignore_errors` because this tool exists to be pointed at a SLICE of a
    113.72 GB remote CSV, and any byte-range slice ends mid-row. It also has to
    survive the real thing: tweet text legitimately contains embedded newlines
    inside quotes - a Bengali-language row at line 4,472 of the head of
    `ioa_tweets.csv` is one - so a strict read fails on well-formed archive data
    as readily as on a truncated sample. `row_count` reports what was actually
    parsed so a silently halved slice cannot pass for a whole one.
    """
    read = (
        f"read_csv('{path}', all_varchar=true, union_by_name=true, "
        "header=true, ignore_errors=true)"
    )
    tail = f" LIMIT {int(limit)}" if limit else ""
    return f"(SELECT * FROM {read}{tail})"


def _numeric_id(column: str = "userid") -> str:
    """Anonymised accounts carry a hash where a public account carries digits."""
    return f"try_cast({column} AS BIGINT) IS NOT NULL"


def check_id_stability(con: duckdb.DuckDBPyConnection, src: str) -> Check:
    """Are hashed ids stable pseudonyms, or per-row hashes?"""
    rows = con.sql(
        f"""
        WITH per_account AS (
            SELECT userid, {_numeric_id()} AS public, count(*) AS tweets
            FROM {src}
            WHERE userid IS NOT NULL AND userid <> 'userid'
            GROUP BY userid, public
        )
        SELECT public,
               count(*) AS accounts,
               sum(tweets) AS tweets,
               avg(tweets) AS mean_tweets,
               median(tweets) AS median_tweets,
               max(tweets) AS max_tweets,
               count(*) FILTER (tweets = 1) / count(*) AS singleton_share
        FROM per_account
        GROUP BY public
        """
    ).df()

    detail = {
        ("public" if bool(r.public) else "hashed"): {
            "accounts": int(r.accounts),
            "tweets": int(r.tweets),
            "mean_tweets": float(r.mean_tweets),
            "median_tweets": float(r.median_tweets),
            "max_tweets": int(r.max_tweets),
            "singleton_share": float(r.singleton_share),
        }
        for r in rows.itertuples()
    }
    hashed = detail.get("hashed")
    if hashed is None:
        return Check(
            "id_stability",
            None,
            "no anonymised (hashed) ids in this slice, so nothing to verify here",
            detail,
        )
    passed = (
        hashed["median_tweets"] >= MIN_MEDIAN_TWEETS_PER_ACCOUNT
        and hashed["singleton_share"] < MAX_SINGLETON_SHARE
    )
    note = (
        f"hashed ids are stable pseudonyms: {hashed['accounts']} accounts carry "
        f"{hashed['tweets']} tweets, median {hashed['median_tweets']:.1f} each"
        if passed
        else f"hashed ids look per-row: {hashed['singleton_share']:.1%} appear exactly once, "
        "so no similarity network can be built from them"
    )
    return Check("id_stability", passed, note, detail)


def check_text_survival(con: duckdb.DuckDBPyConnection, src: str) -> Check:
    """Does tweet text survive anonymisation? `text_similarity` needs it."""
    row = con.sql(
        f"""
        SELECT count(*) AS rows,
               count(*) FILTER (coalesce(trim(tweet_text), '') <> '') / count(*) AS text_share,
               count(*) FILTER ({_numeric_id()} IS false
                                AND coalesce(trim(tweet_text), '') <> '')
                   / nullif(count(*) FILTER ({_numeric_id()} IS false), 0) AS hashed_text_share,
               count(DISTINCT tweet_text) AS distinct_texts,
               avg(len(str_split(trim(coalesce(tweet_text, '')), ' '))) AS mean_words
        FROM {src}
        WHERE userid IS NOT NULL AND userid <> 'userid'
        """
    ).df().iloc[0]

    detail = {
        "rows": int(row["rows"]),
        "text_share": float(row["text_share"]),
        # Reported separately: anonymisation could plausibly strip text only for
        # the accounts it anonymised, which an overall share would hide.
        "hashed_text_share": None
        if row["hashed_text_share"] is None
        else float(row["hashed_text_share"]),
        "distinct_texts": int(row["distinct_texts"]),
        "mean_words": float(row["mean_words"]),
    }
    passed = detail["text_share"] >= MIN_TEXT_SHARE and (
        detail["hashed_text_share"] is None or detail["hashed_text_share"] >= MIN_TEXT_SHARE
    )
    note = (
        f"text present on {detail['text_share']:.1%} of rows, mean "
        f"{detail['mean_words']:.1f} words"
        if passed
        else "tweet text is missing or stripped; the text-similarity trace is unavailable"
    )
    return Check("text_survival", passed, note, detail)


def check_entity_survival(con: duckdb.DuckDBPyConnection, src: str) -> Check:
    """Do URLs and hashtags survive? `co_url` and `hashtag_sequence` need them."""
    empty = "IN ('', '[]')"
    row = con.sql(
        f"""
        SELECT count(*) AS rows,
               count(*) FILTER (coalesce(trim(urls), '') NOT {empty}) / count(*) AS url_share,
               count(*) FILTER (coalesce(trim(hashtags), '') NOT {empty}) / count(*)
                   AS hashtag_share,
               count(*) FILTER (
                   len(str_split(regexp_replace(coalesce(hashtags, ''), '^\\[|\\]$', '', 'g'),
                                 ', ')) >= 3
                   AND coalesce(trim(hashtags), '') NOT {empty}
               ) / count(*) AS three_hashtag_share,
               count(*) FILTER (coalesce(trim(retweet_tweetid), '') <> '') / count(*)
                   AS retweet_share
        FROM {src}
        WHERE userid IS NOT NULL AND userid <> 'userid'
        """
    ).df().iloc[0]

    detail = {k: float(row[k]) for k in ("url_share", "hashtag_share", "three_hashtag_share", "retweet_share")}
    detail["rows"] = int(row["rows"])
    passed = detail["url_share"] > 0 and detail["three_hashtag_share"] > 0
    note = (
        f"urls on {detail['url_share']:.1%} of rows, hashtags on "
        f"{detail['hashtag_share']:.1%}, three or more hashtags on "
        f"{detail['three_hashtag_share']:.1%}, retweets {detail['retweet_share']:.1%}"
    )
    if not passed:
        note += " - a trace with a zero share cannot be built at all"
    return Check("entity_survival", passed, note, detail)


def _runs(con: duckdb.DuckDBPyConnection, src: str, column: str) -> tuple[int, int]:
    """Contiguous runs of a column's value in file order, and its distinct count.

    Runs close to distinct values means the file is ordered in blocks by that
    column; runs far above it means the values are interleaved. This is the
    islands half of the gaps-and-islands query.
    """
    row = con.sql(
        f"""
        WITH ordered AS (
            SELECT row_number() OVER () AS rn, {column} AS value
            FROM {src}
            WHERE userid <> 'userid' AND coalesce({column}, '') <> ''
        ),
        edges AS (
            SELECT value, lag(value) OVER (ORDER BY rn) AS previous FROM ordered
        )
        SELECT count(*) FILTER (previous IS NULL OR value IS DISTINCT FROM previous) AS runs,
               count(DISTINCT value) AS distinct_values
        FROM edges
        """
    ).df().iloc[0]
    return int(row["runs"]), int(row["distinct_values"])


def check_campaign_attribution(con: duckdb.DuckDBPyConnection, src: str) -> Check:
    """What evidence exists for segmenting the concatenation into campaigns?

    Reports evidence only. Guessing a labelling here would put a fabricated
    country column under every per-campaign number in the benchmark, which is
    worse than not having one.
    """
    headers = con.sql(
        f"""
        WITH ordered AS (SELECT row_number() OVER () AS rn, userid, tweetid FROM {src})
        SELECT rn FROM ordered WHERE userid = 'userid' OR tweetid = 'tweetid' ORDER BY rn
        """
    ).df()["rn"].tolist()

    language_runs, languages = _runs(con, src, "account_language")
    user_runs, users = _runs(con, src, "userid")
    creation = con.sql(
        f"""
        SELECT min(account_creation_date) AS earliest,
               max(account_creation_date) AS latest,
               count(DISTINCT account_creation_date) AS distinct_dates
        FROM {src}
        WHERE userid <> 'userid' AND coalesce(account_creation_date, '') <> ''
        """
    ).df().iloc[0]

    detail = {
        "embedded_header_rows": [int(r) for r in headers],
        "account_language_runs": language_runs,
        "account_languages": languages,
        "userid_runs": user_runs,
        "userids": users,
        "account_created_earliest": str(creation["earliest"]),
        "account_created_latest": str(creation["latest"]),
        "distinct_creation_dates": int(creation["distinct_dates"]),
    }
    contiguous = languages > 0 and language_runs <= languages * 2
    evidence = [
        f"{len(headers)} embedded header row(s)"
        + (f" at {headers}" if headers else " - no export boundary is marked in the data"),
        f"account_language: {language_runs} contiguous runs over {languages} distinct values"
        + (" - blocked, so campaign boundaries plausibly align with them" if contiguous
           else " - interleaved, so language does not segment the file"),
        f"userid: {user_runs} contiguous runs over {users} accounts"
        + (" - accounts are blocked" if user_runs <= users * 2 else " - accounts are interleaved"),
        f"account_creation_date spans {creation['earliest']} to {creation['latest']}",
    ]
    return Check(
        "campaign_attribution",
        None,
        "; ".join(evidence),
        detail,
    )


def verify(
    path: str,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    limit: int | None = None,
    src: str | None = None,
) -> list[Check]:
    """Run all four checks against a path, a glob, or an injected relation."""
    con = con or connect()
    src = src or source(path, limit=limit)
    return [
        check_id_stability(con, src),
        check_text_survival(con, src),
        check_entity_survival(con, src),
        check_campaign_attribution(con, src),
    ]


def report(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        status = "EVIDENCE" if check.passed is None else ("PASS" if check.passed else "FAIL")
        lines.append(f"[{status}] {check.name}: {check.note}")
        for key, value in check.detail.items():
            lines.append(f"    {key}: {value}")
    failed = [c.name for c in checks if c.passed is False]
    lines.append("")
    lines.append(
        f"{len(failed)} check(s) failed: {', '.join(failed)}"
        if failed
        else "no check failed; the archive supports the traces it was asked about"
    )
    return "\n".join(lines)


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="path or glob to ioa_tweets.csv, or a sample of it")
    ap.add_argument("--limit", type=int, default=None, help="read only the first N rows")
    args = ap.parse_args()

    checks = verify(args.path, limit=args.limit)
    print(report(checks))
    return 1 if any(c.passed is False for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
