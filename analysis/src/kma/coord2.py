"""Coordination detection v2: Luceri, Pante, Burghardt and Ferrara (WWW 2024).

A faithful implementation of arXiv:2310.09884, "Unmasking the Web of Deceit".
The method and the reasons for adopting it wholesale are in
`docs/plans/2026-09-05-v2-methodology.md`; this module is step A4 of
`docs/plans/2026-09-05-v2-next-steps.md`.

    from kma.db import connect
    from kma import coord2
    con = connect()
    view = coord2.posts_view()
    nets = {
        "co_retweet": coord2.similarity_network(coord2.co_retweet_traces(con, view),
                                                percentile=coord2.EDGE_PERCENTILE["co_retweet"]),
        "co_url": coord2.similarity_network(coord2.co_url_traces(con, view),
                                            percentile=coord2.EDGE_PERCENTILE["co_url"]),
    }
    found = coord2.detect(nets)                      # unsupervised: centrality + pruning
    emb = coord2.node2vec(coord2.fuse(nets))         # supervised: embedding + RF
    metrics, oof = coord2.classify(emb, labels)

The unit of prediction is the ACCOUNT. There is no significance null, no
multiple-testing correction, no hub cap and no community detection anywhere in
the detection path - `kma.coordination` (v1) has all four and this module
deliberately has none of them. Community detection, if wanted, is a post-hoc
description of accounts already classified.

Four traces share one recipe: a bipartite user x entity graph, TF-IDF weighted
over entity popularity, projected to a user x user network whose edge weight is
the cosine similarity of the users' TF-IDF vectors. `text_similarity` is the
exception and has no bipartite stage.

HASHTAG ORDER, the open question from A4 step 1, answered by reading the write
path rather than by assuming:

    X GraphQL `legacy.entities.hashtags`  (array of {text, indices})
      -> twscrape `Tweet.parse`: `[x["text"] for x in ...entities.hashtags]`
      -> collector `x.py`: `hashtags=list(tw.hashtags or [])`
      -> `storage.POST_SCHEMA`: `list<string>`

Nothing in that chain sorts, dedupes or re-orders, so the list is in whatever
order X returned, which in practice is ascending `indices`, i.e. in-tweet order.
It is NOT a set: a repeated hashtag appears repeatedly. Three caveats that stop
this being a guarantee, and they are stated plainly rather than papered over:

  1. X documents no ordering contract for entity arrays, and twscrape DISCARDS
     `indices`, so the order cannot be verified or repaired after the fact from
     what we persist.
  2. For long-form posts, `rawContent` comes from `note_tweet` while `hashtags`
     still comes from the truncated `legacy` entity set, so the sequence can be
     short of what the text contains.
  3. It has not been checked against live rows: a git worktree carries no R2
     credentials, and no local fixture holds both `text` and `hashtags`.

`hashtag_sequence_traces(..., order="text")` therefore exists: it re-derives the
sequence from the post text, where the order is the text's own and the note_tweet
truncation does not apply. Prefer it wherever `text` is populated.

Ambiguities in the paper, and what was chosen. These are choices, not findings:

  - Table 2 gives a percentile for co-retweet, co-URL, hashtag sequence and text
    similarity but only a time interval (60s) for fast retweet, so
    `EDGE_PERCENTILE["fast_retweet"]` is None: that network is not edge-filtered.
  - The percentiles are quoted for the RQ1 edge-filtering evaluation. Whether
    node pruning (RQ2) runs on filtered or unfiltered networks is not stated.
    `percentile=None` runs the unfiltered variant, so both readings are testable.
  - "80th percentile of cosine similarity" is taken over the REALISED edge
    weights - the non-zero cosines - not over all user pairs, most of which are
    zero and would drag any high percentile down to zero.
  - The text-similarity percentile is over TWEET-PAIR similarities, not over
    user-user edge weights: the paper thresholds pairs, then links users who
    have at least one surviving pair. See `text_similarity_network` for why the
    population that percentile is taken over is itself unresolved.
  - Thresholds are PERCENTILES, never constants. The paper's 0.95 is the 96th
    percentile of ITS embedding space (`stsb-xlm-r-multilingual`); this project
    uses `paraphrase-multilingual-mpnet-base-v2` by the Q2 decision, and a
    cosine value does not port across embedding spaces.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from kma.db import posts_source

log = logging.getLogger(__name__)

TRACES = ("co_retweet", "co_url", "hashtag_sequence", "fast_retweet", "text_similarity")

# The paper's optimized parameters (Table 2), which beat the values prior work
# used on every trace. Copied, not re-derived: they are re-tuned only once the
# benchmark reproduces, per the A5 gate.
EDGE_PERCENTILE: dict[str, float | None] = {
    "co_retweet": 80.0,
    "co_url": 80.0,
    "hashtag_sequence": 65.0,
    # Table 2 sweeps the time interval for this trace, not a similarity
    # percentile, so there is no edge filter to apply.
    "fast_retweet": None,
    # Over tweet-pair similarities, not over user edge weights.
    "text_similarity": 96.0,
}

FAST_RETWEET_SECONDS = 60
MIN_HASHTAGS = 3
MIN_TEXT_WORDS = 4
TEXT_WINDOW_DAYS = 365

# The paper's conservative operating point: precision > 99% at average AUC
# > 0.70. Optimising it is an A5 decision, taken against the benchmark.
CENTRALITY_THRESHOLD = 1e-2

NODE2VEC_DIM = 128
NODE2VEC_WALKS = 16
NODE2VEC_LENGTH = 16


# --------------------------------------------------------------------------
# Canonical views. Every extractor below reads these column names, so the
# Kenya corpus and the IO archive differ in one adapter rather than in five
# queries.
# --------------------------------------------------------------------------

VIEW_COLUMNS = (
    "user_id",
    "post_id",
    "created_at",
    "is_retweet",
    "retweet_post_id",
    "retweet_user_id",
    "text",
    "hashtags",
    "urls",
)


def posts_view(source: str | None = None, *, latest: bool = True) -> str:
    """The Kenya `posts` prefix as the canonical view.

    `source` is any relation expression, so a caller can pass a pinned
    `kma.bench.pinned_source(...)`, a local parquet, or a registered frame; it
    defaults to the live posts glob. `retweet_user_id` is NULL because the
    collector never stored the retweeted author - `fast_retweet_traces`
    recovers it by joining to the original post, and loses the retweets whose
    original is not in the corpus.
    """
    src = source or posts_source()
    dedupe = (
        "QUALIFY row_number() OVER (PARTITION BY platform_post_id "
        "ORDER BY collected_at DESC) = 1"
        if latest
        else ""
    )
    return f"""
        SELECT author_id AS user_id,
               platform_post_id AS post_id,
               created_at,
               coalesce(is_repost, repost_of_id IS NOT NULL) AS is_retweet,
               repost_of_id AS retweet_post_id,
               CAST(NULL AS VARCHAR) AS retweet_user_id,
               text,
               hashtags,
               urls
        FROM {src}
        {dedupe}
    """


def _bracket_list(column: str) -> str:
    """The archive writes list columns as a Python list literal, e.g.
    `['WorldCup2018', 'CR7']`, and `[]` when empty. The element quotes are part
    of the literal, not of the value."""
    inner = f"regexp_replace({column}, '^\\[|\\]$', '', 'g')"
    return (
        f"list_filter(list_transform(str_split({inner}, ', '),"
        " x -> regexp_replace(trim(x), '^[''\"]|[''\"]$', '', 'g')),"
        " x -> length(x) > 0)"
    )


def ioa_view(path: str, *, read: str | None = None) -> str:
    """X's information-operations archive (`ioa_tweets.csv`) as the canonical view.

    Verify the file with `kma.ioa_verify` before building anything on it: the
    mirror anonymises accounts under 5,000 followers, and whether their ids are
    stable pseudonyms decides whether any similarity network can be built at all.
    """
    # `ignore_errors` for the same reason `kma.ioa_verify.source` needs it: this
    # reads a SLICE of a 113.72 GB remote CSV, and every byte-range slice ends
    # mid-row. The full file needs it too - tweet text legitimately contains
    # newlines inside quotes, so a strict read fails on well-formed archive data.
    # Without it the adapter raises before a single trace is built.
    src = read or (
        f"read_csv('{path}', union_by_name=true, all_varchar=true, ignore_errors=true)"
    )
    return f"""
        SELECT userid AS user_id,
               tweetid AS post_id,
               try_cast(tweet_time AS TIMESTAMP) AS created_at,
               lower(coalesce(is_retweet, 'false')) IN ('true', '1') AS is_retweet,
               nullif(retweet_tweetid, '') AS retweet_post_id,
               nullif(retweet_userid, '') AS retweet_user_id,
               tweet_text AS text,
               {_bracket_list('coalesce(hashtags, %s)' % "''")} AS hashtags,
               {_bracket_list('coalesce(urls, %s)' % "''")} AS urls
        FROM {src}
    """


# --------------------------------------------------------------------------
# The shared recipe: bipartite user x entity -> TF-IDF -> cosine projection.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Bipartite:
    """A user x entity TF-IDF matrix with L2-normalised rows, so a row dot
    product IS the cosine similarity of two users' vectors."""

    users: np.ndarray
    entities: np.ndarray
    matrix: sparse.csr_matrix


def bipartite_tfidf(
    traces: pd.DataFrame, *, user_col: str = "user_id", entity_col: str = "entity"
) -> Bipartite:
    """TF-IDF over entity popularity, one row per user.

    TF is the number of times the user acted on the entity; IDF is sklearn's
    smoothed `ln((1 + n_users) / (1 + n_users_with_entity)) + 1`. TF-IDF is the
    paper's mechanism for handling popular entities - they are DOWN-WEIGHTED,
    never deleted, which is why v2 has no hub cap.
    """
    from sklearn.feature_extraction.text import TfidfTransformer

    if traces.empty:
        return Bipartite(
            np.array([], dtype=object),
            np.array([], dtype=object),
            sparse.csr_matrix((0, 0), dtype=np.float64),
        )

    users, u_idx = np.unique(traces[user_col].to_numpy(), return_inverse=True)
    entities, e_idx = np.unique(traces[entity_col].to_numpy(), return_inverse=True)
    # coo -> csr sums duplicate (user, entity) pairs, which is the term frequency.
    counts = sparse.coo_matrix(
        (np.ones(len(traces)), (u_idx, e_idx)), shape=(len(users), len(entities))
    ).tocsr()
    matrix = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=False).fit_transform(counts)
    return Bipartite(users, entities, matrix)


def cosine_edges(
    bp: Bipartite, *, chunk: int = 2048, min_weight: float = 1e-12
) -> pd.DataFrame:
    """Project to user x user: every pair with a non-zero cosine similarity.

    Chunked over rows and kept sparse throughout, so peak memory tracks the
    number of co-acting pairs rather than the square of the user count.
    """
    n = bp.matrix.shape[0]
    if n < 2:
        return pd.DataFrame({"source": [], "target": [], "weight": []}, dtype=object).astype(
            {"weight": "float64"}
        )

    blocks = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = (bp.matrix[start:end] @ bp.matrix.T).tocoo()
        rows = block.row + start
        keep = (block.col > rows) & (block.data > min_weight)
        if not keep.any():
            continue
        blocks.append(
            pd.DataFrame(
                {
                    "source": bp.users[rows[keep]],
                    "target": bp.users[block.col[keep]],
                    "weight": block.data[keep].astype("float64"),
                }
            )
        )
    if not blocks:
        return pd.DataFrame({"source": [], "target": [], "weight": []}, dtype=object).astype(
            {"weight": "float64"}
        )
    return pd.concat(blocks, ignore_index=True)


def filter_percentile(edges: pd.DataFrame, percentile: float | None) -> pd.DataFrame:
    """Keep edges at or above the given percentile of the realised weights.

    The percentile is taken over the edges that exist, not over all user pairs:
    the projection is sparse, so most pairs have weight zero and any high
    percentile over the full pair space would collapse to zero and filter
    nothing.
    """
    if percentile is None or edges.empty:
        return edges.reset_index(drop=True)
    threshold = float(np.percentile(edges["weight"].to_numpy(), percentile))
    return edges[edges["weight"] >= threshold].reset_index(drop=True)


# A user acting on a single entity cannot express coordination: two users whose
# only action is the same object have IDENTICAL one-hot TF-IDF vectors, so their
# cosine is 1.0 however hard IDF downweights that object - the normalisation
# cancels it. Measured on the Kenya snapshot 2026-09-07: 54.8% of co_retweet
# users had exactly one entity, one viral tweet drew 308 of them into a clique,
# and eigenvector centrality collapsed onto it - the top 100 accounts shared a
# single centrality value of 1/sqrt(308). At a floor of 2 the top 100 carried 90
# distinct values and the graph consolidated from 1,828 components to 245.
#
# This is a DEVIATION from Luceri et al., forced by data rather than chosen. It
# is also what v1's hub cap defended against, and what CooRTweet spells
# min_repetition.
MIN_ENTITIES_PER_USER = 2


def min_activity(traces: pd.DataFrame, minimum: int = MIN_ENTITIES_PER_USER) -> pd.DataFrame:
    """Drop users acting on fewer than `minimum` distinct entities."""
    if traces.empty or minimum <= 1:
        return traces
    counts = traces.groupby("user_id")["entity"].nunique()
    return traces[traces["user_id"].isin(counts[counts >= minimum].index)]


def similarity_network(
    traces: pd.DataFrame,
    *,
    percentile: float | None = None,
    chunk: int = 2048,
    min_entities: int = MIN_ENTITIES_PER_USER,
) -> pd.DataFrame:
    """One behavioural trace, from trace rows to a filtered similarity network."""
    kept = min_activity(traces, min_entities)
    if kept.empty:
        return pd.DataFrame({"source": [], "target": [], "weight": []})
    return filter_percentile(cosine_edges(bipartite_tfidf(kept), chunk=chunk), percentile)


# --------------------------------------------------------------------------
# Entity extractors. One per trace; each returns (user_id, entity) rows.
# --------------------------------------------------------------------------


def co_retweet_traces(con: duckdb.DuckDBPyConnection, view: str) -> pd.DataFrame:
    """Entity: the retweeted tweet."""
    return con.sql(
        f"""
        SELECT user_id, retweet_post_id AS entity
        FROM ({view})
        WHERE retweet_post_id IS NOT NULL AND user_id IS NOT NULL
        """
    ).df()


def co_url_traces(con: duckdb.DuckDBPyConnection, view: str) -> pd.DataFrame:
    """Entity: the URL as shared.

    Shorteners are NOT resolved. The paper's entity is the URL as it appears in
    the tweet, and on this corpus the question is close to moot: the collector
    stores twscrape's `TextLink.url`, which is X's own `expanded_url`, so t.co
    is already unwrapped one hop by the platform. What remains unresolved is
    third-party shorteners (bit.ly and friends), where two users sharing the
    same destination under different shortlinks are correctly - by the paper's
    definition - not co-sharing.
    """
    return con.sql(
        f"""
        SELECT user_id, unnest(urls) AS entity
        FROM ({view})
        WHERE urls IS NOT NULL AND len(urls) > 0 AND user_id IS NOT NULL
        """
    ).df()


_HASHTAG = re.compile(r"#(\w+)", re.UNICODE)


def hashtags_from_text(text: object) -> list[str]:
    """Hashtags in the order they appear in the text, duplicates kept."""
    if not isinstance(text, str):
        return []
    return _HASHTAG.findall(text)


def hashtag_sequence_traces(
    con: duckdb.DuckDBPyConnection,
    view: str,
    *,
    min_hashtags: int = MIN_HASHTAGS,
    order: str = "list",
) -> pd.DataFrame:
    """Entity: the ordered hashtag sequence, at least `min_hashtags` long.

    `order="list"` trusts the persisted `hashtags` column; `order="text"`
    re-derives the sequence from the post text. See the module docstring for
    why the second exists and when to prefer it.

    Case is folded, because `#Ruto` and `#ruto` are the same coordination
    entity and X's own hashtag matching is case-insensitive. The paper does not
    say, so this is a choice.
    """
    if order not in ("list", "text"):
        raise ValueError(f"order must be 'list' or 'text', got {order!r}")

    if order == "list":
        return con.sql(
            f"""
            SELECT user_id,
                   list_aggregate(list_transform(hashtags, x -> lower(x)), 'string_agg', '|')
                       AS entity
            FROM ({view})
            WHERE hashtags IS NOT NULL AND len(hashtags) >= {int(min_hashtags)}
              AND user_id IS NOT NULL
            """
        ).df()

    rows = con.sql(f"SELECT user_id, text FROM ({view}) WHERE user_id IS NOT NULL").df()
    tags = [hashtags_from_text(t) for t in rows["text"]]
    keep = [i for i, t in enumerate(tags) if len(t) >= min_hashtags]
    return pd.DataFrame(
        {
            "user_id": rows["user_id"].to_numpy()[keep],
            "entity": ["|".join(tag.lower() for tag in tags[i]) for i in keep],
        }
    )


def fast_retweet_traces(
    con: duckdb.DuckDBPyConnection,
    view: str,
    *,
    seconds: int = FAST_RETWEET_SECONDS,
) -> pd.DataFrame:
    """Entity: the retweeted AUTHOR, for retweets landing within `seconds`.

    "Fast" is measured against the ORIGINAL tweet's creation time, so the
    original has to be in the corpus even when the retweeted author is known
    from the retweet row. That is a coverage limit, not a rounding error:
    `engagements/` records retweeter incidence with no timestamp at all, and in
    the IO archive an operation retweeting an organic account leaves no original
    to join to. Report `fast_retweet_coverage` alongside any result from this
    trace.
    """
    return con.sql(
        f"""
        WITH v AS ({view})
        SELECT rt.user_id,
               coalesce(rt.retweet_user_id, orig.user_id) AS entity
        FROM v rt
        JOIN v orig ON orig.post_id = rt.retweet_post_id
        WHERE rt.retweet_post_id IS NOT NULL
          AND rt.user_id IS NOT NULL
          AND coalesce(rt.retweet_user_id, orig.user_id) IS NOT NULL
          AND date_diff('second', orig.created_at, rt.created_at) BETWEEN 0 AND {int(seconds)}
        """
    ).df()


def fast_retweet_coverage(
    con: duckdb.DuckDBPyConnection, view: str, *, seconds: int = FAST_RETWEET_SECONDS
) -> pd.DataFrame:
    """Retweets, how many could be timed at all, and how many were fast."""
    return con.sql(
        f"""
        WITH v AS ({view})
        SELECT count(*) AS retweets,
               count(orig.post_id) AS timeable,
               count(*) FILTER (
                   date_diff('second', orig.created_at, rt.created_at) BETWEEN 0 AND {int(seconds)}
               ) AS fast
        FROM v rt
        LEFT JOIN v orig ON orig.post_id = rt.retweet_post_id
        WHERE rt.retweet_post_id IS NOT NULL
        """
    ).df()


# --------------------------------------------------------------------------
# Text similarity: the trace with no bipartite stage.
# --------------------------------------------------------------------------

_URL = re.compile(r"https?://\S+|www\.\S+", re.UNICODE)
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0000fe00-\U0000fe0f\U0001f1e6-\U0001f1ff]+"
)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def clean_text(text: object, stopwords: frozenset[str] | None = None) -> str:
    """URLs, emoji, punctuation and stopwords removed, per the paper's recipe.

    The stopword list defaults to `kma.semantic.STOPWORDS`, which carries
    Swahili and Sheng function words the English lists miss - this corpus is
    not monolingual and an English-only list leaves `na`, `ya`, `wa` in every
    sentence.
    """
    if not isinstance(text, str):
        return ""
    if stopwords is None:
        stopwords = _default_stopwords()
    stripped = _PUNCT.sub(" ", _EMOJI.sub(" ", _URL.sub(" ", text))).lower()
    return " ".join(w for w in stripped.split() if w not in stopwords)


_STOPWORDS: frozenset[str] | None = None


def _default_stopwords() -> frozenset[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        from kma.semantic import STOPWORDS

        _STOPWORDS = frozenset(STOPWORDS)
    return _STOPWORDS


def text_rows(
    con: duckdb.DuckDBPyConnection,
    view: str,
    *,
    min_words: int = MIN_TEXT_WORDS,
    stopwords: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Non-retweets whose cleaned text carries at least `min_words` words.

    Retweets are excluded because a retweet is not the retweeter's own text;
    short texts because the paper found them noise rather than signal.
    """
    df = con.sql(
        f"""
        SELECT post_id, user_id, created_at, text
        FROM ({view})
        WHERE NOT coalesce(is_retweet, false) AND text IS NOT NULL AND user_id IS NOT NULL
        -- Ordered so a bounded read is the SAME bounded read next time. Without
        -- it, three runs over one pinned snapshot returned 16,478, 15,085 and
        -- 10,883 rows for an identical limit, which silently made a threshold
        -- sweep a comparison of different corpora.
        ORDER BY post_id
        """
    ).df()
    if df.empty:
        return df.assign(clean=pd.Series(dtype="object"))
    df = df.assign(clean=[clean_text(t, stopwords) for t in df["text"]])
    return df[df["clean"].str.split().str.len() >= min_words].reset_index(drop=True)


def epoch_seconds(when: pd.Series) -> np.ndarray:
    """Timestamps as float seconds since the epoch.

    Explicitly, rather than by casting to int64: DuckDB hands back microsecond
    datetimes and pandas hands back nanosecond ones, so a hard-coded divisor
    silently scales the sliding window by 1,000 depending on where the frame
    came from.
    """
    stamps = pd.to_datetime(when, utc=True)
    return (stamps - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


class PairSimilarity(Protocol):
    """The similarity search the text trace needs, narrow enough that a FAISS
    index could implement it without touching the caller.

    Yields `(i, j, similarity)` index blocks with `j > i`, restricted to pairs
    inside the sliding window. FAISS is not a dependency of this project, so
    the shipped implementation is a chunked exact cosine.
    """

    def __call__(
        self,
        vectors: np.ndarray,
        times: np.ndarray | None = None,
        *,
        window_seconds: float | None = None,
        chunk: int = 512,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]: ...


def cosine_pairs(
    vectors: np.ndarray,
    times: np.ndarray | None = None,
    *,
    window_seconds: float | None = None,
    chunk: int = 512,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Exact cosine similarity of every in-window pair, a row block at a time.

    `times` must be sorted ascending, which lets the window bound the candidate
    columns by a binary search instead of a mask over the whole matrix. Peak
    memory is `chunk` x (candidates in window), so `chunk` is the knob when the
    window spans everything.
    """
    n = len(vectors)
    if n < 2:
        return
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1.0, norms)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        if times is not None and window_seconds is not None:
            stop = int(np.searchsorted(times, times[end - 1] + window_seconds, side="right"))
        else:
            stop = n
        if stop <= start + 1:
            continue
        block = unit[start:end] @ unit[start:stop].T
        rows = np.arange(start, end)[:, None]
        cols = np.arange(start, stop)[None, :]
        keep = cols > rows
        if times is not None and window_seconds is not None:
            keep &= (times[cols] - times[rows]) <= window_seconds
        i, j = np.nonzero(keep)
        if len(i) == 0:
            continue
        yield rows[i, 0], cols[0, j], block[i, j]


def gpu_cosine_pairs(min_similarity: float, device: str | None = None) -> PairSimilarity:
    """A `PairSimilarity` that runs on GPU and yields ONLY pairs at or above
    `min_similarity`.

    `cosine_pairs` streams every in-window pair for the caller to filter, which
    is right at benchmark scale and impossible at corpus scale: 411k eligible
    Kenyan posts is 8.4e10 upper-triangle pairs, and materialising their indices
    exhausts memory long before any of them are thresholded. Pushing the
    threshold into the block keeps only survivors, which at a 0.9+ cut is a tiny
    fraction of the block.

    The threshold must therefore be known BEFORE this runs, so estimate it with
    `pair_similarity_percentile` on a sample and pass the result here.
    """
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def pairs(vectors, times=None, *, window_seconds=None, chunk=512):
        n = len(vectors)
        if n < 2:
            return
        mat = torch.as_tensor(np.asarray(vectors), dtype=torch.float32, device=dev)
        mat = mat / mat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        t = None if times is None else torch.as_tensor(np.asarray(times), dtype=torch.float64, device=dev)

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            stop = n
            if t is not None and window_seconds is not None:
                stop = int(np.searchsorted(times, times[end - 1] + window_seconds, side="right"))
            if stop <= start + 1:
                continue
            block = mat[start:end] @ mat[start:stop].T
            rows = torch.arange(start, end, device=dev).unsqueeze(1)
            cols = torch.arange(start, stop, device=dev).unsqueeze(0)
            keep = (cols > rows) & (block >= min_similarity)
            if t is not None and window_seconds is not None:
                keep &= (t[cols] - t[rows]) <= window_seconds
            idx = keep.nonzero(as_tuple=False)
            if idx.numel() == 0:
                continue
            # Both indices are block-local; column k of the block is absolute
            # index start + k, same as the row offset.
            i = (idx[:, 0] + start).cpu().numpy()
            j = (idx[:, 1] + start).cpu().numpy()
            yield i, j, block[keep].cpu().numpy()

    return pairs


def pair_similarity_percentile(
    vectors: np.ndarray,
    times: np.ndarray,
    percentile: float,
    *,
    window_seconds: float,
    max_rows: int = 4000,
    chunk: int = 512,
    seed: int = 0,
    similarity: PairSimilarity = cosine_pairs,
) -> float:
    """The percentile of the in-window pair-similarity distribution.

    Estimated from a uniform random sample of ROWS rather than of pairs, which
    is both unbiased and cheap: uniform row sampling gives every pair the same
    inclusion probability, and the cost falls from O(n^2) to O(max_rows^2).
    Exact when the input is smaller than `max_rows`.
    """
    n = len(vectors)
    if n < 2:
        return float("inf")
    if n > max_rows:
        pick = np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False))
        vectors, times = vectors[pick], times[pick]
    sims = [
        block
        for _, _, block in similarity(vectors, times, window_seconds=window_seconds, chunk=chunk)
    ]
    if not sims:
        return float("inf")
    return float(np.percentile(np.concatenate(sims), percentile))


def text_similarity_network(
    rows: pd.DataFrame,
    vectors: np.ndarray,
    *,
    percentile: float | None = EDGE_PERCENTILE["text_similarity"],
    threshold: float | None = None,
    window_days: float = TEXT_WINDOW_DAYS,
    chunk: int = 512,
    max_sample_rows: int = 4000,
    seed: int = 0,
    similarity: PairSimilarity = cosine_pairs,
) -> pd.DataFrame:
    """Users linked by at least one similar pair of tweets; weight is the MEAN
    similarity over the qualifying pairs.

    `rows` needs `user_id` and `created_at`; `vectors` is one embedding per row,
    in the same order. Embeddings are injected rather than computed here so the
    encoder stays pluggable - the paper uses `stsb-xlm-r-multilingual`, this
    project uses `paraphrase-multilingual-mpnet-base-v2` by the Q2 decision, and
    the whole point of thresholding by percentile is that neither space's cosine
    value ports to the other.

    UNRESOLVED, and it matters: the paper reports 0.95 as the 96th percentile of
    its similarity distribution, but does not say which population that is. Over
    every in-window pair - what this function computes with the shipped
    exhaustive search - a 96th percentile means 4% of ALL pairs become edges,
    which on any real corpus is an enormous and mostly meaningless edge set. The
    figure is far more consistent with a percentile over the top-k neighbours a
    FAISS index returns per tweet, a much smaller and much more similar
    population. Pass `threshold` explicitly to sidestep the question, and treat
    a percentile-derived text network as indicative until the benchmark settles
    which reading reproduces the paper's AUC of 0.52.
    """
    if rows.empty or len(rows) < 2:
        return pd.DataFrame({"source": [], "target": [], "weight": []}).astype({"weight": "float64"})

    order = np.argsort(rows["created_at"].to_numpy(), kind="stable")
    rows = rows.iloc[order].reset_index(drop=True)
    vectors = np.asarray(vectors, dtype="float64")[order]
    times = epoch_seconds(rows["created_at"])
    window_seconds = float(window_days) * 86400.0
    users = rows["user_id"].to_numpy()

    if threshold is None:
        if percentile is None:
            raise ValueError("text_similarity_network needs either a percentile or a threshold")
        threshold = pair_similarity_percentile(
            vectors,
            times,
            percentile,
            window_seconds=window_seconds,
            max_rows=max_sample_rows,
            chunk=chunk,
            seed=seed,
            similarity=similarity,
        )
        log.info("text similarity: %.1fth percentile threshold = %.4f", percentile, threshold)

    parts = []
    for i, j, sim in similarity(vectors, times, window_seconds=window_seconds, chunk=chunk):
        keep = (sim >= threshold) & (users[i] != users[j])
        if not keep.any():
            continue
        a, b = users[i[keep]], users[j[keep]]
        lo = np.where(a <= b, a, b)
        hi = np.where(a <= b, b, a)
        parts.append(
            pd.DataFrame({"source": lo, "target": hi, "weight": sim[keep]})
            .groupby(["source", "target"], as_index=False)
            .agg(weight=("weight", "sum"), pairs=("weight", "size"))
        )
    if not parts:
        return pd.DataFrame({"source": [], "target": [], "weight": []}).astype({"weight": "float64"})

    agg = (
        pd.concat(parts, ignore_index=True)
        .groupby(["source", "target"], as_index=False)
        .agg(weight=("weight", "sum"), pairs=("pairs", "sum"))
    )
    agg["weight"] = agg["weight"] / agg["pairs"]
    return agg.drop(columns=["pairs"])


# --------------------------------------------------------------------------
# Fusion and unsupervised detection.
# --------------------------------------------------------------------------


def fuse(networks: dict[str, pd.DataFrame]) -> nx.Graph:
    """The fused network: two users are linked if they are linked in ANY
    individual similarity network.

    A plain edge UNION, and deliberately not a weighted aggregation. The paper
    tested aggregating normalised weights and taking the maximum centrality
    across networks; the union won, so the fused graph carries no weights at all
    and `which` records only the traces an edge came from.
    """
    graph = nx.Graph()
    for name, edges in networks.items():
        for source, target in zip(edges["source"], edges["target"], strict=True):
            if graph.has_edge(source, target):
                graph[source][target]["traces"].add(name)
            else:
                graph.add_edge(source, target, traces={name})
    return graph


def centrality(
    graph: nx.Graph, nodes: Sequence[str] | None = None, *, dense_below: int = 500
) -> pd.Series:
    """Unweighted eigenvector centrality; nodes absent from the graph score 0.

    Unweighted because the paper measured that weighting it does not improve
    classification. `nodes` is the account population being scored, so accounts
    that appear in no similarity network still get a row, at 0.

    The leading eigenvector of the adjacency matrix is taken directly rather
    than through `nx.eigenvector_centrality_numpy`, which refuses a disconnected
    graph outright (`AmbiguousSolution`) - and a fused similarity network is
    disconnected essentially always. INHERITED CONSEQUENCE, worth knowing before
    reading any output: on a disconnected graph the leading eigenvector
    concentrates on the dominant component, so a small dense clique sitting in
    its own component scores ~0 and is never pruned in. That is a property of
    eigenvector centrality, not of this implementation, and it is the first
    thing to check if the benchmark misses recall.
    """
    order = list(graph.nodes)
    index = list(nodes) if nodes is not None else order

    if graph.number_of_edges() == 0:
        return pd.Series(0.0, index=index, name="centrality")

    if len(order) <= dense_below:
        adj = nx.to_numpy_array(graph, nodelist=order, weight=None)
        vec = np.linalg.eigh(adj)[1][:, -1]
    else:
        from scipy.sparse.linalg import eigsh

        adj = nx.to_scipy_sparse_array(graph, nodelist=order, weight=None, dtype=float)
        vec = eigsh(adj, k=1, which="LA")[1][:, 0]
    vec = np.abs(vec)
    scores = dict(zip(order, vec / np.linalg.norm(vec), strict=True))
    return pd.Series([scores.get(n, 0.0) for n in index], index=index, name="centrality")


def prune(scores: pd.Series, threshold: float = CENTRALITY_THRESHOLD) -> pd.Index:
    """The accounts that survive node pruning - the unsupervised prediction."""
    return scores.index[scores >= threshold]


def detect(
    networks: dict[str, pd.DataFrame],
    *,
    threshold: float = CENTRALITY_THRESHOLD,
    nodes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fuse, score by centrality, prune: one row per account, with a verdict.

    The unit is the ACCOUNT. There is no cluster here and none is implied.
    """
    graph = fuse(networks)
    scores = centrality(graph, nodes)
    return pd.DataFrame(
        {
            "user_id": scores.index,
            "centrality": scores.to_numpy(),
            "predicted": (scores >= threshold).to_numpy(),
        }
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# Supervised detection: node2vec over the fused network, then a random forest.
# --------------------------------------------------------------------------


def random_walks(
    graph: nx.Graph,
    *,
    walks: int = NODE2VEC_WALKS,
    length: int = NODE2VEC_LENGTH,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 0,
) -> list[list[str]]:
    """node2vec's second-order biased walks: `walks` per node, `length` steps each.

    `p` and `q` default to 1, which reduces the walk to a uniform one; the paper
    reports the walk count and length but not the return and in-out parameters.
    """
    rng = np.random.default_rng(seed)
    adjacency = {n: list(graph.neighbors(n)) for n in graph.nodes}
    neighbour_sets = {n: set(nbrs) for n, nbrs in adjacency.items()}
    out: list[list[str]] = []
    for _ in range(walks):
        for start in graph.nodes:
            walk = [start]
            while len(walk) < length:
                current = adjacency[walk[-1]]
                if not current:
                    break
                if len(walk) == 1:
                    walk.append(current[rng.integers(len(current))])
                    continue
                previous = walk[-2]
                weights = np.array(
                    [
                        1.0 / p
                        if n == previous
                        else (1.0 if n in neighbour_sets[previous] else 1.0 / q)
                        for n in current
                    ]
                )
                walk.append(current[rng.choice(len(current), p=weights / weights.sum())])
            out.append(walk)
    return out


def _skipgram_pairs(
    walks: list[list[int]], window: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    centers, contexts = [], []
    for walk in walks:
        arr = np.asarray(walk)
        for offset in range(1, window + 1):
            if offset >= len(arr):
                break
            centers.append(arr[:-offset])
            contexts.append(arr[offset:])
            centers.append(arr[offset:])
            contexts.append(arr[:-offset])
    if not centers:
        return np.array([], dtype="int64"), np.array([], dtype="int64")
    c, x = np.concatenate(centers), np.concatenate(contexts)
    shuffle = rng.permutation(len(c))
    return c[shuffle], x[shuffle]


def node2vec(
    graph: nx.Graph,
    *,
    dim: int = NODE2VEC_DIM,
    walks: int = NODE2VEC_WALKS,
    length: int = NODE2VEC_LENGTH,
    window: int = 10,
    negatives: int = 5,
    epochs: int = 5,
    batch: int = 1024,
    lr: float = 0.025,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 0,
    device: str | None = None,
    score_fn: Callable[[pd.DataFrame], float] | None = None,
    eval_every: int = 20,
    patience: int = 5,
) -> pd.DataFrame:
    """128-dimensional node embeddings: biased walks, then skip-gram with
    negative sampling trained in torch.

    Written out rather than taken from gensim, which is not a dependency here:
    skip-gram with negative sampling over walks is ~40 lines against a new
    top-level package whose scipy pins have broken this environment's before.
    """
    import torch

    nodes = list(graph.nodes)
    if not nodes:
        return pd.DataFrame(index=pd.Index([], name="user_id"))
    index = {n: i for i, n in enumerate(nodes)}
    corpus = [[index[n] for n in walk] for walk in random_walks(
        graph, walks=walks, length=length, p=p, q=q, seed=seed
    )]

    rng = np.random.default_rng(seed)
    centers, contexts = _skipgram_pairs(corpus, window, rng)
    torch.manual_seed(seed)
    # CUDA when it is there, because the epoch counts this needs to converge
    # make a CPU pass the bottleneck on the larger graphs. `device=` overrides.
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inp = torch.nn.Embedding(len(nodes), dim).to(dev)
    out = torch.nn.Embedding(len(nodes), dim).to(dev)
    torch.nn.init.uniform_(inp.weight, -0.5 / dim, 0.5 / dim)
    torch.nn.init.zeros_(out.weight)
    if len(centers) == 0:
        return pd.DataFrame(
            inp.weight.detach().cpu().numpy(), index=pd.Index(nodes, name="user_id")
        )

    # word2vec's unigram^0.75 negative distribution: frequent nodes are sampled
    # as negatives more often, but sub-linearly in their frequency.
    counts = np.bincount(centers, minlength=len(nodes)).astype("float64") ** 0.75
    noise = torch.tensor(counts / counts.sum(), dtype=torch.float, device=dev)
    optimiser = torch.optim.Adam(list(inp.parameters()) + list(out.parameters()), lr=lr)
    centers_t = torch.tensor(centers, dtype=torch.long, device=dev)
    contexts_t = torch.tensor(contexts, dtype=torch.long, device=dev)

    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            inp.weight.detach().cpu().numpy(), index=pd.Index(nodes, name="user_id")
        )

    # Early stopping on a caller-supplied score, mirroring the reference
    # baseline's stop on validation Macro-F1. Without it the epoch count is a
    # compute budget rather than a property of the method, and a comparison then
    # measures how long we were willing to wait.
    best_score = -np.inf
    best_weights = None
    stale = 0

    for epoch in range(epochs):
        for start in range(0, len(centers_t), batch):
            c = inp(centers_t[start : start + batch])
            positive = out(contexts_t[start : start + batch])
            neg_idx = torch.multinomial(noise, len(c) * negatives, replacement=True)
            negative = out(neg_idx).view(len(c), negatives, dim)
            pos_score = torch.nn.functional.logsigmoid((c * positive).sum(-1))
            neg_score = torch.nn.functional.logsigmoid(
                -torch.bmm(negative, c.unsqueeze(-1)).squeeze(-1)
            ).sum(-1)
            loss = -(pos_score + neg_score).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if score_fn is None or (epoch + 1) % eval_every:
            continue
        score = score_fn(_frame())
        if score > best_score:
            best_score, stale = score, 0
            best_weights = inp.weight.detach().clone()
        else:
            stale += 1
            if stale >= patience:
                break

    if best_weights is not None:
        return pd.DataFrame(
            best_weights.cpu().numpy(), index=pd.Index(nodes, name="user_id")
        )
    return _frame()


def classification_metrics(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray, *, threshold: float
) -> dict[str, float]:
    """AUC over the raw score, plus precision/recall/F1 at a decision threshold."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype="float64")
    predicted = (s >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }


def classify(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    folds: int = 10,
    seed: int = 0,
    n_estimators: int = 100,
    threshold: float = 0.5,
) -> tuple[dict[str, float], pd.Series]:
    """A random forest over the node embeddings, scored by 10-fold CV.

    Returns the metrics and the out-of-fold probabilities, so a caller can
    re-threshold or plot without re-fitting. `labels` is aligned to `features`
    by index, and accounts missing a label are dropped rather than assumed
    organic - an unlabelled account in the IO archive is unknown, not negative.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    y = labels.reindex(features.index).dropna().astype(int)
    x = features.loc[y.index]
    counts = y.value_counts()
    splits = min(folds, int(counts.min())) if len(counts) > 1 else 0
    if splits < 2:
        raise ValueError("cross-validation needs at least two examples of each class")
    if splits < folds:
        log.info("reduced to %d folds: the smaller class has %d members", splits, splits)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(model, x.to_numpy(), y.to_numpy(), cv=cv, method="predict_proba")[:, 1]
    oof = pd.Series(probabilities, index=y.index, name="probability")
    return classification_metrics(y.to_numpy(), probabilities, threshold=threshold), oof
