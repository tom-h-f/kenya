"""The canary: a synthetic coordinated group planted in the detector's input.

Benchmark reproduction proves the method was copied correctly. It does not
prove that the live chain - window, traces, fusion, community listing, leads
diff, reader, poller - would put a campaign on a phone. This does, once a week,
by planting a group whose behaviour is shaped from a real information
operation and checking that it comes out the other end as a `[CANARY]` alert.

## The template, and why it is UAE

The IOHunter release ships each operation as five trace networks, so the shape
available is the one the detector consumes. Profiled locally on 2026-09-22
(`investigations/2026-09-22-canary/01_profile_ops.py`), share of operation
accounts each trace touches:

| op | co_retweet | co_url | hashtag_sequence | text_similarity |
|---|---|---|---|---|
| UAE | 0.757 | 0.900 | **0.443** (op-op degree 14.8) | 0.996 |
| cuba | 0.931 | 0.920 | 0.299 (0.85) | 1.000 |
| iran | 0.599 | 0.599 | 0.101 (1.02) | 0 |
| russia, venezuela, china | 0.03-0.45 | 0.86-0.99 | <= 0.13 | 0 |

Kenya's documented threat is hashtag pushing for hire: many accounts posting
the same ordered tag set and near-identical copy, amplifying each other, to
drive a tag into trending. UAE is the only operation where the hashtag-sequence
trace carries real weight - an order of magnitude above every other - and it
pairs that with copypasta (text similarity touching 99.6%) and amplification.
Cuba is denser but its hashtag trace is ten times weaker, so it would test the
co-retweet core and not the trace Kenya's pattern lives in. fast_retweet
touches 0.6% of UAE's accounts and is left out, as UAE leaves it out.

## What never happens

Nothing here writes under an archive prefix. Planted posts live under
`canary/platform=x/kind=posts/`, the expectation under the leads contract's
`leads/platform=x/kind=canary_expected/` (`docs/analysis/leads.md`), and every
key is checked against those two before a write - `_guard` refuses anything
else. The detector sees the canary only through `inject_view`, which UNIONs it
in memory.

Ids follow the leads contract: accounts `canary:<canary_id>:<n>`; posts
`canary:<canary_id>:p<n>` for originals and `:r<n>` for retweets. URLs are on
`.invalid`, a TLD reserved never to resolve, and the campaign hashtags are checked absent from the background
before planting, so nothing the canary does can be mistaken for, or merge
with, a real entity except the real posts it deliberately retweets.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd

from kma import coord2
from kma.db import BUCKET

log = logging.getLogger(__name__)

ID_PREFIX = "canary:"
POSTS_PREFIX = "canary/platform=x/kind=posts"
EXPECTED_PREFIX = "leads/platform=x/kind=canary_expected"
_WRITABLE = (f"{POSTS_PREFIX}/", f"{EXPECTED_PREFIX}/")

# UAE's measured per-trace participation (see the module docstring).
PARTICIPATION = {"co_retweet": 0.757, "co_url": 0.900, "hashtag_sequence": 0.443}

# Acts per account over the window. "low" is below the v2 activity floor's
# comfort zone on purpose: a campaign run from many light accounts is the hard
# case, and the curve has to say where the detector stops seeing it.
ACTIVITY = {"low": 3, "normal": 10}

# The leads pass runs daily at 06:30 UTC; planted before one pass, a verdict can
# exist by the next and the poller has read it within its interval. The leads
# contract's safe figure.
DEADLINE_HOURS = 36

TAG_SETS = (
    ("KenyaMbele", "HakiYetu2027", "TusongeMbele"),
    ("SautiYaMwananchi", "KenyaMbele", "UchaguziHuru"),
    ("HakiYetu2027", "TusongeMbele", "SautiYaMwananchi", "KenyaMbele"),
)

TEMPLATES = (
    "Wakenya wameamka! IEBC must guarantee a free and fair 2027 election for every county {tags}",
    "The people of Kenya will not be silenced. Our vote, our voice, our future in 2027 {tags}",
    "From Mombasa to Kisumu, Kenyans demand accountability before the 2027 election {tags}",
    "Enough is enough. Kenya deserves leaders who listen to wananchi, not cartels {tags}",
)

_VARIANTS = ("", " Share!", " RT", " Tuko pamoja.", " #Wakenya", " Amkeni!")


@dataclass(frozen=True)
class Plant:
    canary_id: str
    posts: pd.DataFrame
    accounts: list[str]
    injected_at: datetime
    deadline: datetime


def account_id(canary_id: str, n: int) -> str:
    return f"{ID_PREFIX}{canary_id}:{n}"


def is_canary(user_id: object) -> bool:
    return isinstance(user_id, str) and user_id.startswith(ID_PREFIX)


def _guard(key: str) -> str:
    if not key.startswith(_WRITABLE):
        raise ValueError(f"canary refuses to write outside {_WRITABLE}: {key}")
    return key


def posts_key(canary_id: str) -> str:
    return _guard(f"{POSTS_PREFIX}/canary={canary_id}/posts.parquet")


def expected_key(canary_id: str, injected_at: datetime) -> str:
    return _guard(f"{EXPECTED_PREFIX}/dt={injected_at:%Y-%m-%d}/canary={canary_id}.parquet")


def retweet_targets(con: duckdb.DuckDBPyConnection, view: str, k: int = 8) -> list[str]:
    """The most-retweeted real posts in the window: what an operation riding a
    live conversation amplifies, and what ties the planted group to real
    accounts the way UAE's op-to-organic co-retweet edges do."""
    rows = con.sql(
        f"""SELECT retweet_post_id FROM ({view}) WHERE retweet_post_id IS NOT NULL
            GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {int(k)}"""
    ).fetchall()
    return [r[0] for r in rows]


def check_tags_unused(con: duckdb.DuckDBPyConnection, view: str) -> None:
    tags = sorted({t.lower() for s in TAG_SETS for t in s})
    listed = ", ".join(f"'{t}'" for t in tags)
    hit = con.sql(
        f"""SELECT count(*) FROM ({view}), unnest(hashtags) AS h(tag)
            WHERE lower(tag) IN ({listed})"""
    ).fetchone()[0]
    if hit:
        raise ValueError(f"{hit} background posts already use a canary hashtag; pick new tags")


def plant(
    canary_id: str,
    *,
    size: int,
    activity: str | int,
    start: datetime,
    targets: list[str],
    days: int = 7,
    seed: int = 0,
) -> pd.DataFrame:
    """Planted posts, in the canonical coord2 view's columns.

    Each account draws which behaviours it shows from UAE's participation, so
    the group reproduces the operation's per-trace coverage rather than being
    a clique in every trace at once. Acts land in two-hour bursts on shared
    campaign days, the rhythm of a tag being pushed.
    """
    if not targets:
        raise ValueError("plant needs real retweet targets to amplify")
    acts = ACTIVITY[activity] if isinstance(activity, str) else int(activity)
    rng = np.random.default_rng(seed)
    burst_days = sorted(rng.choice(days, size=min(3, days), replace=False).tolist())
    bursts = [start + timedelta(days=d, hours=int(rng.integers(6, 20))) for d in burst_days]
    urls = [f"https://{canary_id.replace(':', '-')}.canary.invalid/{k}" for k in range(3)]

    def when():
        return bursts[int(rng.integers(len(bursts)))] + timedelta(seconds=int(rng.integers(0, 7200)))

    plan = []
    for n in range(size):
        does = {t: rng.random() < p for t, p in PARTICIPATION.items()}
        plan += [(n, does, does["co_retweet"] and rng.random() < 0.5) for _ in range(acts)]

    rows, originals = [], []
    for k, (n, does, _) in enumerate(p for p in plan if not p[2]):
        # A tag set drawn per post, not per account: UAE's sequence trace has an
        # op-to-op degree of 14.8, which needs accounts sharing several distinct
        # sequences - one sequence per account falls under the activity floor.
        tags = TAG_SETS[int(rng.integers(len(TAG_SETS)))]
        use_tags = list(tags) if does["hashtag_sequence"] else list(tags[:1])
        text = TEMPLATES[int(rng.integers(len(TEMPLATES)))].format(
            tags=" ".join(f"#{t}" for t in use_tags)
        ) + _VARIANTS[int(rng.integers(len(_VARIANTS)))]
        url = urls[int(rng.integers(len(urls)))] if does["co_url"] and rng.random() < 0.6 else None
        pid = f"{ID_PREFIX}{canary_id}:p{k}"
        originals.append(pid)
        rows.append({"user_id": account_id(canary_id, n), "post_id": pid, "created_at": when(),
                     "is_retweet": False, "retweet_post_id": None, "retweet_user_id": None,
                     "text": text + (f" {url}" if url else ""),
                     "hashtags": use_tags, "urls": [url] if url else []})

    # Mostly the group's own posts: UAE's co-retweet degree is 176.7 op-to-op
    # against 0.75 op-to-organic, so an operation amplifies itself and touches
    # the live conversation only at the edges.
    for k, (n, _, _) in enumerate(p for p in plan if p[2]):
        own = originals and rng.random() < 0.9
        target = originals[int(rng.integers(len(originals)))] if own else targets[
            int(rng.integers(len(targets)))]
        rows.append({"user_id": account_id(canary_id, n), "post_id": f"{ID_PREFIX}{canary_id}:r{k}",
                     "created_at": when(), "is_retweet": True, "retweet_post_id": target,
                     "retweet_user_id": None, "text": None, "hashtags": [], "urls": []})
    frame = pd.DataFrame(rows, columns=list(coord2.VIEW_COLUMNS))
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    return frame


def inject_view(con: duckdb.DuckDBPyConnection, view: str, posts: pd.DataFrame,
                name: str = "_canary_posts") -> str:
    """The window view with the planted posts UNIONed in, in memory only."""
    con.register(name, posts[list(coord2.VIEW_COLUMNS)])
    cols = ", ".join(coord2.VIEW_COLUMNS)
    return f"SELECT {cols} FROM ({view}) UNION ALL SELECT {cols} FROM {name}"


def text_edges(posts: pd.DataFrame, encoder=None, *, threshold: float = 0.85,
               overlap: float | None = coord2.TEXT_MIN_OVERLAP) -> pd.DataFrame:
    """The text-similarity trace among planted posts, on the production cut.

    Production reads a persisted GPU-built trace; the planted posts are not in
    it, so their edges are built here with the same encoder, threshold and
    word-overlap floor. Planted-to-organic text edges are not built - the
    template copy shares nothing with real posts by construction - which
    understates UAE's op-to-organic text degree and so makes the canary, if
    anything, easier to isolate as its own community and harder to rank
    centrally.
    """
    own = posts[~posts["is_retweet"] & posts["text"].notna()].copy()
    if len(own) < 2:
        return pd.DataFrame({"source": [], "target": [], "weight": []})
    own["clean"] = [coord2.clean_text(t) for t in own["text"]]
    own = own[own["clean"].str.split().str.len() >= coord2.MIN_TEXT_WORDS].reset_index(drop=True)
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        from kma.semantic import MODEL

        encoder = SentenceTransformer(MODEL, device="cpu")
    vectors = encoder.encode(own["text"].tolist(), normalize_embeddings=True)
    return coord2.text_similarity_network(
        own[["user_id", "created_at", "clean"]], vectors,
        threshold=threshold, min_overlap=overlap,
    )


def new(
    con: duckdb.DuckDBPyConnection,
    view: str,
    *,
    canary_id: str | None = None,
    size: int = 20,
    activity: str | int = "normal",
    days: int = 7,
    now: datetime | None = None,
    seed: int = 0,
) -> Plant:
    """Plan one weekly canary against the window it will be judged in."""
    now = now or datetime.now(timezone.utc)
    canary_id = canary_id or f"{now:%Y-%m-%d}"
    check_tags_unused(con, view)
    posts = plant(canary_id, size=size, activity=activity, days=days, seed=seed,
                  start=now - timedelta(days=days), targets=retweet_targets(con, view))
    return Plant(canary_id=canary_id, posts=posts,
                 accounts=[account_id(canary_id, n) for n in range(size)],
                 injected_at=now, deadline=now + timedelta(hours=DEADLINE_HOURS))


def persist(con: duckdb.DuckDBPyConnection, p: Plant, uri=None) -> dict[str, str]:
    """Write the planted posts and the leads contract's expectation row."""
    uri = uri or (lambda key: f"r2://{BUCKET}/{key}")
    keys = {"posts": posts_key(p.canary_id), "expected": expected_key(p.canary_id, p.injected_at)}
    frames = {
        "posts": p.posts.assign(canary_id=p.canary_id),
        "expected": pd.DataFrame([{"canary_id": p.canary_id, "injected_at": p.injected_at,
                                   "deadline": p.deadline, "size": len(p.accounts)}]),
    }
    for name, frame in frames.items():
        con.register("_canary_buf", frame)
        try:
            con.execute(f"COPY _canary_buf TO '{uri(keys[name])}' (FORMAT parquet)")
        finally:
            con.unregister("_canary_buf")
    return keys


def load_active(con: duckdb.DuckDBPyConnection, now: datetime | None = None, uri=None) -> pd.DataFrame:
    """Planted posts whose canary has not passed its deadline - what the daily
    pass UNIONs into its window through `inject_view`."""
    uri = uri or (lambda key: f"r2://{BUCKET}/{key}")
    now = now or datetime.now(timezone.utc)
    try:
        live = con.sql(
            f"SELECT canary_id FROM read_parquet('{uri(EXPECTED_PREFIX)}/dt=*/canary=*.parquet') "
            f"WHERE deadline > TIMESTAMPTZ '{now.isoformat()}'"
        ).df()
    except duckdb.Error:
        return pd.DataFrame(columns=list(coord2.VIEW_COLUMNS))
    frames = [con.sql(f"SELECT * FROM read_parquet('{uri(posts_key(c))}')").df()
              for c in live["canary_id"]]
    if not frames:
        return pd.DataFrame(columns=list(coord2.VIEW_COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(coord2.VIEW_COLUMNS)]


def packet(posts: pd.DataFrame, members: list[str], cluster_id: int,
           max_items: int = 8) -> dict:
    """A dossier packet for a planted community, built from its planted posts.

    `dossier.build` reads member posts from the archive, and the canary is
    never in the archive, so without this the reader gets an empty dossier and
    says `unclear` - which proves delivery and nothing about detection. Same
    keys `dossier.build` emits, filled from memory: the reader judges
    real-shaped evidence.
    """
    mine = posts[posts["user_id"].isin(members)]
    own = mine[~mine["is_retweet"] & mine["text"].notna()]
    reps = own.drop_duplicates("user_id").head(max_items)
    texts = (own.assign(norm=own["text"].str.replace(r"https?://\S+", "", regex=True)
                        .str.strip().str.lower())
             .groupby("norm").agg(n_members=("user_id", "nunique"), n_posts=("post_id", "size"),
                                  text=("text", "first"))
             .sort_values("n_members", ascending=False).head(max_items).reset_index(drop=True))
    rts = mine[mine["is_retweet"]]
    objects = (rts.groupby("retweet_post_id").agg(n_members=("user_id", "nunique"))
               .sort_values("n_members", ascending=False).head(max_items).reset_index()
               .rename(columns={"retweet_post_id": "object_id"}))
    return {
        "cluster_id": int(cluster_id),
        "size": int(len(set(members))),
        "shared_objects": objects.to_dict("records"),
        "shared_texts": texts.assign(author_handle=None)[
            ["n_members", "n_posts", "author_handle", "text"]].to_dict("records"),
        "representative_posts": [
            {"author_id": r.user_id, "author_handle": None, "text": r.text,
             "created_at": r.created_at, "engagement": 0}
            for r in reps.itertuples()
        ],
        "amplification_targets": [],
    }


def evaluate(graph, global_scores: pd.Series, planted: list[str], *,
             budget: int = 500, per_group: int = 25, min_size: int = 4,
             report_fn=None) -> dict:
    """Did the planted group come out: ranked, grouped, listed.

    `report_fn` is the community report the daily pass uses
    (`kma.coord2_communities.report`); the listing it produces is what the
    leads pass diffs, so being listed there is what reaching adjudication
    means. Relevance is not applied here - planted posts are never scored by
    the relevance app, and the daily report keeps unscored communities.
    """
    planted = [u for u in planted if u in graph]
    out = {"planted_in_graph": len(planted), "recall_top_budget": 0.0,
           "community_coverage": 0.0, "community_purity": 0.0,
           "community_size": 0, "listed": False, "listing_rank": None,
           "planted_listed": 0, "detected": False, "isolated": False}
    if not planted:
        return out
    top = set(global_scores.sort_values(ascending=False).head(budget).index)
    out["recall_top_budget"] = round(len(top & set(planted)) / len(planted), 3)

    rep = (report_fn or _report)(graph, global_scores, budget=budget,
                                 per_group=per_group, min_size=min_size)
    members = rep.members
    mine = members[members["user_id"].isin(planted)]
    if mine.empty:
        return out
    community = int(mine["community"].value_counts().idxmax())
    group = members[members["community"] == community]
    out["community_size"] = int(len(group))
    out["community_coverage"] = round(float((mine["community"] == community).sum() / len(planted)), 3)
    out["community_purity"] = round(float(group["user_id"].isin(planted).mean()), 3)
    listed = rep.listed
    if len(listed) and community in set(listed["community"]):
        order = list(dict.fromkeys(listed["community"]))
        out["listed"] = True
        out["listing_rank"] = order.index(community) + 1
        out["planted_listed"] = int(listed["user_id"].isin(planted).sum())
    out["detected"] = bool(out["listed"] and out["community_coverage"] >= 0.5)
    # Listed inside a community it is a minority of, a reader sees mostly organic
    # accounts and the operation is diluted; isolated, the lead IS the operation.
    out["isolated"] = bool(out["detected"] and out["community_purity"] >= 0.5)
    return out


@dataclass
class _Report:
    members: pd.DataFrame
    listed: pd.DataFrame


def _report(graph, global_scores, *, budget, per_group, min_size):
    """`coord2_communities.report` where it exists (workstream A), else the
    same Leiden partition and eigenvalue-ordered listing without the relevance
    filter, so the canary runs on a tree where A is not merged yet."""
    try:
        from kma import coord2_communities

        return coord2_communities.report(graph, global_scores, kenya=None, budget=budget,
                                         per_group=per_group, min_size=min_size)
    except ImportError:
        pass
    import igraph as ig
    import leidenalg as la

    names = list(graph.nodes)
    index = {n: k for k, n in enumerate(names)}
    g = ig.Graph(n=len(names), edges=[(index[a], index[b]) for a, b in graph.edges()])
    part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=1.0, seed=0)
    rows = []
    for c, vs in enumerate(part):
        if len(vs) < min_size:
            continue
        nodes = [names[v] for v in vs]
        adj = np.asarray(__import__("networkx").to_numpy_array(graph.subgraph(nodes), nodelist=nodes))
        values, vectors = np.linalg.eigh(adj)
        vec = np.abs(vectors[:, -1])
        rows += [{"user_id": n, "community": c, "community_eigenvalue": float(values[-1]),
                  "within_score": float(v)} for n, v in zip(nodes, vec, strict=True)]
    members = pd.DataFrame(rows, columns=["user_id", "community", "community_eigenvalue",
                                          "within_score"])
    members = members.sort_values(["community_eigenvalue", "within_score"],
                                  ascending=[False, False]).reset_index(drop=True)
    members["rank_in_group"] = members.groupby("community").cumcount()
    listed = members[members["rank_in_group"] < per_group].head(budget)
    return _Report(members=members, listed=listed)


def alert_title(canary_id: str, verdict: str | None = None) -> str:
    """The poller's title for a canary, as the leads contract fixes it."""
    return f"[CANARY] {canary_id}" + (f": {verdict}" if verdict else "")


_ID = re.compile(r"^canary:([^:]+):")


def canary_of(user_id: str) -> str | None:
    m = _ID.match(user_id or "")
    return m.group(1) if m else None
