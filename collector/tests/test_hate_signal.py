"""Unit tests for hate-based seed selection (local temp tables, no R2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from kenya_monitor import hate_signal as hsig

NOW = datetime.now(timezone.utc)

POSTS = "_posts"
AUTHORS = "_authors"
HATE = "_hate"
ENGAGEMENTS = "_eng"


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute(
        """
        CREATE TABLE _posts (
            platform VARCHAR, platform_post_id VARCHAR, author_id VARCHAR,
            text VARCHAR, created_at TIMESTAMPTZ, collected_at TIMESTAMPTZ,
            repost_of_id VARCHAR, in_reply_to_id VARCHAR, conversation_id VARCHAR,
            is_repost BOOLEAN, is_quote BOOLEAN, repost_count BIGINT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE _authors (
            platform VARCHAR, platform_user_id VARCHAR, handle VARCHAR,
            followers_count BIGINT, following_count BIGINT, tweet_count BIGINT,
            bio VARCHAR, profile_image_url VARCHAR,
            created_at TIMESTAMPTZ, collected_at TIMESTAMPTZ
        )
        """
    )
    c.execute(
        "CREATE TABLE _hate (platform_post_id VARCHAR, label VARCHAR, "
        "hate_flag BOOLEAN, p_hate DOUBLE, scored_at TIMESTAMPTZ)"
    )
    c.execute(
        "CREATE TABLE _eng (platform_post_id VARCHAR, platform_user_id VARCHAR, kind VARCHAR)"
    )
    return c


def add_author(con, uid, handle):
    con.execute(
        "INSERT INTO _authors VALUES ('x', ?, ?, 100, 100, 500, 'bio', 'img', ?, ?)",
        [uid, handle, NOW - timedelta(days=800), NOW],
    )


def add_posts(con, uid, n_posts, n_toxic, *, prefix, days=5, conversation=None, reply=False):
    """n_posts by `uid`, the first n_toxic of them flagged hate."""
    for i in range(n_posts):
        pid = f"{prefix}{i}"
        created = NOW - timedelta(days=i % days, hours=i)
        con.execute(
            "INSERT INTO _posts VALUES ('x', ?, ?, ?, ?, ?, NULL, ?, ?, FALSE, FALSE, 0)",
            [pid, uid, f"text {pid}", created, NOW,
             f"root_{conversation}" if reply else None,
             conversation if conversation else None],
        )
        toxic = i < n_toxic
        con.execute(
            "INSERT INTO _hate (platform_post_id, label, hate_flag, p_hate, scored_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [pid, "hate" if toxic else "neither", toxic, 0.9 if toxic else 0.01, NOW],
        )


def add_amplification(con, uid, obj_ids):
    """`uid` retweets each object (via the engagement census)."""
    for obj in obj_ids:
        con.execute("INSERT INTO _eng VALUES (?, ?, 'retweet')", [obj, uid])


def seeds(con, **kw):
    return hsig.hate_accounts(con, HATE, POSTS, AUTHORS, ENGAGEMENTS, **kw)


def test_no_scores_yet_returns_empty(con):
    assert hsig.hate_accounts(con, "_missing_view", POSTS, AUTHORS) == []


def test_lone_ranter_ranks_below_a_coordinated_account(con):
    """Qualification is permissive (three routes); ranking is what is selective.

    A prolific solo toxic poster still qualifies on the volume route - it is a
    legitimate thing to watch - but with no co-amplification and no brigade it
    forfeits 0.45 of the weight, so it must never outrank a coordinated account.
    Putting coordination in the FLOOR instead (the previous design) is what
    silently discarded every brigade participant."""
    add_author(con, "u1", "ranter")
    add_posts(con, "u1", 20, 15, prefix="r")
    _network(con)
    ranked = seeds(con)
    by_handle = {s["handle"]: s for s in ranked}
    assert "ranter" in by_handle
    assert by_handle["ranter"]["n_repeat_peers"] == 0
    assert by_handle["ranter"]["n_brigade_convs"] == 0
    order = [s["handle"] for s in ranked]
    assert order.index("ranter") > max(order.index(h) for h in ("net1", "net2", "net3"))


def test_single_toxic_post_is_not_a_seed(con):
    add_author(con, "u1", "oneoff")
    add_posts(con, "u1", 20, 1, prefix="o")
    add_author(con, "u2", "peer1")
    add_author(con, "u3", "peer2")
    add_amplification(con, "u2", ["o0"])
    add_amplification(con, "u3", ["o0"])
    assert [s["handle"] for s in seeds(con)] == []


def test_thin_history_is_not_a_seed(con):
    """Fewer than HATE_MIN_POSTS: no basis to estimate a rate."""
    add_author(con, "u1", "thin")
    add_posts(con, "u1", 5, 5, prefix="t")
    add_author(con, "u2", "peer1")
    add_author(con, "u3", "peer2")
    add_amplification(con, "u2", ["t0"])
    add_amplification(con, "u3", ["t1"])
    assert seeds(con) == []


def _network(con):
    """Three accounts posting toxic material and amplifying each other."""
    for i in (1, 2, 3):
        add_author(con, f"n{i}", f"net{i}")
        add_posts(con, f"n{i}", 20, 8, prefix=f"n{i}p")
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            if i != j:
                add_amplification(con, f"n{i}", [f"n{j}p0", f"n{j}p1"])


def test_mutually_amplifying_network_is_seeded(con):
    _network(con)
    got = {s["handle"] for s in seeds(con)}
    assert got == {"net1", "net2", "net3"}
    for s in seeds(con):
        assert s["n_repeat_peers"] >= hsig.HATE_MIN_REPEAT_PEERS


def test_wilson_bound_beats_raw_rate(con):
    """A sustained 40/300 must outrank a lucky 3/10 on the rate component."""
    add_author(con, "hi", "sustained")
    add_posts(con, "hi", 300, 40, prefix="hi")
    add_author(con, "lo", "lucky")
    add_posts(con, "lo", 10, 3, prefix="lo")
    # Everyone amplifies the same two toxic objects, so both clear min_peers and
    # the ranking difference is attributable to the rate component alone.
    for uid, handle in (("p1", "peer1"), ("p2", "peer2")):
        add_author(con, uid, handle)
    for uid in ("hi", "lo", "p1", "p2"):
        add_amplification(con, uid, ["hi0", "lo0"])
    by_handle = {s["handle"]: s for s in seeds(con)}
    assert {"sustained", "lucky"} <= set(by_handle)
    assert by_handle["sustained"]["toxic_rate_lb"] > by_handle["lucky"]["toxic_rate_lb"]


def test_brigade_needs_enough_distinct_repliers(con):
    """Two accounts replying is a conversation; BRIGADE_MIN_AUTHORS is a pile-on."""
    _network(con)
    for i in (1, 2):
        add_posts(con, f"n{i}", 4, 4, prefix=f"b{i}", conversation="conv1", reply=True)
    assert all(s["n_brigade_convs"] == 0 for s in seeds(con))

    add_posts(con, "n3", 4, 4, prefix="b3", conversation="conv1", reply=True)
    got = {s["handle"]: s["n_brigade_convs"] for s in seeds(con)}
    assert all(v >= 1 for v in got.values())


def test_coordination_shape_outweighs_toxicity_level(con):
    """The central design claim: co-amplification + brigade (0.45) carries more
    than the toxicity rate (0.25), so a well-connected moderate beats an
    isolated extremist."""
    assert (
        hsig.HATE_SEED_WEIGHTS["co_amplify"] + hsig.HATE_SEED_WEIGHTS["brigade"]
        > hsig.HATE_SEED_WEIGHTS["toxic_rate_lb"] + hsig.HATE_SEED_WEIGHTS["persistence"]
    )
    assert abs(sum(hsig.HATE_SEED_WEIGHTS.values()) - 1.0) < 1e-9


def test_top_hate_seeds_returns_handles(con):
    _network(con)
    handles = hsig.top_hate_seeds(con, HATE, POSTS, AUTHORS, ENGAGEMENTS, n=2)
    assert len(handles) == 2
    assert all(isinstance(h, str) for h in handles)


def test_works_without_an_engagement_census(con):
    """Retweets collected as posts still count when engagements/ is absent."""
    for i in (1, 2, 3):
        add_author(con, f"n{i}", f"net{i}")
        add_posts(con, f"n{i}", 20, 8, prefix=f"n{i}p")
    for i, j in ((1, 2), (2, 1), (3, 1), (1, 3), (2, 3), (3, 2)):
        con.execute(
            "INSERT INTO _posts VALUES ('x', ?, ?, 'rt', ?, ?, ?, NULL, NULL, TRUE, FALSE, 0)",
            [f"rt{i}{j}", f"n{i}", NOW - timedelta(hours=1), NOW, f"n{j}p0"],
        )
    got = {s["handle"] for s in hsig.hate_accounts(con, HATE, POSTS, AUTHORS, None)}
    assert got == {"net1", "net2", "net3"}


def test_stale_scores_are_detected(con):
    add_posts(con, "u1", 1, 1, prefix="s")
    assert hsig.scores_are_stale(con, HATE) is False
    con.execute("UPDATE _hate SET scored_at = ?", [NOW - timedelta(days=3)])
    assert hsig.scores_are_stale(con, HATE) is True


def test_stale_check_handles_missing_view(con):
    assert hsig.scores_are_stale(con, "_nope") is True


def test_hot_toxic_objects_ranks_by_toxicity(con):
    add_author(con, "u1", "a")
    add_posts(con, "u1", 6, 3, prefix="h", days=1)
    con.execute("UPDATE _posts SET repost_count = 50 WHERE platform_post_id = 'h0'")
    con.execute("UPDATE _posts SET repost_count = 99 WHERE platform_post_id = 'h5'")
    retweeted, conversations, missing = hsig.hot_toxic_objects(con, HATE, POSTS)
    # h5 has more retweets but is not toxic, so it must not be selected.
    assert "h0" in retweeted
    assert "h5" not in retweeted
    assert missing == []


def test_hot_toxic_objects_finds_brigaded_conversations(con):
    add_author(con, "u1", "a")
    add_posts(con, "u1", 6, 6, prefix="c", days=1, conversation="conv9", reply=True)
    _, conversations, _ = hsig.hot_toxic_objects(con, HATE, POSTS)
    assert "conv9" in conversations


def test_hot_toxic_objects_without_scores(con):
    assert hsig.hot_toxic_objects(con, "_nope", POSTS) == ([], [], [])


def test_kenya_scope_and_coded_columns_are_used_when_present(con):
    cols = {"label", "hate_flag", "in_kenya_scope", "coded_suspect"}
    expr = hsig._toxic_expr(cols)
    assert "coded_suspect" in expr and "in_kenya_scope" in expr
    bare = hsig._toxic_expr({"label", "hate_flag"})
    assert "coded_suspect" not in bare and "in_kenya_scope" not in bare


def test_off_domain_posts_are_excluded_once_scope_is_persisted(con):
    """With in_kenya_scope persisted, US-politics toxicity must not seed."""
    con.execute("ALTER TABLE _hate ADD COLUMN in_kenya_scope BOOLEAN")
    _network(con)
    con.execute("UPDATE _hate SET in_kenya_scope = TRUE")
    assert {s["handle"] for s in seeds(con)} == {"net1", "net2", "net3"}
    con.execute("UPDATE _hate SET in_kenya_scope = FALSE")
    assert seeds(con) == []


# --- co-amplification guards -------------------------------------------------


def test_hub_objects_do_not_create_peers(con):
    """Measured 2026-07-31: 9 objects with >50 amplifiers produced 43% of all
    co-amplification rows. Two accounts both retweeting one viral toxic post is
    organic, not coordination."""
    add_author(con, "a", "alpha")
    add_author(con, "b", "beta")
    add_posts(con, "a", 12, 4, prefix="ap")
    add_posts(con, "b", 12, 4, prefix="bp")
    # One toxic object amplified by a crowd well over HUB_CAP_MIN.
    for i in range(hsig.HUB_CAP_MIN + 20):
        add_amplification(con, f"crowd{i}", ["ap0"])
    add_amplification(con, "a", ["ap0"])
    add_amplification(con, "b", ["ap0"])
    by_handle = {s["handle"]: s for s in seeds(con)}
    assert by_handle["alpha"]["n_repeat_peers"] == 0
    assert by_handle["beta"]["n_repeat_peers"] == 0


def test_one_shared_object_is_not_repeat_co_amplification(con):
    add_author(con, "a", "alpha")
    add_author(con, "b", "beta")
    add_posts(con, "a", 12, 4, prefix="ap")
    add_posts(con, "b", 12, 4, prefix="bp")
    for uid in ("a", "b"):
        add_amplification(con, uid, ["ap0"])          # one shared object only
    assert all(s["n_repeat_peers"] == 0 for s in seeds(con))

    for uid in ("a", "b"):
        add_amplification(con, uid, ["ap1"])          # now two -> repeat
    by_handle = {s["handle"]: s for s in seeds(con)}
    assert by_handle["alpha"]["n_repeat_peers"] >= 1
    assert by_handle["beta"]["n_repeat_peers"] >= 1


# --- qualification routes ----------------------------------------------------


def test_repeat_brigader_qualifies_without_volume(con):
    """The bug this whole change exists for: brigading is many accounts each
    posting once, so a volume floor removed 936 of 939 brigade participants and
    left the brigade component at exactly 0 for every seed."""
    for i in range(4):
        add_author(con, f"b{i}", f"brig{i}")
    # Two separate pile-ons, each with 4 distinct toxic repliers, 1 post each.
    for conv in ("conv1", "conv2"):
        for i in range(4):
            add_posts(con, f"b{i}", 1, 1, prefix=f"{conv}_{i}_", conversation=conv, reply=True)
    got = {s["handle"]: s for s in seeds(con)}
    assert set(got) == {"brig0", "brig1", "brig2", "brig3"}
    for s in got.values():
        assert s["n_posts"] < hsig.HATE_MIN_POSTS      # nowhere near the volume floor
        assert s["n_brigade_convs"] >= hsig.HATE_MIN_BRIGADES


def test_single_brigade_is_not_enough(con):
    for i in range(4):
        add_author(con, f"b{i}", f"brig{i}")
    for i in range(4):
        add_posts(con, f"b{i}", 1, 1, prefix=f"c_{i}_", conversation="conv1", reply=True)
    assert seeds(con) == []


# --- observation-effort control ---------------------------------------------


def test_volume_bucketing_stops_effort_buying_rank(con):
    """Collecting more of an account's timeline must move it to a higher-volume
    stratum, not up the ranking."""
    for i in range(6):                      # low-volume cohort
        add_author(con, f"l{i}", f"low{i}")
        add_posts(con, f"l{i}", 12, 4, prefix=f"l{i}p")
    for i in range(6):                      # high-volume cohort, same behaviour
        add_author(con, f"h{i}", f"high{i}")
        add_posts(con, f"h{i}", 120, 40, prefix=f"h{i}p")
    for grp in ("l", "h"):
        for i in range(6):
            add_amplification(con, f"{grp}{i}", [f"{grp}0p0", f"{grp}0p1"])
    rows = seeds(con)
    buckets = {r["handle"]: r["volume_bucket"] for r in rows}
    assert len({buckets[h] for h in buckets if h.startswith("low")}) >= 1
    # the two cohorts must not share a bucket - that is the whole point
    low_buckets = {b for h, b in buckets.items() if h.startswith("low")}
    high_buckets = {b for h, b in buckets.items() if h.startswith("high")}
    assert low_buckets.isdisjoint(high_buckets)


def test_thin_cohort_suppresses_the_score(con):
    """percent_rank is 0 for one row and 1/(n-1) steps for a few; report the
    evidence rather than dressing noise up as a ranking."""
    add_author(con, "u1", "solo")
    add_posts(con, "u1", 20, 15, prefix="s")
    rows = seeds(con)
    assert len(rows) < hsig.MIN_COHORT_FOR_RANK
    assert all(r["hate_seed_score"] is None for r in rows)


# --- coordination edges are deliberately NOT used for peer counts ------------


def test_edge_staleness_probe_tolerates_a_missing_prefix(con):
    """`edges_are_stale` still guards `adaptive`'s cluster promotion, which is
    where persisted coordination edges DO belong."""
    assert hsig.edges_are_stale(con, "_no_such_edges") is True


def test_co_amplification_is_scoped_to_toxic_objects(con):
    """Sharing a non-toxic object must not create a peer edge. This is why the
    all-posts coordination edges could not be substituted here."""
    add_author(con, "a", "alpha")
    add_author(con, "b", "beta")
    add_posts(con, "a", 12, 1, prefix="ap")   # ap0 toxic, ap1+ not
    add_posts(con, "b", 12, 1, prefix="bp")
    for uid in ("a", "b"):
        add_amplification(con, uid, ["ap2", "ap3"])   # two shared NON-toxic objects
    assert all(s["n_repeat_peers"] == 0 for s in seeds(con))


# --- censused-TTL symmetry with the baseline selector ------------------------


def _censused(con, object_ids, hours_ago=0):
    """An engagements table shaped like the R2 prefix (with collected_at)."""
    con.execute(
        "CREATE OR REPLACE TABLE _eng_ttl "
        "(platform VARCHAR, platform_post_id VARCHAR, platform_user_id VARCHAR, "
        " kind VARCHAR, collected_at TIMESTAMPTZ)"
    )
    for oid in object_ids:
        con.execute(
            "INSERT INTO _eng_ttl VALUES ('x', ?, 'u1', 'retweet', ?)",
            [oid, NOW - timedelta(hours=hours_ago)],
        )
    return "_eng_ttl"


def test_hot_toxic_objects_excludes_recently_censused_objects(con):
    """The toxic selector is deterministic too, so without the TTL filter it
    re-picks completed work every pass and `_due` throws it away. This is the
    same bug that was fixed on the baseline path on 2026-08-01 and left live
    here."""
    add_author(con, "u1", "a")
    add_posts(con, "u1", 6, 6, prefix="h", days=1)
    con.execute("UPDATE _posts SET repost_count = 50 WHERE platform_post_id = 'h0'")
    con.execute("UPDATE _posts SET repost_count = 40 WHERE platform_post_id = 'h1'")
    eng = _censused(con, ["h0"])

    unfiltered, _, _ = hsig.hot_toxic_objects(con, HATE, POSTS)
    assert "h0" in unfiltered

    filtered, _, _ = hsig.hot_toxic_objects(
        con, HATE, POSTS, engagements_view=eng, refresh_hours=12
    )
    assert "h0" not in filtered
    assert "h1" in filtered


def test_toxic_censused_objects_return_once_the_ttl_lapses(con):
    add_author(con, "u1", "a")
    add_posts(con, "u1", 6, 6, prefix="h", days=1)
    con.execute("UPDATE _posts SET repost_count = 50 WHERE platform_post_id = 'h0'")
    eng = _censused(con, ["h0"], hours_ago=20)

    within = hsig.hot_toxic_objects(con, HATE, POSTS, engagements_view=eng, refresh_hours=48)[0]
    lapsed = hsig.hot_toxic_objects(con, HATE, POSTS, engagements_view=eng, refresh_hours=12)[0]

    assert "h0" not in within      # censused 20h ago, TTL 48h -> still done
    assert "h0" in lapsed          # TTL 12h -> due again


def test_toxic_selector_tolerates_a_missing_engagements_view(con):
    add_author(con, "u1", "a")
    add_posts(con, "u1", 6, 6, prefix="h", days=1)
    con.execute("UPDATE _posts SET repost_count = 50 WHERE platform_post_id = 'h0'")

    got, _, _ = hsig.hot_toxic_objects(con, HATE, POSTS, engagements_view="_nope")

    assert "h0" in got


def test_both_selectors_use_the_one_censused_filter(monkeypatch):
    """The two selectors band on different columns and drifted once already:
    the TTL exclusion was added to `hot_objects` and not to
    `hot_toxic_objects`, which then re-picked completed work every pass. One
    helper, called by both, with the column as an argument."""
    from kenya_monitor import census, runner

    seen: list[str] = []

    def record_expr(con, engagements_view, column, refresh_hours=12):
        seen.append(column)
        return "FALSE"

    def record_filter(con, engagements_view, column, refresh_hours=12):
        seen.append(column)
        return ""

    monkeypatch.setattr(census, "censused_expr", record_expr)
    monkeypatch.setattr(census, "censused_filter", record_filter)
    monkeypatch.setattr(hsig.census_sel, "censused_filter", record_filter)

    c = duckdb.connect()
    c.execute(
        "CREATE TABLE p (platform VARCHAR, platform_post_id VARCHAR, author_id VARCHAR,"
        " created_at TIMESTAMPTZ, collected_at TIMESTAMPTZ, repost_of_id VARCHAR,"
        " quoted_post_id VARCHAR, in_reply_to_id VARCHAR, conversation_id VARCHAR,"
        " repost_count BIGINT, reply_count BIGINT, quote_count BIGINT, is_repost BOOLEAN)"
    )
    c.execute("CREATE TABLE h (platform_post_id VARCHAR, platform VARCHAR,"
              " scored_at TIMESTAMPTZ, label VARCHAR, hate_flag BOOLEAN, p_hate DOUBLE)")

    runner.hot_objects(c, "p", engagements_view="e")
    hsig.hot_toxic_objects(c, "h", "p", engagements_view="e")

    assert seen == ["recent.repost_of_id", "recent.platform_post_id"]


def test_censused_column_must_be_table_qualified():
    """A bare `platform_post_id` binds to the engagements table inside the
    correlated subquery, making the predicate self-referential and true for
    every row - so every candidate is excluded and the census stalls silently."""
    from kenya_monitor import census

    c = duckdb.connect()
    c.execute("CREATE TABLE e (platform_post_id VARCHAR, collected_at TIMESTAMPTZ)")
    with pytest.raises(ValueError, match="table-qualified"):
        census.censused_expr(c, "e", "platform_post_id")
