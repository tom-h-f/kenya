from datetime import datetime, timezone

import duckdb
import networkx as nx
import pandas as pd
import pytest

from kma import adjudicate, canary, coord2

START = datetime(2026, 9, 7, tzinfo=timezone.utc)
TARGETS = ["real1", "real2"]


def _plant(size=10, activity="normal", seed=0):
    return canary.plant("t1", size=size, activity=activity, start=START, targets=TARGETS, seed=seed)


def test_every_planted_id_is_marked_and_in_the_view_schema():
    posts = _plant()
    assert list(posts.columns) == list(coord2.VIEW_COLUMNS)
    assert posts["user_id"].map(canary.is_canary).all()
    assert posts["post_id"].str.startswith("canary:t1:").all()
    assert posts["post_id"].is_unique
    assert set(posts["user_id"]) == {canary.account_id("t1", n) for n in range(10)}


def test_urls_never_resolve_and_retweets_hit_only_the_group_or_the_given_targets():
    posts = _plant(size=30)
    urls = [u for us in posts["urls"] for u in us]
    assert urls and all(".canary.invalid/" in u for u in urls)
    rts = posts.loc[posts["is_retweet"], "retweet_post_id"]
    own = set(posts.loc[~posts["is_retweet"], "post_id"])
    assert rts.isin(own | set(TARGETS)).all()
    assert rts.isin(own).mean() > 0.7


def test_the_sequence_trace_survives_the_activity_floor():
    """One tag set per account would give every account one distinct sequence
    and the floor would drop them all - the first calibration's result."""
    con = duckdb.connect()
    view = canary.inject_view(con, "SELECT * FROM _empty", _plant(size=30), name="_p")
    con.register("_empty", _plant(size=0).iloc[0:0])
    rows = coord2.hashtag_sequence_traces(con, view)
    assert len(coord2.min_activity(rows)) > 0


def test_inject_view_unions_in_memory_without_touching_the_background():
    con = duckdb.connect()
    background = _plant(size=3).assign(user_id="real", post_id=lambda f: "b" + f["post_id"])
    con.register("bg", background)
    view = canary.inject_view(con, "SELECT * FROM bg", _plant(size=4))
    got = con.sql(f"SELECT count(*), count(DISTINCT user_id) FROM ({view})").fetchone()
    assert got == (len(background) + len(_plant(size=4)), 5)
    assert con.sql("SELECT count(*) FROM bg").fetchone()[0] == len(background)


def test_writes_are_refused_outside_the_canary_prefixes():
    assert canary.posts_key("x").startswith("canary/")
    assert canary.expected_key("x", START).startswith("leads/platform=x/kind=canary_expected/")
    for bad in ("posts/platform=x/type=search/x.parquet", "authors/x.parquet", "canary-posts/x"):
        with pytest.raises(ValueError):
            canary._guard(bad)


def test_persist_writes_posts_and_the_expected_row_where_the_leads_contract_reads(tmp_path):
    con = duckdb.connect()
    p = canary.Plant("t1", _plant(size=4), [canary.account_id("t1", n) for n in range(4)],
                     START, START + pd.Timedelta(hours=canary.DEADLINE_HOURS))

    def uri(key):
        (tmp_path / key).parent.mkdir(parents=True, exist_ok=True)
        return str(tmp_path / key)

    keys = canary.persist(con, p, uri=uri)
    expected = pd.read_parquet(tmp_path / keys["expected"])
    assert list(expected["canary_id"]) == ["t1"]
    assert (expected["deadline"] - expected["injected_at"]).iloc[0] == pd.Timedelta(hours=36)
    assert len(canary.load_active(con, now=START, uri=uri)) == len(p.posts)
    assert canary.load_active(con, now=START + pd.Timedelta(hours=37), uri=uri).empty


def test_tags_already_in_the_background_stop_the_plant():
    con = duckdb.connect()
    con.register("bg", _plant(size=2))
    with pytest.raises(ValueError):
        canary.check_tags_unused(con, "SELECT * FROM bg")


def test_the_packet_carries_the_planted_evidence_to_the_reader():
    """B's gap: from the archive the canary's dossier is empty and the reader
    says `unclear`. From memory it carries the copy, the tags and the shared
    objects - the evidence a real operation would show."""
    posts = _plant(size=10)
    members = [canary.account_id("t1", n) for n in range(10)]
    packet = canary.packet(posts, members, cluster_id=7)
    assert packet["size"] == 10
    assert packet["representative_posts"] and packet["shared_texts"] and packet["shared_objects"]
    assert max(t["n_members"] for t in packet["shared_texts"]) >= 2
    rendered = adjudicate.render(packet)
    assert "Kenya" in rendered and "#" in rendered


def test_evaluate_finds_an_isolated_clique_that_the_global_score_misses():
    core = nx.complete_graph([f"r{i}" for i in range(40)])
    planted = [canary.account_id("t1", n) for n in range(6)]
    graph = nx.compose(core, nx.complete_graph(planted))
    graph.add_edge(planted[0], "r0")
    scores = coord2.centrality(graph)
    got = canary.evaluate(graph, scores, planted, budget=20, per_group=10)
    assert got["recall_top_budget"] == 0.0
    assert got["community_coverage"] == 1.0 and got["community_purity"] == 1.0
    assert got["listed"] and got["detected"] and got["isolated"]


def test_titles_and_ids_follow_the_leads_contract():
    assert canary.alert_title("2026-09-28") == "[CANARY] 2026-09-28"
    assert canary.alert_title("2026-09-28", "unclear") == "[CANARY] 2026-09-28: unclear"
    assert canary.canary_of("canary:2026-09-28:4") == "2026-09-28"
    assert canary.canary_of("12345") is None
