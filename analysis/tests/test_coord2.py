"""v2 coordination detection, checked against hand computation and structure.

Everything here runs offline against local fixtures: no R2, no Modal, no model
downloads. Embeddings are injected as arrays, which is also how the module is
meant to be used - the encoder is a caller's choice, not this module's.

The acceptance criteria from the A4 plan are covered explicitly: a hand-computed
TF-IDF cosine on a ten-user fixture, and fusion verified as a union rather than
an intersection.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from kma import coord2

BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _con(posts: pd.DataFrame) -> tuple[duckdb.DuckDBPyConnection, str]:
    con = duckdb.connect()
    con.register("posts", posts)
    return con, coord2.posts_view("posts")


def _post(
    post_id: str,
    user_id: str,
    *,
    at: datetime = BASE,
    repost_of: str | None = None,
    text: str = "",
    hashtags: list[str] | None = None,
    urls: list[str] | None = None,
) -> dict:
    return {
        "author_id": user_id,
        "platform_post_id": post_id,
        "created_at": at,
        "is_repost": repost_of is not None,
        "repost_of_id": repost_of,
        "text": text,
        "hashtags": hashtags or [],
        "urls": urls or [],
        "collected_at": BASE,
    }


# --------------------------------------------------------------------------
# The shared recipe
# --------------------------------------------------------------------------


def test_tfidf_cosine_matches_hand_computation_on_ten_users():
    """The A4 acceptance criterion, computed from the formula rather than the code.

    Ten users share one popular entity; two of them also share a rare one. With
    sklearn's smoothed IDF, ln((1 + n) / (1 + df)) + 1 over n = 10 users:
    the popular entity carries idf = 1, the rare one idf = ln(11/3) + 1.
    """
    rows = [{"user_id": f"u{i:02d}", "entity": "popular"} for i in range(10)]
    rows += [{"user_id": u, "entity": "rare"} for u in ("u00", "u01")]
    edges = coord2.similarity_network(pd.DataFrame(rows), min_entities=1)

    idf_rare = math.log((1 + 10) / (1 + 2)) + 1
    shared = np.array([1.0, idf_rare])
    shared /= np.linalg.norm(shared)
    popular_only = np.array([1.0, 0.0])

    pair = edges.set_index(["source", "target"])["weight"]
    assert pair[("u00", "u01")] == pytest.approx(float(shared @ shared))
    assert pair[("u00", "u02")] == pytest.approx(float(shared @ popular_only))
    # 45 pairs among ten users, all of which co-act on the popular entity.
    assert len(edges) == 45


def test_tfidf_downweights_popular_entities_rather_than_deleting_them():
    """v2 has no hub cap: a viral object still contributes, at a lower weight."""
    rows = [{"user_id": f"u{i}", "entity": "viral"} for i in range(50)]
    rows += [{"user_id": u, "entity": "niche"} for u in ("u0", "u1")]
    edges = coord2.similarity_network(pd.DataFrame(rows), min_entities=1).set_index(["source", "target"])["weight"]
    assert edges[("u0", "u1")] > edges[("u0", "u2")] > 0


def test_term_frequency_counts_repeated_actions():
    repeated = pd.DataFrame(
        {"user_id": ["u1", "u1", "u1", "u2"], "entity": ["e1", "e1", "e2", "e1"]}
    )
    once = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "entity": ["e1", "e2", "e1"]})
    assert (
        coord2.similarity_network(repeated, min_entities=1)["weight"][0]
        > coord2.similarity_network(once, min_entities=1)["weight"][0]
    )


def test_similarity_network_of_nothing_is_empty_not_an_error():
    empty = pd.DataFrame({"user_id": [], "entity": []})
    assert coord2.similarity_network(empty).empty
    assert coord2.similarity_network(pd.DataFrame({"user_id": ["u1"], "entity": ["e"]})).empty


def test_cosine_edges_chunking_does_not_change_the_result():
    rows = [{"user_id": f"u{i}", "entity": f"e{i % 4}"} for i in range(20)]
    bp = coord2.bipartite_tfidf(pd.DataFrame(rows))
    one = coord2.cosine_edges(bp, chunk=1000).sort_values(["source", "target"], ignore_index=True)
    many = coord2.cosine_edges(bp, chunk=3).sort_values(["source", "target"], ignore_index=True)
    pd.testing.assert_frame_equal(one, many)


def test_percentile_is_a_percentile_not_a_constant():
    """The whole point of the Q2 decision: the same shape scored in a space with
    different absolute values keeps the same edges."""
    edges = pd.DataFrame(
        {
            "source": list("aaaaa"),
            "target": list("bcdef"),
            "weight": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    kept = coord2.filter_percentile(edges, 80.0)
    assert kept["target"].tolist() == ["f"]

    compressed = edges.assign(weight=edges["weight"] * 0.01)
    assert coord2.filter_percentile(compressed, 80.0)["target"].tolist() == ["f"]


def test_percentile_none_keeps_every_edge():
    edges = pd.DataFrame({"source": ["a"], "target": ["b"], "weight": [0.01]})
    assert len(coord2.filter_percentile(edges, None)) == 1


def test_fast_retweet_has_no_edge_percentile():
    """Table 2 sweeps a time interval for this trace, not a similarity percentile."""
    assert coord2.EDGE_PERCENTILE["fast_retweet"] is None
    assert coord2.EDGE_PERCENTILE["co_retweet"] == 80.0
    assert coord2.EDGE_PERCENTILE["co_url"] == 80.0
    assert coord2.EDGE_PERCENTILE["hashtag_sequence"] == 65.0
    assert coord2.EDGE_PERCENTILE["text_similarity"] == 96.0
    assert coord2.FAST_RETWEET_SECONDS == 60
    assert coord2.CENTRALITY_THRESHOLD == 1e-2


# --------------------------------------------------------------------------
# Entity extractors
# --------------------------------------------------------------------------


def test_co_retweet_entity_is_the_retweeted_tweet():
    con, view = _con(
        pd.DataFrame(
            [
                _post("p1", "author"),
                _post("p2", "u1", repost_of="p1"),
                _post("p3", "u2", repost_of="p1"),
                _post("p4", "u3", text="an original post"),
            ]
        )
    )
    traces = coord2.co_retweet_traces(con, view)
    assert set(map(tuple, traces.to_numpy())) == {("u1", "p1"), ("u2", "p1")}


def test_co_url_entity_is_the_url_as_shared():
    """Shorteners are not resolved: two shortlinks to one destination are two
    entities, which is the paper's definition rather than an oversight."""
    con, view = _con(
        pd.DataFrame(
            [
                _post("p1", "u1", urls=["https://example.org/a", "https://example.org/b"]),
                _post("p2", "u2", urls=["https://bit.ly/xyz"]),
            ]
        )
    )
    traces = coord2.co_url_traces(con, view)
    assert sorted(traces["entity"]) == [
        "https://bit.ly/xyz",
        "https://example.org/a",
        "https://example.org/b",
    ]
    assert coord2.similarity_network(traces).empty


def test_hashtag_sequence_needs_at_least_three_tags():
    con, view = _con(
        pd.DataFrame(
            [
                _post("p1", "u1", hashtags=["ruto", "raila"]),
                _post("p2", "u2", hashtags=["ruto", "raila", "iebc"]),
            ]
        )
    )
    traces = coord2.hashtag_sequence_traces(con, view)
    assert traces["user_id"].tolist() == ["u2"]
    assert traces["entity"].tolist() == ["ruto|raila|iebc"]
    assert sorted(coord2.hashtag_sequence_traces(con, view, min_hashtags=2)["user_id"]) == [
        "u1",
        "u2",
    ]


def test_hashtag_sequence_is_a_sequence_not_a_set():
    """Same three tags, different order: not the same entity, so no edge."""
    con, view = _con(
        pd.DataFrame(
            [
                _post("p1", "u1", hashtags=["ruto", "raila", "iebc"]),
                _post("p2", "u2", hashtags=["iebc", "raila", "ruto"]),
                _post("p3", "u3", hashtags=["RUTO", "Raila", "IEBC"]),
            ]
        )
    )
    traces = coord2.hashtag_sequence_traces(con, view)
    assert traces.set_index("user_id")["entity"].to_dict() == {
        "u1": "ruto|raila|iebc",
        "u2": "iebc|raila|ruto",
        "u3": "ruto|raila|iebc",
    }
    edges = coord2.similarity_network(traces, min_entities=1)
    assert set(zip(edges["source"], edges["target"], strict=True)) == {("u1", "u3")}


def test_hashtag_sequence_from_text_recovers_in_tweet_order():
    """The persisted list drops X's `indices`, so the text is the only place the
    order can be re-derived from - see the module docstring."""
    con, view = _con(
        pd.DataFrame(
            [_post("p1", "u1", text="vote #Ruto then #Raila then #IEBC", hashtags=["iebc", "raila", "ruto"])]
        )
    )
    assert coord2.hashtag_sequence_traces(con, view, order="text")["entity"].tolist() == [
        "ruto|raila|iebc"
    ]
    with pytest.raises(ValueError, match="order must be"):
        coord2.hashtag_sequence_traces(con, view, order="indices")


def test_hashtags_from_text_keeps_duplicates_and_order():
    assert coord2.hashtags_from_text("#a b #c #a") == ["a", "c", "a"]
    assert coord2.hashtags_from_text(None) == []


def test_fast_retweet_is_bounded_at_sixty_seconds():
    con, view = _con(
        pd.DataFrame(
            [
                _post("p0", "target"),
                _post("p1", "onTime", at=BASE + timedelta(seconds=60), repost_of="p0"),
                _post("p2", "tooLate", at=BASE + timedelta(seconds=61), repost_of="p0"),
                _post("p3", "beforeIt", at=BASE - timedelta(seconds=5), repost_of="p0"),
            ]
        )
    )
    traces = coord2.fast_retweet_traces(con, view)
    assert traces["user_id"].tolist() == ["onTime"]
    assert traces["entity"].tolist() == ["target"], "entity is the retweeted AUTHOR"
    assert len(coord2.fast_retweet_traces(con, view, seconds=61)) == 2


def test_fast_retweet_coverage_reports_retweets_it_cannot_time():
    """`engagements/` carries no retweet time and the archive may not hold the
    original, so the untimeable share is a reportable property, not an error."""
    con, view = _con(
        pd.DataFrame(
            [
                _post("p0", "target"),
                _post("p1", "u1", at=BASE + timedelta(seconds=10), repost_of="p0"),
                _post("p2", "u2", at=BASE + timedelta(seconds=10), repost_of="missing"),
            ]
        )
    )
    coverage = coord2.fast_retweet_coverage(con, view).iloc[0]
    assert (coverage["retweets"], coverage["timeable"], coverage["fast"]) == (2, 1, 1)


def test_posts_view_takes_the_latest_row_per_post():
    posts = pd.DataFrame([_post("p1", "u1", text="first"), _post("p1", "u1", text="second")])
    posts.loc[1, "collected_at"] = BASE + timedelta(hours=1)
    con, view = _con(posts)
    assert con.sql(f"SELECT text FROM ({view})").df()["text"].tolist() == ["second"]


def test_ioa_view_parses_the_archive_shape():
    """The mirror stores list columns as `['a', 'b']` text and ids as strings."""
    con = duckdb.connect()
    con.register(
        "ioa",
        pd.DataFrame(
            {
                "tweetid": ["1", "2"],
                "userid": ["hashedaaa", "999"],
                "tweet_time": ["2018-06-15 09:00", "2018-06-15 09:00"],
                "is_retweet": ["false", "true"],
                "retweet_tweetid": ["", "1"],
                "retweet_userid": ["", "hashedaaa"],
                "tweet_text": ["a post", "RT a post"],
                "hashtags": ["['WorldCup2018', 'CR7']", "[]"],
                "urls": ["['https://example.org/a']", ""],
            }
        ),
    )
    got = con.sql(f"SELECT * FROM ({coord2.ioa_view('ignored', read='ioa')})").df()
    assert got["hashtags"].tolist()[0].tolist() == ["WorldCup2018", "CR7"]
    assert got["hashtags"].tolist()[1].tolist() == []
    assert got["urls"].tolist()[1].tolist() == []
    assert got["is_retweet"].tolist() == [False, True]
    assert pd.isna(got["retweet_post_id"][0]) and got["retweet_post_id"][1] == "1"


# --------------------------------------------------------------------------
# Text similarity
# --------------------------------------------------------------------------


def test_clean_text_strips_urls_emoji_punctuation_and_stopwords():
    cleaned = coord2.clean_text("The IEBC, https://x.com/a, is not ready! 🔥🔥")
    assert "http" not in cleaned and "🔥" not in cleaned and "," not in cleaned
    assert "the" not in cleaned.split()
    assert "iebc" in cleaned.split()
    assert coord2.clean_text(None) == ""


def test_text_rows_drop_retweets_and_texts_under_four_words():
    con, view = _con(
        pd.DataFrame(
            [
                _post("p1", "u1", text="the electoral commission rigged tallying centres nationwide"),
                _post("p2", "u2", text="short one", repost_of=None),
                _post("p3", "u3", text="electoral commission rigged tallying centres", repost_of="p1"),
            ]
        )
    )
    rows = coord2.text_rows(con, view)
    assert rows["post_id"].tolist() == ["p1"]


def test_cosine_pairs_stay_inside_the_sliding_window():
    vectors = np.eye(3)
    times = np.array([0.0, 10.0, 1000.0])
    pairs = {
        (int(i), int(j))
        for a, b, _ in coord2.cosine_pairs(vectors, times, window_seconds=100.0)
        for i, j in zip(a, b, strict=True)
    }
    assert pairs == {(0, 1)}
    unbounded = {
        (int(i), int(j))
        for a, b, _ in coord2.cosine_pairs(vectors, times)
        for i, j in zip(a, b, strict=True)
    }
    assert unbounded == {(0, 1), (0, 2), (1, 2)}


def test_cosine_pairs_chunking_does_not_change_the_result():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(9, 4))
    times = np.arange(9, dtype="float64")

    def collect(chunk):
        return {
            (int(i), int(j)): round(float(s), 9)
            for a, b, sims in coord2.cosine_pairs(vectors, times, window_seconds=3.0, chunk=chunk)
            for i, j, s in zip(a, b, sims, strict=True)
        }

    assert collect(2) == collect(100)


def _text_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    """Two users posting the same thing, and a third posting something else."""
    rows = pd.DataFrame(
        {
            "post_id": ["p1", "p2", "p3", "p4"],
            "user_id": ["u1", "u2", "u3", "u1"],
            "created_at": [BASE, BASE + timedelta(days=1), BASE, BASE + timedelta(days=2)],
        }
    )
    vectors = np.array(
        [[1.0, 0.0], [0.99, 0.141], [0.0, 1.0], [0.96, 0.28]],
        dtype="float64",
    )
    return rows, vectors


def test_text_similarity_links_users_and_weights_by_the_mean():
    rows, vectors = _text_fixture()
    edges = coord2.text_similarity_network(rows, vectors, threshold=0.9)
    assert set(zip(edges["source"], edges["target"], strict=True)) == {("u1", "u2")}

    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    expected = float(np.mean([unit[0] @ unit[1], unit[3] @ unit[1]]))
    assert edges["weight"].iloc[0] == pytest.approx(expected)


def test_text_similarity_ignores_a_user_matching_themselves():
    rows = pd.DataFrame(
        {
            "post_id": ["p1", "p2"],
            "user_id": ["u1", "u1"],
            "created_at": [BASE, BASE + timedelta(days=1)],
        }
    )
    assert coord2.text_similarity_network(rows, np.array([[1.0, 0.0], [1.0, 0.0]]), threshold=0.5).empty


def test_text_similarity_window_excludes_pairs_more_than_a_year_apart():
    rows, vectors = _text_fixture()
    rows.loc[1, "created_at"] = BASE + timedelta(days=400)
    assert coord2.text_similarity_network(rows, vectors, threshold=0.9).empty
    assert not coord2.text_similarity_network(
        rows, vectors, threshold=0.9, window_days=500
    ).empty


def test_text_similarity_threshold_is_derived_from_the_observed_distribution():
    """A percentile, not the paper's 0.95: the threshold has to be re-derived
    per embedding space, because a cosine value does not port between spaces."""
    rng = np.random.default_rng(1)
    n = 40
    rows = pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(n)],
            "user_id": [f"u{i}" for i in range(n)],
            "created_at": [BASE + timedelta(hours=i) for i in range(n)],
        }
    )
    vectors = rng.normal(size=(n, 8))
    times = np.array([(BASE + timedelta(hours=i)).timestamp() for i in range(n)])

    threshold = coord2.pair_similarity_percentile(
        vectors, times, 96.0, window_seconds=coord2.TEXT_WINDOW_DAYS * 86400
    )
    sims = np.concatenate(
        [s for _, _, s in coord2.cosine_pairs(vectors, times, window_seconds=1e12)]
    )
    assert threshold == pytest.approx(float(np.percentile(sims, 96.0)))

    edges = coord2.text_similarity_network(rows, vectors, percentile=96.0)
    assert len(edges) == int((sims >= threshold).sum())
    assert len(edges) < len(sims) / 10

    with pytest.raises(ValueError, match="percentile or a threshold"):
        coord2.text_similarity_network(rows, vectors, percentile=None)


def test_pair_similarity_percentile_samples_rows_when_the_input_is_large():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(300, 4))
    times = np.arange(300, dtype="float64")
    sampled = coord2.pair_similarity_percentile(
        vectors, times, 90.0, window_seconds=1e9, max_rows=60
    )
    exact = coord2.pair_similarity_percentile(vectors, times, 90.0, window_seconds=1e9)
    assert sampled == pytest.approx(exact, abs=0.1)


# --------------------------------------------------------------------------
# Fusion and unsupervised detection
# --------------------------------------------------------------------------


def _edges(*pairs: tuple[str, str], weight: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"source": [a for a, _ in pairs], "target": [b for _, b in pairs], "weight": weight}
    )


def test_fusion_is_a_union_not_an_intersection():
    """The A4 acceptance criterion. An edge in ANY network is an edge in the
    fused network, so a pair seen once survives."""
    networks = {
        "co_retweet": _edges(("u1", "u2"), ("u2", "u3")),
        "co_url": _edges(("u1", "u2"), ("u4", "u5")),
    }
    fused = coord2.fuse(networks)
    assert set(map(frozenset, fused.edges)) == {
        frozenset(p) for p in [("u1", "u2"), ("u2", "u3"), ("u4", "u5")]
    }
    assert fused["u1"]["u2"]["traces"] == {"co_retweet", "co_url"}
    assert fused["u4"]["u5"]["traces"] == {"co_url"}


def test_fusion_of_nothing_is_an_empty_graph():
    assert coord2.fuse({"co_url": _edges()}).number_of_nodes() == 0


def test_centrality_scores_nodes_absent_from_every_network_zero():
    fused = coord2.fuse({"co_retweet": _edges(("u1", "u2"), ("u2", "u3"))})
    scores = coord2.centrality(fused, nodes=["u1", "u2", "u3", "never_seen"])
    assert scores["never_seen"] == 0.0
    assert scores["u2"] > scores["u1"] > 0


def test_centrality_ignores_edge_weights():
    """Weighted centrality did not improve the paper's performance, so weights
    must not reach the centrality computation."""
    light = coord2.fuse({"a": _edges(("u1", "u2"), ("u2", "u3"), weight=0.01)})
    heavy = coord2.fuse({"a": _edges(("u1", "u2"), ("u2", "u3"), weight=100.0)})
    pd.testing.assert_series_equal(coord2.centrality(light), coord2.centrality(heavy))


def test_centrality_handles_a_two_node_graph():
    scores = coord2.centrality(coord2.fuse({"a": _edges(("u1", "u2"))}))
    assert scores["u1"] == pytest.approx(scores["u2"])
    assert scores["u1"] == pytest.approx(1 / math.sqrt(2))


def test_centrality_of_an_edgeless_graph_is_zero():
    graph = nx.Graph()
    graph.add_nodes_from(["u1", "u2"])
    assert coord2.centrality(graph).tolist() == [0.0, 0.0]


def test_detect_predicts_accounts_and_prunes_below_the_threshold():
    """A dense core plus a barely-connected pair: the core survives pruning."""
    core = [(f"c{i}", f"c{j}") for i in range(6) for j in range(i + 1, 6)]
    networks = {"co_retweet": _edges(*core), "co_url": _edges(("x1", "x2"))}
    found = coord2.detect(networks, nodes=[f"c{i}" for i in range(6)] + ["x1", "x2", "lonely"])

    assert list(found.columns) == ["user_id", "centrality", "predicted"]
    predicted = set(found.loc[found["predicted"], "user_id"])
    assert predicted == {f"c{i}" for i in range(6)}
    assert found.set_index("user_id").loc["lonely", "centrality"] == 0.0


def test_prune_returns_the_surviving_accounts():
    scores = pd.Series({"u1": 0.5, "u2": 0.001})
    assert coord2.prune(scores).tolist() == ["u1"]
    assert coord2.prune(scores, threshold=0.0).tolist() == ["u1", "u2"]


# --------------------------------------------------------------------------
# Supervised detection
# --------------------------------------------------------------------------


def _two_cliques(size: int = 8) -> nx.Graph:
    graph = nx.Graph()
    for prefix in ("a", "b"):
        for i in range(size):
            for j in range(i + 1, size):
                graph.add_edge(f"{prefix}{i}", f"{prefix}{j}")
    graph.add_edge("a0", "b0")
    return graph


def test_random_walks_have_the_paper_s_shape_and_stay_on_edges():
    graph = _two_cliques(4)
    walks = coord2.random_walks(graph, walks=16, length=16, seed=0)
    assert len(walks) == 16 * graph.number_of_nodes()
    assert {len(w) for w in walks} == {16}
    for walk in walks:
        for a, b in zip(walk, walk[1:], strict=False):
            assert graph.has_edge(a, b)


def test_random_walks_stop_at_an_isolated_node():
    graph = nx.Graph()
    graph.add_node("alone")
    assert coord2.random_walks(graph, walks=2, length=16) == [["alone"], ["alone"]]


def test_random_walks_are_deterministic_under_a_seed():
    graph = _two_cliques(4)
    assert coord2.random_walks(graph, seed=7) == coord2.random_walks(graph, seed=7)
    assert coord2.random_walks(graph, seed=7) != coord2.random_walks(graph, seed=8)


def test_node2vec_embeds_every_node():
    graph = _two_cliques(4)
    embedding = coord2.node2vec(graph, dim=16, walks=4, length=8, epochs=1, seed=0)
    assert embedding.shape == (graph.number_of_nodes(), 16)
    assert set(embedding.index) == set(graph.nodes)


def test_node2vec_defaults_are_the_paper_s():
    assert (coord2.NODE2VEC_DIM, coord2.NODE2VEC_WALKS, coord2.NODE2VEC_LENGTH) == (128, 16, 16)


def test_node2vec_embeddings_separate_two_communities():
    graph = _two_cliques(8)
    embedding = coord2.node2vec(graph, dim=32, walks=16, length=16, epochs=5, seed=0)
    labels = pd.Series({n: int(n.startswith("a")) for n in graph.nodes})
    metrics, oof = coord2.classify(embedding, labels, folds=4, seed=0)
    assert metrics["auc"] > 0.8
    assert set(oof.index) == set(graph.nodes)


def test_classify_drops_accounts_with_no_label():
    """An unlabelled account in the IO archive is unknown, never a negative."""
    graph = _two_cliques(6)
    embedding = coord2.node2vec(graph, dim=8, walks=4, length=8, epochs=1, seed=0)
    labels = pd.Series({n: int(n.startswith("a")) for n in graph.nodes if n != "b5"})
    _, oof = coord2.classify(embedding, labels, folds=3, seed=0)
    assert "b5" not in oof.index


def test_classify_needs_both_classes():
    graph = _two_cliques(3)
    embedding = coord2.node2vec(graph, dim=8, walks=2, length=4, epochs=1, seed=0)
    labels = pd.Series({n: 1 for n in graph.nodes})
    with pytest.raises(ValueError, match="two examples of each class"):
        coord2.classify(embedding, labels)


def test_classification_metrics_reads_scores_and_a_threshold():
    metrics = coord2.classification_metrics([1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1], threshold=0.5)
    assert metrics == pytest.approx({"auc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0})
    lenient = coord2.classification_metrics([1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1], threshold=0.2)
    assert lenient["precision"] == pytest.approx(2 / 3)
    assert lenient["recall"] == 1.0


def test_unsupervised_metrics_score_centrality_directly():
    """The unsupervised half is scored on centrality as a ranking, with pruning
    as the decision rule - the same metric function, no cluster in sight."""
    core = [(f"c{i}", f"c{j}") for i in range(5) for j in range(i + 1, 5)]
    found = coord2.detect(
        {"co_retweet": _edges(*core)}, nodes=[f"c{i}" for i in range(5)] + ["organic"]
    )
    labels = [1] * 5 + [0]
    metrics = coord2.classification_metrics(
        labels, found["centrality"], threshold=coord2.CENTRALITY_THRESHOLD
    )
    assert metrics == pytest.approx({"auc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0})


# --------------------------------------------------------------------------
# End to end, on posts rather than on hand-built trace rows
# --------------------------------------------------------------------------


def test_five_traces_fuse_into_one_account_level_verdict():
    """The ring acts on TWO objects per trace, not one.

    A ring sharing a single object is indistinguishable from a viral tweet's
    audience - see MIN_ENTITIES_PER_USER - so a fixture built that way would be
    asserting behaviour the method deliberately no longer has.
    """
    # Two DIFFERENT target authors: fast_retweet's entity is the retweeted
    # author, so two posts by one author is a single entity there and the
    # activity floor would drop that trace.
    posts = [
        _post("orig", "target", text="the original claim about the commission"),
        _post("orig2", "target2", text="a second claim from another account"),
    ]
    for i in range(4):
        for j, parent in enumerate(("orig", "orig2")):
            posts.append(
                _post(
                    f"rt{i}_{j}",
                    f"ring{i}",
                    at=BASE + timedelta(seconds=5 + j),
                    repost_of=parent,
                    text="RT",
                )
            )
        posts.append(
            _post(
                f"tag{i}",
                f"ring{i}",
                at=BASE + timedelta(minutes=i),
                text="stop the steal now #iebc #ruto #raila",
                hashtags=["iebc", "ruto", "raila"],
                urls=["https://example.org/dossier"],
            )
        )
        posts.append(
            _post(
                f"tag2{i}",
                f"ring{i}",
                at=BASE + timedelta(minutes=30 + i),
                text="again the same line #iebc #raila #ruto now",
                hashtags=["iebc", "raila", "ruto"],
                urls=["https://example.org/second"],
            )
        )
    posts.append(_post("solo", "organic", text="watching the news tonight", hashtags=["news"]))
    con, view = _con(pd.DataFrame(posts))

    networks = {
        "co_retweet": coord2.similarity_network(
            coord2.co_retweet_traces(con, view), percentile=coord2.EDGE_PERCENTILE["co_retweet"]
        ),
        "co_url": coord2.similarity_network(
            coord2.co_url_traces(con, view), percentile=coord2.EDGE_PERCENTILE["co_url"]
        ),
        "hashtag_sequence": coord2.similarity_network(
            coord2.hashtag_sequence_traces(con, view),
            percentile=coord2.EDGE_PERCENTILE["hashtag_sequence"],
        ),
        "fast_retweet": coord2.similarity_network(
            coord2.fast_retweet_traces(con, view),
            percentile=coord2.EDGE_PERCENTILE["fast_retweet"],
        ),
    }
    assert all(not edges.empty for edges in networks.values())

    population = con.sql(f"SELECT DISTINCT user_id FROM ({view})").df()["user_id"].tolist()
    found = coord2.detect(networks, nodes=population).set_index("user_id")
    assert found.loc[[f"ring{i}" for i in range(4)], "predicted"].all()
    assert not found.loc["organic", "predicted"]


def test_min_activity_drops_single_entity_users():
    """A user with one action cannot be coordinated with anyone: their one-hot
    TF-IDF vector is identical to every other single-actor on that entity, so
    cosine is 1.0 and a viral object becomes a clique."""
    traces = pd.DataFrame(
        {
            "user_id": ["a", "a", "b", "b", "c"],
            "entity": ["x", "y", "x", "y", "x"],
        }
    )
    kept = coord2.min_activity(traces, 2)
    assert set(kept["user_id"]) == {"a", "b"}


def test_min_activity_of_one_is_a_no_op():
    traces = pd.DataFrame({"user_id": ["a", "b"], "entity": ["x", "x"]})
    assert len(coord2.min_activity(traces, 1)) == 2


def test_similarity_network_breaks_the_single_object_clique():
    """Ten users sharing exactly one viral object form a complete graph at
    min_entities=1 and vanish at the default floor."""
    viral = pd.DataFrame(
        {"user_id": [f"u{i}" for i in range(10)], "entity": ["viral"] * 10}
    )
    wide_open = coord2.similarity_network(viral, min_entities=1)
    assert len(wide_open) == 45  # complete graph on 10 nodes

    filtered = coord2.similarity_network(viral)
    assert filtered.empty


def test_min_activity_keeps_genuine_co_action():
    """The floor must not remove real coordination: users sharing two objects
    survive it."""
    traces = pd.DataFrame(
        {
            "user_id": ["a", "a", "b", "b", "noise"],
            "entity": ["x", "y", "x", "y", "z"],
        }
    )
    edges = coord2.similarity_network(traces)
    assert len(edges) == 1
    assert {edges.iloc[0]["source"], edges.iloc[0]["target"]} == {"a", "b"}


def test_gpu_cosine_pairs_matches_the_cpu_implementation():
    """Threshold pushdown must change only WHICH pairs are yielded, never the
    similarity values or the indices they carry."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(40, 8))
    times = np.arange(40, dtype="float64") * 60.0

    cpu = {}
    for i, j, sim in coord2.cosine_pairs(vectors, times, window_seconds=1e9, chunk=7):
        for a, b, s in zip(i, j, sim, strict=True):
            cpu[(int(a), int(b))] = float(s)

    gpu = {}
    for i, j, sim in coord2.gpu_cosine_pairs(0.3, device="cpu")(
        vectors, times, window_seconds=1e9, chunk=7
    ):
        for a, b, s in zip(i, j, sim, strict=True):
            gpu[(int(a), int(b))] = float(s)

    assert set(gpu) == {k for k, v in cpu.items() if v >= 0.3}
    for key, value in gpu.items():
        assert value == pytest.approx(cpu[key], abs=1e-5)


def test_gpu_cosine_pairs_respects_the_window():
    vectors = np.ones((6, 4))
    times = np.array([0.0, 10.0, 20.0, 1000.0, 1010.0, 1020.0])
    pairs = list(coord2.gpu_cosine_pairs(0.0, device="cpu")(vectors, times, window_seconds=100.0, chunk=3))
    seen = {(int(a), int(b)) for i, j, _ in pairs for a, b in zip(i, j, strict=True)}
    assert (0, 3) not in seen
    assert (0, 1) in seen


def test_text_rows_is_deterministically_ordered():
    """A bounded read must be the same bounded read next time: an unordered
    query turned one threshold sweep into a comparison of three corpora."""
    posts = pd.DataFrame(
        [
            _post(f"p{i:03d}", f"u{i % 7}", text=f"a genuine sentence number {i} here")
            for i in range(60)
        ]
    )
    con, view = _con(posts)
    first = coord2.text_rows(con, view)
    second = coord2.text_rows(con, view)
    assert first["post_id"].tolist() == second["post_id"].tolist()
    assert first["post_id"].tolist() == sorted(first["post_id"])
