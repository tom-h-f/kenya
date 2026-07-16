"""Unit tests for kma.coordination (no R2 required)."""

import numpy as np
import pandas as pd
import pytest

from kma import coordination as co


def test_validate_svn_fdr_and_bonferroni():
    edges = pd.DataFrame(
        {
            "src": ["a", "a", "b"],
            "dst": ["b", "c", "c"],
            "n_objects_shared": [5, 2, 4],
            "weight": [5, 2, 4],
        }
    )
    degrees = pd.DataFrame(
        {"author_id": ["a", "b", "c"], "n_objects": [10, 8, 6]}
    )
    out = co.validate_svn(edges, degrees, n_objects=50, method="fdr_bh", alpha=0.05)
    assert "p_value" in out.columns
    assert out["validated"].dtype == bool
    bonf = co.validate_svn(edges, degrees, 50, method="bonferroni", alpha=0.05)
    assert bonf["validated"].sum() <= out["validated"].sum()


def test_validate_svn_empty():
    out = co.validate_svn(
        pd.DataFrame(columns=["src", "dst", "n_objects_shared", "weight"]),
        pd.DataFrame(columns=["author_id", "n_objects"]),
        0,
    )
    assert out.empty


def test_filter_clustering_posts_drops_retweets_and_dupes():
    df = pd.DataFrame(
        [
            {
                "author_id": "a",
                "text": "iebc rigged",
                "is_repost": False,
                "created_at": pd.Timestamp("2026-07-08", tz="UTC"),
                "platform_post_id": "p1",
            },
            {
                "author_id": "a",
                "text": "iebc rigged",
                "is_repost": False,
                "created_at": pd.Timestamp("2026-07-09", tz="UTC"),
                "platform_post_id": "p2",
            },
            {
                "author_id": "b",
                "text": "RT @x: iebc rigged",
                "is_repost": False,
                "created_at": pd.Timestamp("2026-07-08", tz="UTC"),
                "platform_post_id": "p3",
            },
            {
                "author_id": "c",
                "text": "iebc rigged",
                "is_repost": True,
                "created_at": pd.Timestamp("2026-07-08", tz="UTC"),
                "platform_post_id": "p4",
            },
            {
                "author_id": "d",
                "text": "different claim",
                "is_repost": False,
                "created_at": pd.Timestamp("2026-07-08", tz="UTC"),
                "platform_post_id": "p5",
            },
        ]
    )
    out = co._filter_clustering_posts(df)
    assert set(out["platform_post_id"]) == {"p1", "p5"}


def test_is_manual_retweet():
    assert co._is_manual_retweet("RT @NationAfrica: headline here")
    assert co._is_manual_retweet("  rt @foo: bar")
    assert not co._is_manual_retweet("the RT meaning is retweet")
    assert not co._is_manual_retweet("original post")


def test_percentile_filter():
    edges = pd.DataFrame({"src": ["a", "b", "c"], "dst": ["b", "c", "d"], "weight": [1, 5, 10]})
    out = co.percentile_filter(edges, q=0.667)
    assert out["validated"].all()
    assert out["weight"].min() >= 5


def test_evaluate_recovery_perfect():
    syn = [f"s{i}" for i in range(10)]
    members = pd.DataFrame(
        {"author_id": syn + ["x", "y"], "cluster_id": [0] * 10 + [1, 1]}
    )
    rec = co.evaluate_recovery(members, syn)
    assert rec["precision"] == 1.0
    assert rec["recall"] == 1.0
    assert rec["f1"] == 1.0


def test_evaluate_recovery_empty():
    rec = co.evaluate_recovery(pd.DataFrame(columns=["author_id", "cluster_id"]), ["s0"])
    assert rec["recall"] == 0.0


def test_edge_report_jaccard():
    edges = pd.DataFrame(
        {
            "src": ["a", "a"],
            "dst": ["b", "c"],
            "sig_bonferroni": [True, False],
            "sig_fdr": [True, True],
            "sig_percentile": [True, True],
        }
    )
    report = co.edge_report(edges)
    assert set(report["method"]) >= {"bonferroni", "fdr", "percentile", "jaccard(bonferroni,fdr)"}


def test_codelta_counts():
    tr = pd.DataFrame(
        {
            "author_id": ["a", "b", "c", "a"],
            "action_object": ["x", "x", "x", "x"],
            "created_at": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:30",
                    "2026-01-01 00:01:00",
                    "2026-01-01 00:02:00",
                ]
            ),
        }
    )
    obj_codes = tr["action_object"].astype("category").cat.codes.to_numpy()
    order = np.argsort(obj_codes, kind="stable")
    bounds = np.flatnonzero(np.diff(obj_codes[order])) + 1
    groups = [g for g in np.split(order, bounds) if len(g) >= 2]
    authors = tr["author_id"].to_numpy()
    times = (tr["created_at"].astype("int64") // 10**9).to_numpy()
    counts = co._codelta_counts(groups, authors, times, delta=60)
    assert counts[("a", "b")] >= 1
    assert counts[("b", "c")] >= 1


def test_aggregate_layers_and_corroborate():
    layers = {
        "a": pd.DataFrame({"src": ["1"], "dst": ["2"], "weight": [3], "min_gap": [1.0]}),
        "b": pd.DataFrame({"src": ["1"], "dst": ["2"], "weight": [6], "min_gap": [2.0]}),
    }
    agg = co.aggregate_layers(layers)
    assert agg.loc[0, "n_channels"] == 2
    corr = co.corroborate(layers)
    assert corr.loc[0, "n_channels"] == 2


def test_cluster_names_from_member_text():
    import duckdb

    members = pd.DataFrame(
        {
            "author_id": ["a1", "a2", "b1", "b2"],
            "cluster_id": [0, 0, 1, 1],
        }
    )
    summary = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "size": [2, 2],
            "channels": [["co_retweet"], ["text_sim"]],
            "n_channels": [1, 1],
        }
    )
    posts = pd.DataFrame(
        {
            "author_id": ["a1", "a2", "b1", "b2"],
            "text": [
                "IEBC cannot be trusted in Kenya elections",
                "IEBC rigging claims spread online",
                "World Cup football match highlights",
                "FIFA football world cup final",
            ],
        }
    )
    con = duckdb.connect()
    con.register("_posts", posts)

    names = co.cluster_names(con, members, summary, posts_view="_posts")
    assert len(names) == 2
    assert "name" in names.columns
    assert "label" in names.columns
    assert all(names["name"].str.split().str.len() <= 3)
    assert "iebc" in names.loc[names["cluster_id"] == 0, "name"].iloc[0].lower()
    assert "label" in names.columns
    assert "(n=2)" in names["label"].iloc[0]


def test_cluster_names_channel_fallback():
    import duckdb

    members = pd.DataFrame({"author_id": ["a1"], "cluster_id": [3]})
    summary = pd.DataFrame(
        {
            "cluster_id": [3],
            "size": [1],
            "channels": [["co_retweet", "text_sim"]],
            "n_channels": [2],
        }
    )
    posts = pd.DataFrame({"author_id": ["a1"], "text": [""]})
    con = duckdb.connect()
    con.register("_posts", posts)

    names = co.cluster_names(con, members, summary, posts_view="_posts")
    assert names.loc[0, "name"] == "multi-signal"


def test_with_cluster_names_adds_display_columns():
    df = pd.DataFrame({"cluster_id": [0, 1], "size": [5, 3]})
    names = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "name": ["iebc rigging", "world cup"],
            "label": ["iebc rigging (n=5)", "world cup (n=3)"],
        }
    )
    out = co.with_cluster_names(df, names)
    assert list(out["name"]) == ["iebc rigging", "world cup"]
    assert list(out["label"]) == ["iebc rigging (n=5)", "world cup (n=3)"]


def _incidence_frames(rows):
    """rows = [(author, object)] -> (edges min_rep=1, degrees, M, object_degrees)."""
    df = pd.DataFrame(rows, columns=["author_id", "action_object"]).drop_duplicates()
    pairs = {}
    for obj, grp in df.groupby("action_object"):
        users = sorted(grp["author_id"])
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                pairs[(users[i], users[j])] = pairs.get((users[i], users[j]), 0) + 1
    edges = pd.DataFrame(
        {
            "src": [p[0] for p in pairs],
            "dst": [p[1] for p in pairs],
            "n_objects_shared": list(pairs.values()),
            "weight": list(pairs.values()),
        }
    )
    degrees = df.groupby("author_id")["action_object"].nunique().rename("n_objects")
    degrees = degrees.reset_index()
    obj_deg = df.groupby("action_object")["author_id"].nunique()
    return edges, degrees, df["action_object"].nunique(), obj_deg


def test_degree_corrected_svn_ignores_popular_object_chance_overlap():
    """Two viral objects with big random audiences: the uniform hypergeometric
    FDR flags chance co-retweeters (the failure seen live 2026-07-08); the
    configuration-model null must not."""
    rng = np.random.default_rng(7)
    users = [f"u{i}" for i in range(200)]
    rows = []
    for obj in ("viral1", "viral2"):
        for u in rng.choice(users, 60, replace=False):
            rows.append((u, obj))
    for i, u in enumerate(users):  # long tail of degree-1 niche objects
        rows.append((u, f"niche{i}"))
    edges, degrees, m, obj_deg = _incidence_frames(rows)

    uniform = co.validate_svn(edges, degrees, m, "fdr_bh", alpha=0.01)
    corrected = co.validate_svn(
        edges, degrees, m, "fdr_bh", alpha=0.01, object_degrees=obj_deg
    )
    assert uniform["validated"].sum() > 100  # the failure mode
    assert corrected["validated"].sum() == 0  # the fix


def test_degree_corrected_svn_still_recovers_planted_cluster():
    rng = np.random.default_rng(7)
    users = [f"u{i}" for i in range(200)]
    rows = []
    for obj in ("viral1", "viral2"):
        for u in rng.choice(users, 40, replace=False):
            rows.append((u, obj))
    for i, u in enumerate(users):  # organic background activity
        for k in range(3):
            rows.append((u, f"niche{i}_{k}"))
    bots = [f"bot{i}" for i in range(10)]
    for b in bots:  # planted cluster: 10 accounts sharing 6 seed objects
        for s in range(6):
            rows.append((b, f"seed{s}"))
    edges, degrees, m, obj_deg = _incidence_frames(rows)
    out = co.validate_svn(edges, degrees, m, "fdr_bh", alpha=0.01, object_degrees=obj_deg)
    hits = out[out["validated"]]
    bot_pairs = {(a, b) for a in bots for b in bots if a < b}
    found = set(zip(hits["src"], hits["dst"]))
    assert bot_pairs <= found  # every planted pair validated
    assert found == bot_pairs  # and nothing else


def test_story_account_set_members_and_amplifiers():
    story = pd.DataFrame({"author_id": ["a", "b", "a"]})
    amps = pd.DataFrame({"platform_user_id": ["c"], "kind": ["retweet"], "n": [2]})
    assert co.story_account_set(story, amps) == {"a", "b", "c"}
    assert co.story_account_set(pd.DataFrame(), None) == set()


def test_claim_coordination_filters_edges_to_accounts():
    edges = pd.DataFrame(
        {
            "src": ["a", "a", "x"],
            "dst": ["b", "z", "y"],
            "weight": [3, 2, 9],
            "channel": ["co_retweet", "co_retweet", "text_sim"],
        }
    )
    clusters = pd.DataFrame(
        {"author_id": ["a", "b", "x"], "cluster_id": [1, 1, 2]}
    )
    out = co.claim_coordination({"a", "b"}, edges, clusters)
    assert set(zip(out["edges"]["src"], out["edges"]["dst"])) == {("a", "b")}
    assert set(out["clusters"]["author_id"]) == {"a", "b"}
    assert out["summary"]["n_edges"] == 1
    assert out["summary"]["n_clusters"] == 1
    assert "malice" in out["summary"]["note"] or "probabilistic" in out["summary"]["note"]


def test_claim_coordination_empty_accounts():
    edges = pd.DataFrame({"src": ["a"], "dst": ["b"], "weight": [1], "channel": ["co_retweet"]})
    out = co.claim_coordination(set(), edges, None)
    assert out["edges"].empty
    assert out["summary"]["n_accounts"] == 0
