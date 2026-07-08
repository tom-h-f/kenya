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
