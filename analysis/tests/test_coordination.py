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


# --- per-pass counters (census-tuning measurement layer) --------------------


def _trace_con(rows):
    """Register (author_id, action_object) rows as a trace table."""
    import duckdb

    con = duckdb.connect()
    df = pd.DataFrame(rows, columns=["author_id", "action_object"])
    df["created_at"] = pd.NaT
    con.register("_t", df)
    con.execute("CREATE TABLE t AS SELECT * FROM _t")
    return con


def test_stats_are_absent_when_not_requested():
    """Default None must leave behaviour and the return value untouched."""
    con = _trace_con([("a", "o1"), ("a", "o2"), ("b", "o1"), ("b", "o2")])
    out = co.validated_edges(con, "co_retweet", trace_table="t")
    assert list(out.columns).count("sig_fdr") == 1
    assert len(out) == 1  # the a-b pair, tested


def test_pairable_excludes_accounts_holding_one_non_hub_object():
    """`nohub_amp` is not the metric of record. `projected_edges` requires
    `min_repetition` distinct shared objects, so an account with a single
    non-hub trace can never appear in any edge no matter what else is
    collected. Measured live 2026-08-01: 25,083 survivors, 7,079 pairable -
    the survivor count overstates by 3.5x."""
    rows = [
        ("pair1", "o1"), ("pair1", "o2"),   # 2 objects -> pairable
        ("pair2", "o1"), ("pair2", "o2"),   # 2 objects -> pairable
        ("lone1", "o3"),                    # 1 object  -> survives, unpairable
        ("lone2", "o4"),
        ("lone3", "o5"),
    ]
    con = _trace_con(rows)
    stats: dict = {}
    co.validated_edges(con, "co_retweet", trace_table="t", hub_cap=100, stats=stats)

    assert stats["accounts_all"] == 5
    assert stats["nohub_amp"] == 5          # nothing was capped away
    assert stats["pairable"] == 2           # only the two with >= 2 objects
    assert stats["min_repetition"] == 2


def test_stats_report_the_cap_and_the_hub_objects_it_excluded():
    rows = [(f"u{i}", "hub") for i in range(12)]
    rows += [("a", "o1"), ("a", "o2"), ("b", "o1"), ("b", "o2")]
    con = _trace_con(rows)
    stats: dict = {}
    co.validated_edges(con, "co_retweet", trace_table="t", hub_cap=5, stats=stats)

    assert stats["hub_cap"] == 5
    assert stats["hub_objects"] == 1
    assert stats["hub_max_degree"] == 12
    assert stats["objects_all"] == 3            # hub, o1, o2
    assert stats["objects_nohub"] == 2          # the hub is gone
    # every account whose only trace was the hub is evicted with it
    assert stats["accounts_all"] == 14
    assert stats["nohub_amp"] == 2
    assert stats["traces_all"] == 16
    assert stats["traces_nohub"] == 4


def test_stats_do_not_trigger_a_second_trace_pass(monkeypatch):
    """Guards the memory-constrained host against the obvious future refactor
    ("just add a channel_stats() helper that recomputes it"). tf1 already OOMs
    at projected_edges; a second traces() build is the worst option available."""
    con = _trace_con([("a", "o1"), ("a", "o2"), ("b", "o1"), ("b", "o2")])
    calls = []
    monkeypatch.setattr(co, "traces", lambda *a, **k: calls.append(1))

    co.validated_edges(con, "co_retweet", trace_table="t", stats={})

    assert calls == []  # trace_table supplied; nothing rebuilt it


def test_timed_channel_stats_omit_the_hub_fields():
    """No cap is applied on the Monte-Carlo path, so the hub fields must be
    ABSENT rather than zero - zero reads as "no hubs were excluded"."""
    con = _trace_con([("a", "o1"), ("a", "o2"), ("b", "o1"), ("b", "o2")])
    stats: dict = {}
    co.validated_edges(
        con, "fast_co_share", delta=300, n_iter=5, trace_table="t", stats=stats
    )

    assert stats["timed"] is True
    assert "hub_cap" not in stats
    assert "nohub_amp" not in stats
    assert "pairable" not in stats
    assert stats["accounts_all"] == 2      # the untimed counters still land


def test_build_layers_does_not_share_one_stats_dict_between_channels():
    """Forwarded through **params a single dict would be overwritten by each
    channel in turn, leaving only the last."""
    con = _trace_con([("a", "o1"), ("a", "o2"), ("b", "o1"), ("b", "o2")])
    stats: dict = {}
    co.build_layers(
        con, channels=["co_retweet", "co_reply"], stats=stats, trace_table="t"
    )

    assert set(stats) == {"co_retweet", "co_reply"}
    assert stats["co_retweet"]["channel"] == "co_retweet"
    assert stats["co_reply"]["channel"] == "co_reply"
    assert stats["co_retweet"] is not stats["co_reply"]
    assert all("edges_kept" in s for s in stats.values())


class _LocalCopyCon:
    """Wraps a real DuckDB connection and redirects the r2:// COPY target to a
    local file, so the write path (row shaping, dtypes, parquet round-trip) is
    exercised for real without a bucket."""

    def __init__(self, con, out_path):
        self._con, self.out = con, out_path

    def register(self, name, df):
        return self._con.register(name, df)

    def unregister(self, name):
        return self._con.unregister(name)

    def execute(self, sql, *a):
        import re

        sql = re.sub(r"'r2://[^']+'", f"'{self.out}'", sql)
        return self._con.execute(sql, *a)

    def sql(self, q):
        return self._con.sql(q)


def test_persist_run_metrics_writes_one_row_per_channel(tmp_path):
    import duckdb

    out = tmp_path / "m.parquet"
    con = _LocalCopyCon(duckdb.connect(), out)
    key = co.persist_run_metrics(
        con,
        {
            "co_retweet": {"channel": "co_retweet", "nohub_amp": 25083,
                           "pairable": 7079, "hub_objects": 533, "edges_kept": 65570},
            "co_reply": {"channel": "co_reply", "nohub_amp": 900,
                         "pairable": 300, "hub_objects": 0, "edges_kept": 1898},
        },
        {"channels": ["co_retweet", "co_reply"], "method": "fdr",
         "n_clusters": 298, "n_corroborated_clusters": 14,
         "n_corroborated_accounts": 60, "n_accounts": 1200},
    )

    assert "kind=run_metrics" in key and key.endswith(".parquet")
    df = duckdb.connect().sql(f"SELECT * FROM '{out}'").df()
    assert len(df) == 2
    assert set(df["channel"]) == {"co_retweet", "co_reply"}
    # run-level columns repeat on every row
    assert set(df["n_corroborated_clusters"]) == {14}
    assert df.loc[df["channel"] == "co_retweet", "pairable"].item() == 7079
    assert set(df["channels"]) == {"co_retweet,co_reply"}


def test_persist_run_metrics_keeps_integer_columns_nullable(tmp_path):
    """Timed channels leave the hub fields NULL. Without an explicit nullable
    dtype pandas promotes them to float64, so successive runs write int64 and
    double under one glob and union_by_name fails later, far from the cause."""
    import duckdb

    out = tmp_path / "m.parquet"
    con = _LocalCopyCon(duckdb.connect(), out)
    co.persist_run_metrics(
        con,
        {"fast_co_share": {"channel": "fast_co_share", "timed": True,
                           "accounts_all": 5}},   # no hub_cap / pairable
        {"n_clusters": 0},
    )

    types = {
        r[0]: r[1]
        for r in duckdb.connect()
        .sql(f"DESCRIBE SELECT * FROM '{out}'")
        .fetchall()
    }
    assert types["pairable"] == "BIGINT"
    assert types["hub_cap"] == "BIGINT"
    assert types["accounts_all"] == "BIGINT"
    df = duckdb.connect().sql(f"SELECT * FROM '{out}'").df()
    assert pd.isna(df["pairable"].iloc[0])


def test_persist_run_metrics_writes_a_row_even_with_no_channel_stats(tmp_path):
    """A run that produced no channel counters still dates itself."""
    import duckdb

    out = tmp_path / "m.parquet"
    con = _LocalCopyCon(duckdb.connect(), out)
    co.persist_run_metrics(con, {}, {"n_clusters": 0, "n_accounts": 0})

    df = duckdb.connect().sql(f"SELECT * FROM '{out}'").df()
    assert len(df) == 1
    assert df["n_clusters"].item() == 0


def test_clustering_defaults_to_bonferroni_not_fdr():
    """Measured on the live corpus 2026-08-02, not a preference for caution.

    FDR admits 4% (co_retweet) and 28% (co_reply) of degree-preserving shuffled
    noise where Bonferroni admits 0 and 1. It buys no recall in exchange: both
    score F1 = 1.0 on a 20-account/10-object plant. At the detection limit it is
    strictly worse - with 8 accounts sharing 5 objects both filters contain all
    28 planted pairs, but Leiden clusters 0 of 8 planted accounts under FDR
    against 8 of 8 under Bonferroni, because the marginal edges wire the group
    to unrelated accounts until CPM cannot hold it together.

    Community detection is what fails, not the null. Same dilution the hub cap
    exists to prevent, one layer down."""
    import inspect

    from kma import coordination_run as cr

    assert co.DEFAULT_EDGE_METHOD == "bonferroni"
    assert inspect.signature(co.build_layers).parameters["method"].default == "bonferroni"
    assert inspect.signature(cr.run).parameters["method"].default == "bonferroni"


def test_percentile_is_computed_but_never_the_default():
    """It missed even the 20x10 plant on co_retweet (F1 = 0.0). Kept for
    comparison in the persisted edge sets; never used for clustering."""
    assert co.DEFAULT_EDGE_METHOD != "percentile"
