"""The daily pass: community report, persistence, and the stage order.

No R2 and no Modal: networks are built by hand, tables land in a temp
directory, and the text trace and rank stages are fakes that record calls.
"""

from __future__ import annotations

import itertools
from datetime import date, datetime, timezone

import duckdb
import networkx as nx
import pandas as pd
import pytest

from kma import coord2, coord2_communities, v2_daily, window


def _clique(prefix: str, n: int) -> list[tuple[str, str]]:
    return list(itertools.combinations([f"{prefix}{i}" for i in range(n)], 2))


def _networks() -> dict[str, pd.DataFrame]:
    """Three dense groups joined by one bridge each, plus a pair too small to
    report. Co-retweet for two of them, text for the third."""
    co = _clique("k", 8) + _clique("o", 8) + [("k0", "o0")]
    text = _clique("u", 6) + [("u0", "k1"), ("s0", "s1")]
    return {
        "co_retweet": pd.DataFrame(co, columns=["source", "target"]),
        "text_similarity": pd.DataFrame(text, columns=["source", "target"]),
    }


def _graph() -> tuple[nx.Graph, pd.Series]:
    graph = coord2.fuse(_networks())
    return graph, coord2.centrality(graph)


def _kenya(scored: dict[str, float]) -> pd.DataFrame:
    """`model_kenya_share`'s shape: users absent from `scored` have no score."""
    graph, _ = _graph()
    rows = []
    for node in graph.nodes:
        rows.append({"user_id": node, "posts": 10, "scored": 10 if node[0] in scored else 0,
                     "kenya_model": scored.get(node[0])})
    return pd.DataFrame(rows).set_index("user_id")


def _community_of(rep: coord2_communities.Report, user: str) -> int:
    return int(rep.members.loc[rep.members["user_id"] == user, "community"].iloc[0])


def test_the_groups_come_out_as_communities_and_the_pair_does_not():
    graph, glob = _graph()
    rep = coord2_communities.report(graph, glob, kenya=None, min_size=4)
    assert len(rep.communities) == 3
    assert "s0" not in set(rep.members["user_id"])
    for prefix in "kou":
        ids = {_community_of(rep, f"{prefix}{i}") for i in range(6)}
        assert len(ids) == 1


def test_a_scored_off_domain_community_is_dropped():
    graph, glob = _graph()
    rep = coord2_communities.report(graph, glob, kenya=_kenya({"k": 1.0, "o": 0.0, "u": 1.0}))
    table = rep.communities.set_index("community")
    assert not table.loc[_community_of(rep, "o0"), "kept"]
    assert table.loc[_community_of(rep, "k0"), "kept"]
    assert not set(rep.listed["user_id"]) & {f"o{i}" for i in range(8)}


def test_an_unscored_community_is_kept_and_flagged():
    """The departure from the investigation script: a daily window's newest
    communities are the ones the relevance app has not reached yet, and they
    are the ones a daily run exists to catch."""
    graph, glob = _graph()
    rep = coord2_communities.report(graph, glob, kenya=_kenya({"k": 1.0, "o": 0.0}))
    row = rep.communities.set_index("community").loc[_community_of(rep, "u0")]
    assert row["kept"]
    assert not row["relevance_scored"]


def test_the_budget_and_the_per_group_cap_bind():
    graph, glob = _graph()
    rep = coord2_communities.report(graph, glob, kenya=None, per_group=3, budget=7)
    assert len(rep.listed) == 7
    assert rep.listed.groupby("community").size().max() == 3


def test_every_member_keeps_the_papers_global_score():
    graph, glob = _graph()
    rep = coord2_communities.report(graph, glob, kenya=None)
    got = rep.members.set_index("user_id")["centrality"]
    for user in ("k0", "o3", "u5"):
        assert got[user] == pytest.approx(glob[user])


def _uri(root):
    def uri(key: str) -> str:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return uri


def _rank(tmp_path, **kw):
    sizes = pd.DataFrame([{"trace": t, "edges": len(e)} for t, e in _networks().items()])
    return v2_daily.rank(
        duckdb.connect(), window.Window(date(2026, 9, 21), 90), relevance=False,
        networks=(_networks(), sizes, "SELECT 1"), uri=_uri(tmp_path),
        now=datetime(2026, 9, 22, 3, 30, tzinfo=timezone.utc), **kw,
    )


def test_rank_writes_every_table_under_the_daily_kinds(tmp_path):
    result = _rank(tmp_path)
    assert set(result.keys) == {"accounts", "communities", "members", "runs"}
    for name, key in result.keys.items():
        assert key == f"coord2/platform=x/kind=daily_{name}/dt=2026-09-22/run=20260922T033000Z.parquet"
        frame = pd.read_parquet(tmp_path / key)
        assert set(frame["window_id"]) == {"daily-2026-09-21-90d"}
        assert set(frame["run_id"]) == {"20260922T033000Z"}


def test_the_run_row_carries_trace_sizes_and_counts(tmp_path):
    result = _rank(tmp_path)
    runs = pd.read_parquet(tmp_path / result.keys["runs"])
    row = runs.iloc[0]
    assert row["edges_co_retweet"] == len(_networks()["co_retweet"])
    assert row["communities"] == 3
    assert row["communities_unscored"] == 3


def test_an_extra_pass_is_persisted_in_the_same_run(tmp_path):
    seen = {}

    def windowed(ctx: v2_daily.PassContext) -> dict[str, pd.DataFrame]:
        seen["window"] = ctx.window.id
        seen["accounts"] = ctx.graph.number_of_nodes()
        return {"windowed": pd.DataFrame({"user_id": ["k0"], "score": [1.0]})}

    result = _rank(tmp_path, extra_passes=[windowed])
    assert seen == {"window": "daily-2026-09-21-90d", "accounts": 24}
    frame = pd.read_parquet(tmp_path / result.keys["windowed"])
    assert list(frame["user_id"]) == ["k0"]


def test_an_extra_pass_cannot_overwrite_a_daily_table(tmp_path):
    with pytest.raises(ValueError, match="collides"):
        _rank(tmp_path, extra_passes=[lambda ctx: {"members": pd.DataFrame()}])


def _manifest(rows: int) -> pd.DataFrame:
    return pd.DataFrame([{"prefix": "posts", "rows": rows}] if rows else
                        [{"prefix": "embeddings", "rows": 1}])


def test_orchestrate_builds_the_text_trace_only_when_missing():
    calls = []
    kw = dict(
        textsim=lambda sid: calls.append(("textsim", sid)) or {"edges": 1},
        rank_fn=lambda w: calls.append(("rank", w.id)) or {"listed": 0},
        con=duckdb.connect(), ensure=lambda w: _manifest(5),
    )
    v2_daily.orchestrate("2026-09-21", 90, trace_exists=lambda sid: False, **kw)
    assert calls == [("textsim", "daily-2026-09-21-90d"), ("rank", "daily-2026-09-21-90d")]
    calls.clear()
    out = v2_daily.orchestrate("2026-09-21", 90, trace_exists=lambda sid: True, **kw)
    assert calls == [("rank", "daily-2026-09-21-90d")]
    assert out["textsim"] == "reused"


def test_orchestrate_refuses_an_empty_window():
    with pytest.raises(RuntimeError, match="no posts"):
        v2_daily.orchestrate(
            "2026-09-21", 90, textsim=lambda sid: {}, rank_fn=lambda w: {},
            con=duckdb.connect(), ensure=lambda w: _manifest(0), trace_exists=lambda sid: True,
        )
