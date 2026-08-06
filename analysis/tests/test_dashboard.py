"""Unit tests for kma.dashboard (no R2, no models).

The public-safety tests here are the reason the summary blob can be trusted.
"Public-safe by construction" is a docstring claim; these assert it against the
built payload, using fabricated data seeded with exactly the things that must
never reach the page.
"""

import re

import pandas as pd
import pytest

from kma import dashboard as d

SNOWFLAKE = re.compile(r"^[0-9]{15,20}$")

FORBIDDEN_KEYS = {
    "author_id", "author_handle", "platform_user_id", "platform_post_id",
    "handle", "screen_name", "display_name", "text", "url", "bio",
    "profile_image_url", "src", "dst", "member_post_ids",
}

# Values planted in the fixtures below. None may appear anywhere in the payload.
LEAKY_VALUES = {
    "1234567890123456789", "9876543210987654321", "1111111111111111111",
    "2222222222222222222", "3333333333333333333",
    "kenyanpatriot", "wanjiku254", "@rutoWatch",
}


def _walk(node, path="$"):
    """Yield (path, key, value) for every scalar in a nested structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
            if not isinstance(v, (dict, list)):
                yield (path, k, v)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
            if not isinstance(v, (dict, list)):
                yield (f"{path}[{i}]", None, v)


class _Rel:
    """Stands in for a DuckDB relation: the readers only ever have .df() called."""

    def __init__(self, frame):
        self._frame = frame

    def df(self):
        return self._frame


def _members():
    """Two clusters: one corroborated (2 channels), one candidate. Author ids are
    snowflakes and one cluster name carries a curated handle as bare prose."""
    return pd.DataFrame({
        "author_id": [
            "1234567890123456789", "9876543210987654321",
            "1111111111111111111", "2222222222222222222",
            "3333333333333333333",
        ],
        "cluster_id": [0, 0, 0, 0, 1],
        "size": [4, 4, 4, 4, 1],
        "channels": [["co_retweet", "co_reply"]] * 4 + [["co_retweet"]],
        "n_channels": [2, 2, 2, 2, 1],
        "internal_edge_share": [0.62, 0.62, 0.62, 0.62, 0.0],
        "name": ["wantam ruto"] * 4 + ["kenyanpatriot rigged"],
        "label": ["wantam ruto (n=4)"] * 4 + ["kenyanpatriot rigged (n=1)"],
    })


def _edges():
    return pd.DataFrame({
        "src": [
            "1234567890123456789", "1234567890123456789",
            "9876543210987654321", "1111111111111111111",
        ],
        "dst": [
            "9876543210987654321", "1111111111111111111",
            "2222222222222222222", "2222222222222222222",
        ],
        "weight": [5.0, 3.0, 2.0, 1.0],
        "min_gap": [4.0, 9.0, 30.0, 60.0],
        "channel": ["co_retweet", "co_retweet", "co_reply", "co_retweet"],
    })


def _metrics():
    return pd.DataFrame({
        "computed_at": pd.to_datetime(["2026-08-04T12:00:00Z", "2026-08-04T12:00:00Z"]),
        "code_version": ["bb56440", "bb56440"],
        "channel": ["co_retweet", "co_reply"],
        "method": ["bonferroni", "bonferroni"],
        "resolution": [0.05, 0.05],
        "min_size": [3, 3],
        "lookback_days": [14, 14],
        "n_clusters": [2, 2],
        "n_accounts": [5, 5],
        "n_corroborated_clusters": [1, 1],
        "n_corroborated_accounts": [4, 4],
    })


@pytest.fixture
def payload(monkeypatch):
    """A full summary built from the fabricated frames, with R2 stubbed out."""
    def fake_run_latest(con, kind="clusters", platform="x", channel="*", method="*"):
        return _Rel(_members() if kind == "clusters" else _edges())

    monkeypatch.setattr(d.db, "coordination_run_latest", fake_run_latest)
    monkeypatch.setattr(d.db, "coordination_metrics", lambda con, platform="x": _Rel(_metrics()))
    # `leak_corrected` calls this positionally, `build_coordination` by keyword.
    monkeypatch.setattr(
        d.db, "curated_handles", lambda *a, **kw: {"kenyanpatriot", "wanjiku254"}
    )
    monkeypatch.setattr(d, "_collection_start", lambda con, platform="x": "2026-07-04T13:17:40+00:00")
    monkeypatch.setattr(d, "NET_MIN_SIZE", 2)
    monkeypatch.setattr(d, "_start_cache", None)
    return d.build_summary(object(), platform="x")


def test_no_identifying_keys(payload):
    offenders = [
        f"{path}.{key}" for path, key, _ in _walk(payload) if key in FORBIDDEN_KEYS
    ]
    assert not offenders, f"identifying keys in public payload: {offenders}"


def test_no_author_ids_or_snowflakes(payload):
    hits = []
    for path, _, value in _walk(payload):
        if isinstance(value, str):
            if value in LEAKY_VALUES or SNOWFLAKE.match(value):
                hits.append(f"{path}={value}")
    assert not hits, f"identifiers leaked into public payload: {hits}"


def test_cluster_name_carrying_a_handle_is_dropped(payload):
    names = [
        c["name"]
        for c in payload["coordination"]["network"]["clusters"]
        + payload["coordination"]["corroborated"]["clusters"]
    ]
    assert "wantam ruto" in names
    assert not any(n and "kenyanpatriot" in n for n in names)


def test_safe_name_passes_clean_names():
    assert d._safe_name("wantam ruto", {"kenyanpatriot"}) == "wantam ruto"
    assert d._safe_name("kenyanpatriot rigged", {"kenyanpatriot"}) is None
    assert d._safe_name(None, set()) is None
    assert d._safe_name("   ", set()) is None


def test_node_and_cluster_ids_are_opaque_ints(payload):
    net = payload["coordination"]["network"]
    assert net["nodes"], "fixture should produce a drawable network"
    assert all(isinstance(n["id"], int) for n in net["nodes"])
    assert all(isinstance(c["id"], int) for c in net["clusters"])
    assert all(isinstance(e["s"], int) and isinstance(e["t"], int) for e in net["edges"])
    ids = [n["id"] for n in net["nodes"]]
    assert len(ids) == len(set(ids))


def test_corroborated_and_candidates_are_never_summed(payload):
    co = payload["coordination"]
    head = co["headline"]
    assert head["clusters_corroborated"] == 1
    assert co["candidates"]["n"] == 1
    assert head["clusters_total"] == 2
    # The two are reported separately; the total is not a third blended count.
    assert head["clusters_total"] == head["clusters_corroborated"] + co["candidates"]["n"]
    assert len(co["corroborated"]["clusters"]) == head["clusters_corroborated"]


def test_edges_respect_the_payload_cap(payload, monkeypatch):
    assert len(payload["coordination"]["network"]["edges"]) <= d.NET_MAX_EDGES


def test_multiplex_scales_layers_by_mass():
    """A layer with many edges must not dominate the ranking by count alone."""
    agg = d._multiplex(_edges())
    assert set(agg.columns) >= {"src", "dst", "weight", "channels", "n_channels"}
    # co_reply carries one edge of weight 2.0, so mass-normalised it is 1.0 -
    # heavier than any single co_retweet edge despite the layer being smaller.
    reply = agg[agg["channels"].apply(lambda c: c == ["co_reply"])]
    assert float(reply["weight"].iloc[0]) == pytest.approx(1.0)


def test_unpublished_phases_are_null_not_missing(payload):
    for key in ("toxicity", "share_of_voice", "targeting"):
        assert key in payload
        assert payload[key] is None


def test_meta_carries_the_methodology_rules(payload):
    meta = payload["meta"]
    assert meta["scope"] == "baseline"
    assert meta["search_horizon_days"] == 14
    assert isinstance(meta["leak_corrected"], bool)
    assert meta["collection_start"].startswith("2026-07-04")
    assert "hate_search" in meta["excluded_types"]
    assert meta["model_limits"]["hate_classifier"]


@pytest.mark.parametrize(
    "k,n",
    [(0, 100), (1, 100), (5, 20), (11, 1843), (96, 1843), (50, 100), (100, 100), (3, 3)],
)
def test_wilson_matches_statsmodels(k, n):
    sm = pytest.importorskip("statsmodels.stats.proportion")
    lo, hi = d._wilson(k, n)
    elo, ehi = sm.proportion_confint(k, n, alpha=0.05, method="wilson")
    assert lo == pytest.approx(elo, abs=1e-9)
    assert hi == pytest.approx(ehi, abs=1e-9)


def test_wilson_stays_in_bounds_on_zero_and_empty():
    assert d._wilson(0, 0) == (0.0, 0.0)
    lo, hi = d._wilson(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.0 < hi < 1.0
