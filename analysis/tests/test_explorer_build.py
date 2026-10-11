"""Lineage, layout and the small pure helpers of the explorer builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kma import explorer_build as eb


def ids(prefix: str, n: int) -> set[str]:
    return {f"{prefix}{i}" for i in range(n)}


def by_status(matches: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for m in matches:
        out.setdefault(m["status"], []).append(m)
    return out


def test_a_community_that_keeps_most_members_continues_with_its_lineage():
    prev = {0: ids("a", 10)}
    cur = {5: ids("a", 9) | {"new"}}
    matches, lineage = eb.match_communities(cur, prev, {0: "root"})
    assert lineage == {5: "root"}
    (m,) = matches
    assert (m["status"], m["cid"], m["previous_cid"]) == ("continuing", 5, 0)
    assert m["joined"] == 1 and m["left"] == 1 and m["size"] == 10 and m["previous_size"] == 10


def test_an_unmatched_community_is_new_with_its_own_lineage():
    matches, lineage = eb.match_communities({1: ids("n", 6)}, {0: ids("a", 6)}, {0: "root"})
    got = by_status(matches)
    assert [m["cid"] for m in got["new"]] == [1]
    assert [m["previous_cid"] for m in got["vanished"]] == [0]
    assert lineage[1] == eb.lineage_id(ids("n", 6)) != "root"
    assert got["vanished"][0]["lineage"] == "root" and got["vanished"][0]["cid"] is None


def test_a_community_split_in_two_keeps_one_lineage_and_starts_another():
    prev = {0: ids("a", 20)}
    cur = {1: ids("a", 12), 2: set(f"a{i}" for i in range(12, 20)) | {"x"}}
    matches, lineage = eb.match_communities(cur, prev, {0: "root"})
    got = by_status(matches)
    assert [m["cid"] for m in got["continuing"]] == [1]
    assert [m["cid"] for m in got["split"]] == [2]
    assert got["split"][0]["previous_cid"] == 0
    assert lineage[1] == "root" and lineage[2] != "root"
    assert "vanished" not in got


def test_two_communities_merged_into_one_report_the_absorbed_one():
    prev = {0: ids("a", 20), 1: ids("b", 8)}
    cur = {7: ids("a", 20) | ids("b", 8)}
    matches, lineage = eb.match_communities(cur, prev, {0: "ra", 1: "rb"})
    got = by_status(matches)
    assert [m["cid"] for m in got["continuing"]] == [7]
    assert [(m["cid"], m["previous_cid"]) for m in got["merged"]] == [(7, 1)]
    assert lineage[7] == "ra"


def test_matching_pairs_by_overlap_not_by_id_and_is_deterministic():
    prev = {0: ids("a", 10), 1: ids("b", 10)}
    cur = {3: ids("b", 9) | {"x"}, 4: ids("a", 9) | {"y"}}
    first, lineage = eb.match_communities(cur, prev, {0: "r0", 1: "r1"})
    again, _ = eb.match_communities(cur, prev, {0: "r0", 1: "r1"})
    assert first == again
    assert lineage == {3: "r1", 4: "r0"}
    assert {(m["cid"], m["previous_cid"]) for m in first if m["status"] == "continuing"} == {(3, 1), (4, 0)}


def test_lineage_id_ignores_order_and_changes_with_membership():
    assert eb.lineage_id(["b", "a"]) == eb.lineage_id(["a", "b"])
    assert eb.lineage_id(["a", "b"]) != eb.lineage_id(["a", "c"])
    assert len(eb.lineage_id(["a"])) == 10


def test_shards_follow_the_contract_rules():
    assert 0 <= eb.account_shard("9000000000000000001") < 64
    assert eb.hashtag_shard("Water") == eb.hashtag_shard("water")
    assert 0 <= eb.hashtag_shard("water") < 16


def test_layout_is_seeded_and_scaled_into_the_unit_box():
    pairs = np.array([[i, (i + 1) % 30] for i in range(30)] + [[0, 15], [5, 20]])
    first = eb.layout(30, pairs, seed=0)
    second = eb.layout(30, pairs, seed=0)
    assert np.allclose(first, second)
    assert np.abs(first).max() <= 1.0 + 1e-9
    assert first.shape == (30, 2)


def test_layout_handles_empty_single_and_edgeless_graphs():
    assert eb.layout(0, np.zeros((0, 2), dtype=int)).shape == (0, 2)
    assert eb.layout(1, np.zeros((0, 2), dtype=int)).shape == (1, 2)
    assert eb.layout(6, np.zeros((0, 2), dtype=int)).shape == (6, 2)


def test_the_stamp_carries_the_window_and_the_parameters():
    run = {"run_id": "20261010T031500Z", "window_id": "daily-2026-10-09-21d",
           "window_end": "2026-10-09", "window_days": 21,
           "computed_at": pd.Timestamp("2026-10-10T03:41:07Z"), "text_threshold": 0.85,
           "text_overlap": 0.1, "min_entities": 2, "min_kenya": 0.6}
    stamp = eb.make_stamp(run, pd.Timestamp("2026-10-10T04:00:00Z").to_pydatetime())
    assert stamp["window_start"] == "2026-09-19" and stamp["window_days"] == 21
    assert stamp["params"]["leiden_seed"] == 0 and stamp["params"]["edge_percentile"]["fast_retweet"] is None
    assert stamp["computed_at"] == "2026-10-10T03:41:07Z"
