"""The leads diff, queue and verdict rows - no R2, no Modal.

What is tested is what would make the monitor either miss a campaign or ring
every morning about the same one: matching across runs whose ids mean
nothing, the budget, idempotency, and the canary contract.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kma import leads, leads_run


def listing(groups: dict[int, list[str]], **cols) -> pd.DataFrame:
    rows = [{"user_id": u, "community": c, **cols} for c, us in groups.items() for u in us]
    return pd.DataFrame(rows)


def accounts(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


# -- matching ---------------------------------------------------------------


def test_a_relabelled_community_is_the_same_community():
    """Leiden reissues labels every run; community 0 becoming community 7 with
    the same members is not news."""
    prev = listing({0: accounts("a", 10)})
    cur = listing({7: accounts("a", 10)})
    d = leads.diff_communities(prev, cur)
    assert d["status"].tolist() == [leads.STATUS_SAME]
    assert d["jaccard"].iloc[0] == 1.0


def test_one_member_of_jitter_on_a_small_community_is_same():
    """The 0.5-0.6 bump in the measured distribution is small communities
    swapping one member. Re-judging those is the noise the cut exists to stop."""
    prev = listing({0: ["a", "b", "c", "d", "e"]})
    cur = listing({0: ["a", "b", "c", "d", "f"]})
    assert leads.diff_communities(prev, cur)["status"].iloc[0] == leads.STATUS_SAME


def test_a_community_with_no_predecessor_is_new():
    prev = listing({0: accounts("a", 10)})
    cur = listing({0: accounts("a", 10), 1: accounts("z", 6)})
    d = leads.diff_communities(prev, cur).set_index("size")
    assert d.loc[6, "status"] == leads.STATUS_NEW
    assert d.loc[6, "jaccard"] == 0.0


def test_growth_is_a_change_even_though_every_old_member_stayed():
    """Containment 1.0 with a falling Jaccard: the same campaign, bigger."""
    prev = listing({0: accounts("a", 8)})
    cur = listing({0: accounts("a", 8) + accounts("n", 6)})
    row = leads.diff_communities(prev, cur).iloc[0]
    assert row["containment"] == 1.0
    assert row["jaccard"] == pytest.approx(8 / 14)
    assert row["status"] == leads.STATUS_CHANGED


def test_small_growth_below_the_floor_is_not_a_change():
    prev = listing({0: accounts("a", 20)})
    cur = listing({0: accounts("a", 20) + accounts("n", 4)})
    assert leads.diff_communities(prev, cur)["status"].iloc[0] == leads.STATUS_SAME


def test_half_replaced_community_is_changed():
    prev = listing({0: accounts("a", 10)})
    cur = listing({0: accounts("a", 5) + accounts("n", 5)})
    row = leads.diff_communities(prev, cur).iloc[0]
    assert row["jaccard"] == pytest.approx(5 / 15)
    assert row["status"] == leads.STATUS_CHANGED


def test_best_match_picks_the_higher_jaccard_predecessor():
    prev = listing({0: accounts("a", 4), 1: accounts("a", 10)})
    cur = listing({0: accounts("a", 10)})
    m = leads.best_matches(leads.groups(prev, "community", "user_id"),
                           leads.groups(cur, "community", "user_id"))
    assert m["previous"].iloc[0] == 1


def test_communities_under_four_are_ignored_on_both_sides():
    prev = listing({0: accounts("a", 10)})
    cur = listing({0: accounts("a", 10), 1: ["x", "y", "z"]})
    assert len(leads.diff_communities(prev, cur)) == 1


def test_a_listing_without_communities_fails_loudly():
    with pytest.raises(ValueError, match="community"):
        leads.diff_communities(None, pd.DataFrame({"user_id": ["a"]}))


def test_candidate_ids_are_stable_for_the_same_membership():
    a = leads.diff_communities(None, listing({3: accounts("a", 5)}))
    b = leads.diff_communities(None, listing({9: accounts("a", 5)}))
    assert a["candidate_id"].iloc[0] == b["candidate_id"].iloc[0]


# -- trends -----------------------------------------------------------------


def trends(rows):
    return pd.DataFrame(rows, columns=["tag", "authors", "emergence"])


def test_a_tag_not_in_the_previous_run_is_new():
    d = leads.diff_trends(trends([("old", 5, 3.0)]), trends([("old", 5, 3.0), ("fresh", 2, 9.0)]))
    assert d.set_index("tag")["status"].to_dict() == {"fresh": "new", "old": "same"}


def test_a_reselected_tag_is_only_changed_if_its_authors_doubled():
    prev = trends([("a", 8, 2.0), ("b", 8, 2.0)])
    cur = trends([("a", 26, 2.0), ("b", 12, 2.0)])
    d = leads.diff_trends(prev, cur).set_index("tag")["status"]
    assert d["a"] == leads.STATUS_CHANGED
    assert d["b"] == leads.STATUS_SAME


def test_every_tag_is_new_without_a_previous_run():
    assert set(leads.diff_trends(None, trends([("a", 2, 1.0)]))["status"]) == {"new"}


# -- queue ------------------------------------------------------------------


def comm_diff(n_new: int, n_changed: int = 0, n_same: int = 0) -> pd.DataFrame:
    status = ["new"] * n_new + ["changed"] * n_changed + ["same"] * n_same
    return pd.DataFrame({"candidate_id": [f"c{i}" for i in range(len(status))],
                         "status": status, "group": range(len(status))})


def trend_diff(n_new: int) -> pd.DataFrame:
    return pd.DataFrame({"candidate_id": [f"trend:t{i}" for i in range(n_new)],
                         "status": ["new"] * n_new, "tag": [f"t{i}" for i in range(n_new)]})


def test_the_queue_never_exceeds_the_budget_and_records_what_it_skipped():
    q, s = leads.queue(comm_diff(20), trend_diff(5), limit=10)
    assert len(q) == 10
    assert len(s) == 15
    assert (q["source"] == "trend").sum() == leads.TREND_RESERVE


def test_same_is_never_queued():
    q, s = leads.queue(comm_diff(0, 0, 50), trend_diff(0))
    assert q.empty and s.empty


def test_unused_trend_slots_pass_to_communities():
    q, _ = leads.queue(comm_diff(20), trend_diff(1), limit=10)
    assert (q["source"] == "community").sum() == 9


def test_new_is_queued_before_changed():
    q, _ = leads.queue(comm_diff(0, 5).pipe(lambda d: pd.concat(
        [d, comm_diff(3).assign(candidate_id=lambda x: "n" + x["candidate_id"])])),
        trend_diff(0), limit=3)
    assert set(q["status"]) == {"new"}


# -- plan: baseline, reset, canary ------------------------------------------


def test_a_first_run_is_a_baseline_and_judges_no_communities():
    cur = leads_run.Listing("r2", listing({0: accounts("a", 5)}))
    p = leads_run.plan(None, cur, None, trends([]), run_id="20260922T000000Z")
    assert p.baseline
    assert p.queued.empty


def test_a_wholesale_turnover_is_a_reset_not_a_wave_of_campaigns():
    prev = leads_run.Listing("r1", listing({0: accounts("a", 5), 1: accounts("b", 5)}))
    cur = leads_run.Listing("r2", listing({0: accounts("x", 5), 1: accounts("y", 5)}))
    p = leads_run.plan(prev, cur, None, trends([]), run_id="20260922T000000Z")
    assert p.reset
    assert (p.queued.get("source", pd.Series(dtype=str)) == "community").sum() == 0


def test_a_canary_is_judged_outside_the_budget():
    prev = leads_run.Listing("r1", listing({i: accounts(f"p{i}_", 5) for i in range(20)}))
    groups = {i: accounts(f"p{i}_", 5) for i in range(20)}
    groups.update({100 + i: accounts(f"new{i}_", 5) for i in range(3)})
    groups[999] = [f"canary:2026-09-28:{i}" for i in range(6)]
    cur = leads_run.Listing("r2", listing(groups))
    p = leads_run.plan(prev, cur, None, trends([]), limit=2, run_id="20260922T000000Z")
    assert len(p.queued) == 3
    assert p.queued["canary_id"].dropna().tolist() == ["2026-09-28"]


def test_a_canary_is_judged_even_on_a_baseline_run():
    cur = leads_run.Listing("r2", listing({0: [f"canary:c1:{i}" for i in range(5)]}))
    p = leads_run.plan(None, cur, None, trends([]), run_id="20260922T000000Z")
    assert p.baseline
    assert p.queued["canary_id"].tolist() == ["c1"]


def test_off_domain_communities_are_not_queued_when_relevance_is_known():
    prev = leads_run.Listing("r1", pd.concat([listing({i: accounts(f"p{i}_", 5)}, kenya_share=0.9)
                                              for i in range(5)]))
    cur = leads_run.Listing("r2", pd.concat(
        [prev.frame,
         listing({50: accounts("kpop", 5)}, kenya_share=0.0),
         listing({51: accounts("ke", 5)}, kenya_share=0.8)]))
    p = leads_run.plan(prev, cur, None, trends([]), run_id="20260922T000000Z")
    assert p.queued["group"].tolist() == [51]


# -- worth a look -----------------------------------------------------------


@pytest.mark.parametrize("kind, relevant, status, expected", [
    ("influence_operation", False, "changed", True),
    ("political_campaign", True, "new", True),
    ("unclear", True, "new", True),
    ("political_campaign", True, "changed", False),
    ("political_campaign", False, "new", False),
    ("engagement_pod", True, "new", False),
    ("fandom_or_interest", True, "new", False),
])
def test_worth_a_look(kind, relevant, status, expected):
    v = {"cluster_type": kind, "kenya_relevant": relevant}
    assert leads_run.worth_a_look(v, status) is expected


# -- verdict rows -----------------------------------------------------------


def test_verdict_rows_map_back_to_candidates_with_provenance():
    prev = leads_run.Listing("r1", listing({0: accounts("a", 10)}))
    cur = leads_run.Listing("r2", listing({0: accounts("a", 10), 5: accounts("z", 6)}))
    p = leads_run.plan(prev, cur, None, trends([]), run_id="20260922T063000Z")
    cand = p.queued["candidate_id"].iloc[0]
    rows = leads_run.verdict_rows(
        p, [{"cluster_id": 5, "cluster_type": "influence_operation", "kenya_relevant": True,
             "confidence": "low", "rationale": "r", "what_would_change_this": "w"}],
        {5: cand}, {cand: "k"}, adjudicator="m", rubric="rb", mode="production")
    row = rows.iloc[0]
    assert row["candidate_id"] == cand and row["status"] == "new"
    assert row["worth_a_look"] and row["dossier_key"] == "k"
    assert row["listing_run"] == "r2" and row["previous_run"] == "r1"
    assert row["adjudicator"] == "m" and row["mode"] == "production"
