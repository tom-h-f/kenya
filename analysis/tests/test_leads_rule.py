"""The deterministic leads rule, its gauge, and the canary evaluation - no R2.

What is tested is what would make the monitor miss a planted group or push the
same group every morning: the Kenya gate, the non-retweet share, the "is it
news" logic across runs, the memory, and the size band that decides a push.
"""

from __future__ import annotations

import pandas as pd

from kma import canary_check, leads, leads_rule, leads_run, textfile


def communities(*rows: dict) -> pd.DataFrame:
    base = {"kept": True, "template_label": None, "share_co_retweet": 1.0,
            "share_text_similarity": None, "share_hashtag_sequence": None,
            "share_co_url": None, "share_fast_retweet": None, "kenya_model": 0.9, "size": 10}
    return pd.DataFrame([{**base, **r} for r in rows])


def diff(*rows: dict) -> pd.DataFrame:
    base = {"previous": None, "previous_size": 0, "jaccard": 0.0, "joined": 0,
            "status": leads.STATUS_NEW, "size": 10, "canary_id": None}
    return pd.DataFrame([{"candidate_id": f"c{r['group']}", **base, **r} for r in rows])


def text_group(community: int, share: float = 1.0, **extra) -> dict:
    return {"community": community, "share_co_retweet": 1 - share,
            "share_text_similarity": share, **extra}


# -- signals ------------------------------------------------------------------


def test_a_retweet_network_is_not_flagged():
    s = leads_rule.signals(communities({"community": 0}))
    assert not s["flagged"].iloc[0]


def test_similar_wording_over_the_threshold_is_flagged():
    s = leads_rule.signals(communities(text_group(0, 0.5), text_group(1, 0.49)))
    assert s["flagged"].tolist() == [True, False]


def test_non_retweet_traces_add_up():
    """A group split between shared URLs and fast retweets is as un-retweet-like
    as one on a single trace."""
    s = leads_rule.signals(communities(
        {"community": 0, "share_co_retweet": 0.4, "share_co_url": 0.3, "share_fast_retweet": 0.3}))
    assert s["flagged"].iloc[0]
    assert s["non_retweet_share"].iloc[0] == 0.6


def test_a_template_label_flags_a_retweet_group():
    s = leads_rule.signals(communities({"community": 0, "template_label": "template"}))
    assert s["flagged"].iloc[0]
    assert "template:template" in s["signals"].iloc[0]


def test_a_group_that_failed_the_kenya_gate_is_never_flagged():
    s = leads_rule.signals(communities(text_group(0, 1.0, kept=False, template_label="template")))
    assert not s["flagged"].iloc[0]


# -- is it news ---------------------------------------------------------------


def test_a_new_flagged_group_alerts():
    j = leads_rule.judge(diff({"group": 3}), communities(text_group(3)), None)
    assert j["alert"].tolist() == [True]


def test_a_changed_group_that_was_flagged_yesterday_does_not_alert_again():
    j = leads_rule.judge(
        diff({"group": 3, "previous": 9, "status": leads.STATUS_CHANGED}),
        communities(text_group(3)), communities(text_group(9)))
    assert j["previously_flagged"].tolist() == [True]
    assert j["alert"].tolist() == [False]


def test_a_group_that_newly_crosses_the_line_alerts_even_if_same():
    j = leads_rule.judge(
        diff({"group": 3, "previous": 9, "status": leads.STATUS_SAME}),
        communities(text_group(3, 0.7)), communities(text_group(9, 0.3)))
    assert j["alert"].tolist() == [True]


def test_a_group_alerted_in_the_last_fortnight_does_not_alert_again():
    """A small group that drops under the size floor and returns is 'new' to
    the one-run diff; the memory is what stops a second push."""
    members = {3: frozenset({"a", "b", "c", "d"})}
    j = leads_rule.judge(diff({"group": 3}), communities(text_group(3)), None,
                         members=members, recent=[frozenset({"a", "b", "c", "x"})])
    assert j["seen_before"].tolist() == [True]
    assert j["alert"].tolist() == [False]


def test_a_different_group_with_one_shared_member_still_alerts():
    members = {3: frozenset({"a", "b", "c", "d"})}
    j = leads_rule.judge(diff({"group": 3}), communities(text_group(3)), None,
                         members=members, recent=[frozenset({"a", "x", "y", "z"})])
    assert j["alert"].tolist() == [True]


def test_size_decides_the_band():
    j = leads_rule.judge(diff({"group": 3, "size": 7}, {"group": 4, "size": 8}),
                         communities(text_group(3), text_group(4)), None)
    assert j["band"].tolist() == [leads_rule.SMALL, leads_rule.LARGE]


# -- gauge and rows -----------------------------------------------------------


def plan(communities_diff: pd.DataFrame) -> leads_run.Plan:
    empty = pd.DataFrame(columns=["candidate_id", "status"])
    return leads_run.Plan("20261111T040000Z", "cur", "prev", False, False,
                          empty, empty, communities_diff, empty)


def test_rows_keep_every_flagged_group_and_mark_the_alerts():
    d = diff({"group": 3}, {"group": 4, "previous": 9, "status": leads.STATUS_CHANGED},
             {"group": 5})
    j = leads_rule.judge(d, communities(text_group(3), text_group(4), {"community": 5}),
                         communities(text_group(9)))
    rows = leads_run.rule_rows(plan(d), j, {3: frozenset({"u1", "u2"})}, leads_run.MODE_PRODUCTION)
    assert rows["candidate_id"].tolist() == ["c3", "c4"]
    assert rows["worth_a_look"].tolist() == [True, False]
    assert rows["members"].iloc[0] == ["u1", "u2"]
    assert (rows["adjudicator"] == leads_rule.ADJUDICATOR).all()


def test_the_gauge_counts_by_band_and_leaves_canaries_out():
    alerts = pd.DataFrame({"band": ["large", "small", "small", "large"],
                           "canary_id": [None, None, None, "check-x"]})
    text = leads_run.leads_gauge(alerts, reset=False, now=1_000)
    assert 'kma_leads_alerts{band="large"} 1\n' in text
    assert 'kma_leads_alerts{band="small"} 2\n' in text
    assert "kma_leads_last_success_timestamp_seconds 1000\n" in text
    assert "kma_leads_reset 0\n" in text


def test_a_quiet_day_reports_zero_in_both_bands():
    text = leads_run.leads_gauge(pd.DataFrame(columns=["band", "canary_id"]), reset=True, now=5)
    assert 'kma_leads_alerts{band="large"} 0\n' in text
    assert 'kma_leads_alerts{band="small"} 0\n' in text
    assert "kma_leads_reset 1\n" in text


def test_textfile_write_is_atomic_and_readable(tmp_path):
    path = textfile.write(tmp_path / "tf", "x.prom", "a 1\n")
    assert path.read_text() == "a 1\n"
    assert oct(path.stat().st_mode & 0o777) == "0o644"
    assert [p.name for p in (tmp_path / "tf").iterdir()] == ["x.prom"]


# -- canary evaluation --------------------------------------------------------


def test_a_planted_group_landing_together_in_a_flagged_community_is_detected():
    planted = [f"canary:t:{i}" for i in range(4)]
    members = pd.DataFrame({"user_id": planted + ["real"], "community": [7, 7, 7, 2, 7]})
    out = canary_check.evaluate(planted, members, communities(text_group(7, size=20)))
    assert out["detected"] and out["community"] == 7 and out["together"] == 0.75
    assert out["band"] == leads_rule.LARGE


def test_a_planted_group_in_a_retweet_only_community_is_not_detected():
    planted = [f"canary:t:{i}" for i in range(4)]
    members = pd.DataFrame({"user_id": planted, "community": [7] * 4})
    out = canary_check.evaluate(planted, members, communities({"community": 7}))
    assert not out["detected"]


def test_a_scattered_canary_is_not_detected():
    planted = [f"canary:t:{i}" for i in range(4)]
    members = pd.DataFrame({"user_id": planted, "community": [1, 2, 3, 3]})
    out = canary_check.evaluate(planted, members,
                                communities(text_group(1), text_group(2), text_group(3)))
    assert out["together"] == 0.5 and out["detected"]
    members = pd.DataFrame({"user_id": planted[:3], "community": [1, 2, 3]})
    assert not canary_check.evaluate(planted, members, communities(
        text_group(1), text_group(2), text_group(3)))["detected"]


def test_a_canary_missing_from_the_graph_is_not_detected():
    out = canary_check.evaluate(["canary:t:0"], pd.DataFrame(columns=["user_id", "community"]),
                                communities())
    assert out["in_graph"] == 0 and not out["detected"]
