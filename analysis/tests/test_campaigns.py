"""Unit tests for campaign linking and verdict inheritance (no R2).

Motivated by a measured failure. Adjudicating confirmed information operations
2026-08-15, a reader got 4 of 6 - and the misses were the operation's own PURE
ENGAGEMENT-BAIT clusters, whose content is genuinely indistinguishable from the
reciprocal pods in our corpus. What connects them to the political part of the
same operation is not content or any statistic; it is who they amplify.
"""

from __future__ import annotations

import pandas as pd

from kma import coordination as co


def _members(spec: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"cluster_id": cid, "author_id": a} for cid, authors in spec.items()
        for a in authors
    ])


def _acts(pairs) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["actor", "object_author"])


def test_clusters_pushing_the_same_accounts_become_one_campaign():
    """The bait-plus-payload case: two clusters that look nothing alike in
    content but drive the same four accounts are one operation."""
    members = _members({0: ["a1", "a2"], 1: ["b1", "b2"]})
    targets = ["t1", "t2", "t3", "t4"]
    acts = _acts([("a1", t) for t in targets] + [("b1", t) for t in targets])

    got = co.campaigns(members, acts).set_index("cluster_id")

    assert got.loc[0, "campaign_id"] == got.loc[1, "campaign_id"]
    assert got.loc[0, "campaign_size"] == 2


def test_clusters_with_disjoint_targets_stay_separate():
    members = _members({0: ["a1"], 1: ["b1"]})
    acts = _acts(
        [("a1", f"t{i}") for i in range(5)] + [("b1", f"u{i}") for i in range(5)]
    )

    got = co.campaigns(members, acts).set_index("cluster_id")

    assert got.loc[0, "campaign_id"] != got.loc[1, "campaign_id"]
    assert set(got["campaign_size"]) == {1}


def test_one_shared_viral_target_is_not_a_campaign():
    """`min_shared` exists for this. Two clusters both retweeting one hugely
    popular post share a target by coincidence, not by coordination - and the
    hub cap does not catch it because this is the AMPLIFIED author, not the
    amplified object."""
    members = _members({0: ["a1"], 1: ["b1"]})
    acts = _acts(
        [("a1", "viral")] + [("a1", f"t{i}") for i in range(4)]
        + [("b1", "viral")] + [("b1", f"u{i}") for i in range(4)]
    )

    got = co.campaigns(members, acts).set_index("cluster_id")

    assert got.loc[0, "campaign_id"] != got.loc[1, "campaign_id"]


def test_a_large_cluster_does_not_absorb_everything_it_brushes():
    """`min_jaccard` exists for this. A cluster amplifying hundreds of accounts
    will share `min_shared` with almost anything; without the ratio it would
    swallow the whole run into one campaign."""
    members = _members({0: ["big"], 1: ["small"]})
    acts = _acts(
        [("big", f"t{i}") for i in range(200)]
        + [("small", f"t{i}") for i in range(4)]
    )

    got = co.campaigns(members, acts).set_index("cluster_id")

    assert got.loc[0, "campaign_id"] != got.loc[1, "campaign_id"]


def test_linking_is_transitive():
    """A-B and B-C linked means all three are one operation, which is what
    connected components give and pairwise labelling would not."""
    members = _members({0: ["a"], 1: ["b"], 2: ["c"]})
    acts = _acts(
        [("a", f"t{i}") for i in range(4)]
        + [("b", f"t{i}") for i in range(4)]
        + [("b", f"u{i}") for i in range(4)]
        + [("c", f"u{i}") for i in range(4)]
        + [("c", f"t{i}") for i in range(4)]
    )

    got = co.campaigns(members, acts).set_index("cluster_id")

    assert got["campaign_id"].nunique() == 1
    assert set(got["campaign_size"]) == {3}


def test_every_cluster_gets_a_row_even_when_linked_to_nothing():
    """A missing row would read as "no campaign" downstream, which is different
    from "its own campaign of one"."""
    members = _members({0: ["a"], 1: ["b"], 2: ["c"]})

    got = co.campaigns(members, _acts([]))

    assert len(got) == 3
    assert set(got["campaign_size"]) == {1}
    assert got["campaign_id"].nunique() == 3


def test_no_clusters_is_an_empty_frame():
    got = co.campaigns(pd.DataFrame(columns=["cluster_id", "author_id"]), _acts([]))

    assert got.empty
    assert list(got.columns) == [
        "cluster_id", "campaign_id", "campaign_size", "n_linked_clusters"
    ]


# --- verdict inheritance ----------------------------------------------------


def test_a_bait_cluster_inherits_its_campaigns_operation_verdict():
    """The measured miss mode, fixed. IO-3 was pure follow-back bait in the same
    operation as clusters carrying explicit political payload."""
    verdicts = pd.DataFrame([
        {"cluster_id": 0, "cluster_type": "influence_operation"},
        {"cluster_id": 1, "cluster_type": "engagement_pod"},
    ])
    campaign_map = pd.DataFrame([
        {"cluster_id": 0, "campaign_id": 0},
        {"cluster_id": 1, "campaign_id": 0},
    ])

    got = co.inherit_verdicts(verdicts, campaign_map).set_index("cluster_id")

    assert got.loc[1, "cluster_type"] == "influence_operation"
    assert got.loc[1, "inherited_from"] == 0


def test_inheritance_is_one_directional():
    """An operation must never be downgraded by the pods it keeps company with.
    The asymmetry is the whole point."""
    verdicts = pd.DataFrame([
        {"cluster_id": 0, "cluster_type": "influence_operation"},
        {"cluster_id": 1, "cluster_type": "engagement_pod"},
        {"cluster_id": 2, "cluster_type": "engagement_pod"},
    ])
    campaign_map = pd.DataFrame([
        {"cluster_id": c, "campaign_id": 0} for c in (0, 1, 2)
    ])

    got = co.inherit_verdicts(verdicts, campaign_map).set_index("cluster_id")

    assert got.loc[0, "cluster_type"] == "influence_operation"
    assert pd.isna(got.loc[0, "inherited_from"]), "a first-hand call is not inherited"


def test_a_verdict_reached_on_its_own_evidence_is_marked_as_such():
    """A reader has to be able to tell "judged" from "judged by association".
    An untraceable verdict is not usable as evidence."""
    verdicts = pd.DataFrame([
        {"cluster_id": 0, "cluster_type": "engagement_pod"},
        {"cluster_id": 1, "cluster_type": "engagement_pod"},
    ])
    campaign_map = pd.DataFrame([
        {"cluster_id": 0, "campaign_id": 0}, {"cluster_id": 1, "campaign_id": 0},
    ])

    got = co.inherit_verdicts(verdicts, campaign_map)

    assert got["inherited_from"].isna().all()


def test_separate_campaigns_do_not_cross_contaminate():
    verdicts = pd.DataFrame([
        {"cluster_id": 0, "cluster_type": "influence_operation"},
        {"cluster_id": 1, "cluster_type": "engagement_pod"},
    ])
    campaign_map = pd.DataFrame([
        {"cluster_id": 0, "campaign_id": 0}, {"cluster_id": 1, "campaign_id": 1},
    ])

    got = co.inherit_verdicts(verdicts, campaign_map).set_index("cluster_id")

    assert got.loc[1, "cluster_type"] == "engagement_pod"
    assert pd.isna(got.loc[1, "inherited_from"])


def test_inheritance_on_empty_input_is_a_noop():
    empty = pd.DataFrame(columns=["cluster_id", "cluster_type"])
    assert co.inherit_verdicts(empty, pd.DataFrame()).empty
