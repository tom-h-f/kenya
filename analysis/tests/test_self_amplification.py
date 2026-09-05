"""Unit tests for the pod discriminator (no R2).

Reciprocity is what separates an engagement pod from an influence operation:
pods retweet each other, so their amplification points inward. The measure is
useless raw, because a 3-account cluster has almost nothing of its own to
amplify - so what is under test here is mostly the size correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kma import coordination as co


def _acts(pairs) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["actor", "object_author"])


def _members(groups: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"cluster_id": cid, "author_id": a} for cid, ids in groups.items() for a in ids]
    )


def _pod(members_, outsiders, acts_each=6):
    """A closed pod: every member amplifies only other members."""
    return [
        (a, b) for a in members_ for b in members_ if a != b
    ][: acts_each * len(members_)]


def _open_group(members_, outsiders, acts_each=6):
    """The same accounts amplifying only outsiders."""
    return [(a, o) for a in members_ for o in outsiders][: acts_each * len(members_)]


OUTSIDERS = [f"o{i}" for i in range(60)]


def test_a_closed_pod_scores_far_above_its_null():
    pod = ["p0", "p1", "p2", "p3"]
    acts = _acts(_pod(pod, OUTSIDERS) + _open_group(OUTSIDERS, OUTSIDERS))

    got = co.self_amplification(
        None, _members({0: pod}), n_perm=200, seed=0, acts=acts
    ).set_index("cluster_id")

    assert got.loc[0, "self_amplification"] == pytest.approx(1.0)
    assert got.loc[0, "self_amplification_z"] > 3
    assert got.loc[0, "p_self_amplification"] < 0.01


def test_an_outward_facing_group_does_not():
    grp = ["g0", "g1", "g2", "g3"]
    acts = _acts(_open_group(grp, OUTSIDERS) + _open_group(OUTSIDERS, OUTSIDERS))

    got = co.self_amplification(
        None, _members({0: grp}), n_perm=200, seed=0, acts=acts
    ).set_index("cluster_id")

    assert got.loc[0, "self_amplification"] == pytest.approx(0.0)
    assert got.loc[0, "p_self_amplification"] > 0.05


def test_the_null_corrects_for_cluster_size():
    """The whole reason for the permutation. Raw internal share rises with size
    for free - a bigger group owns more of the corpus it might amplify - so two
    equally-pod-like clusters of different sizes must land at comparable z, even
    though their raw shares differ."""
    small, big = ["s0", "s1", "s2"], [f"b{i}" for i in range(12)]
    acts = _acts(
        _pod(small, OUTSIDERS) + _pod(big, OUTSIDERS)
        + _open_group(OUTSIDERS, OUTSIDERS)
    )

    got = co.self_amplification(
        None, _members({0: small, 1: big}), n_perm=200, seed=0, acts=acts
    ).set_index("cluster_id")

    assert got.loc[0, "self_amplification"] == pytest.approx(1.0)
    assert got.loc[1, "self_amplification"] == pytest.approx(1.0)
    # Both are maximally inward; neither may be dismissed for being small.
    assert got.loc[0, "self_amplification_z"] > 3
    assert got.loc[1, "self_amplification_z"] > 3
    assert got.loc[0, "p_self_amplification"] < 0.01
    assert got.loc[1, "p_self_amplification"] < 0.01


def test_a_null_drawn_from_the_whole_amplifier_pool_is_degenerate():
    """Why `clustered` is the default. With a pool far larger than the clusters,
    random same-size groups essentially never hit themselves, so the null mean
    is ~0 at EVERY size and carries no size information for the z to use.
    Measured on live data: 329,708 amplifiers, null means 0.0001-0.0030,
    corr(size, z) = +0.740 against +0.545 for the raw share."""
    pool = [f"x{i}" for i in range(400)]
    acts = _acts([(a, b) for a, b in zip(pool, pool[1:] + pool[:1])] * 3)

    got = co.self_amplification(
        None,
        _members({0: pool[:3], 1: pool[:12], 2: pool[:40]}),
        n_perm=200, seed=0, acts=acts, universe="amplifiers",
    ).set_index("cluster_id")

    means = got["self_amplification_null_mean"]
    assert means.max() < 0.15, "a wide pool should give a near-zero null"


def test_an_unknown_universe_is_rejected():
    with pytest.raises(ValueError):
        co.self_amplification(
            None, _members({0: ["a", "b", "c"]}), n_perm=10,
            acts=_acts([("a", "b")]), universe="everyone",
        )


def test_the_null_mean_grows_with_group_size():
    """Pins the confound itself: a random group of 12 hits its own members more
    often than a random group of 3 does, purely by owning more of the corpus.
    If this ever stops holding, the size correction is not correcting anything."""
    actors = [f"a{i}" for i in range(80)]
    acts = _acts([(a, b) for a in actors for b in actors if a != b][:4000])

    got = co.self_amplification(
        None,
        _members({0: actors[:3], 1: actors[:12], 2: actors[:40]}),
        n_perm=200, seed=0, acts=acts,
    ).set_index("cluster_id")

    means = got["self_amplification_null_mean"]
    assert means.loc[0] < means.loc[1] < means.loc[2]


def test_p_values_use_the_shared_floor_convention():
    """`internal_validation` already established (1 + count) / (1 + n); a second
    convention in the same module would make the two incomparable."""
    pod = ["p0", "p1", "p2", "p3"]
    acts = _acts(_pod(pod, OUTSIDERS) + _open_group(OUTSIDERS, OUTSIDERS))

    got = co.self_amplification(
        None, _members({0: pod}), n_perm=99, seed=0, acts=acts
    ).set_index("cluster_id")

    assert got.loc[0, "p_self_amplification"] >= 1 / (1 + 99)


def test_clusters_below_min_size_are_skipped():
    acts = _acts(_open_group(OUTSIDERS, OUTSIDERS))

    got = co.self_amplification(
        None, _members({0: ["o0", "o1"]}), n_perm=50, seed=0, acts=acts
    )

    assert got.empty


def test_no_acts_at_all_is_an_empty_frame_not_a_crash():
    got = co.self_amplification(
        None, _members({0: ["a", "b", "c"]}), n_perm=50, seed=0, acts=_acts([])
    )

    assert got.empty
    assert "self_amplification_z" in got.columns


def test_self_amplification_is_not_in_the_inauthenticity_index():
    """Score, do not gate. The index is a within-run percentile blend, so adding
    a component silently reweights every existing one and moves every ranking
    before the new signal has been watched over several runs."""
    assert "self_amplification" not in co.INAUTHENTICITY_WEIGHTS
    assert "self_amplification_z" not in co.INAUTHENTICITY_WEIGHTS
    assert set(co.INAUTHENTICITY_WEIGHTS) == {
        "bot_likeness", "synchrony", "homogeneity", "concealment", "corroboration"
    }


def test_attach_leaves_nulls_when_the_measure_fails(monkeypatch):
    """A scorecard must survive a dead discriminator - NaN means 'not
    measurable', which is a different claim from 0 ('points outward')."""
    def boom(*a, **k):
        raise RuntimeError("engagements unreadable")

    monkeypatch.setattr(co, "self_amplification", boom)
    card = pd.DataFrame([{"cluster_id": 0, "size": 3}])

    out = co.attach_self_amplification(None, _members({0: ["a", "b", "c"]}), card)

    assert len(out) == 1
    assert np.isnan(out.loc[0, "self_amplification_z"])
    assert "p_self_amplification" in out.columns


def test_attach_does_not_drop_unscored_clusters(monkeypatch):
    """Clusters below min_size get no row from the measure; a left join keeps
    them on the scorecard with nulls rather than deleting them."""
    monkeypatch.setattr(
        co, "self_amplification",
        lambda *a, **k: pd.DataFrame(
            [{"cluster_id": 0, "self_amplification": 0.5,
              "self_amplification_null_mean": 0.1,
              "self_amplification_z": 2.0, "p_self_amplification": 0.01}]
        ),
    )
    card = pd.DataFrame([{"cluster_id": 0}, {"cluster_id": 1}])

    out = co.attach_self_amplification(None, _members({0: ["a"]}), card)

    assert set(out["cluster_id"]) == {0, 1}
    assert out.set_index("cluster_id").loc[0, "self_amplification_z"] == 2.0
    assert np.isnan(out.set_index("cluster_id").loc[1, "self_amplification_z"])
