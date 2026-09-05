"""The gate has to fail when the copy is wrong, so these tests check the
protocol mechanics rather than the published numbers.

Fixtures are synthetic graphs with a known answer. Nothing here reads the
IOHunter release, so the suite still runs with no data and no network.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from kma import iohunter_gate as gate


def _planted(n_drivers: int = 8, n_organic: int = 40) -> gate.Country:
    """A dense driver clique inside a sparse organic ring, which is the shape
    centrality is supposed to find."""
    g = nx.Graph()
    drivers = list(range(n_drivers))
    organic = list(range(n_drivers, n_drivers + n_organic))
    for i in drivers:
        for j in drivers:
            if i < j:
                g.add_edge(i, j, weight=1.0)
    for a, b in zip(organic, organic[1:] + organic[:1], strict=True):
        g.add_edge(a, b, weight=1.0)
    g.add_edge(drivers[0], organic[0], weight=1.0)

    labels = np.array([1] * n_drivers + [0] * n_organic)
    total = n_drivers + n_organic
    mask = np.zeros(total, dtype=bool)
    mask[:] = True
    splits = {0: {"train": mask.copy(), "val": mask.copy(), "test": mask.copy()}}
    return gate.Country(name="venezuela", graph=g, labels=labels, splits=splits, traces={})


def test_rewire_connects_isolated_nodes_and_leaves_the_original_alone():
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_node(99)
    assert g.degree(99) == 0

    out = gate.rewire_isolated(g, seed=0)
    assert out.degree(99) == 3  # only three non-isolated nodes exist to attach to
    assert g.degree(99) == 0, "the caller's graph must not be mutated"


def test_rewire_drops_a_self_loop_on_a_rewired_node():
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(4, 4)
    out = gate.rewire_isolated(g, seed=0)
    assert not out.has_edge(4, 4)
    assert out.degree(4) > 0


def test_predictions_are_percentiles_not_a_fixed_cut():
    """The whole point of the sweep: the same relative structure at a different
    scale must give the same predictions. A fixed threshold would not."""
    small = np.linspace(0.01, 0.10, 100)
    large = small / 1000.0

    preds_small = gate.predictions_by_percentile(small)
    preds_large = gate.predictions_by_percentile(large)
    for a, b in zip(preds_small, preds_large, strict=True):
        assert np.array_equal(a, b)


def test_high_centrality_is_the_positive_class():
    centrality = np.array([0.9, 0.8, 0.1, 0.05])
    at_50th = gate.predictions_by_percentile(centrality)[100]  # index 100 = 50th percentile
    assert at_50th.tolist() == [1, 1, 0, 0]


def test_sweep_covers_the_full_percentile_range():
    preds = gate.predictions_by_percentile(np.linspace(0, 1, 50))
    assert len(preds) == int(100 / gate.PERCENTILE_STEP)


def test_scoring_recovers_a_planted_clique():
    country = _planted()
    result = gate.score_nodepruning(country)
    assert result["test_macro_f1"].iloc[0] > 0.9


def test_dict_and_list_splits_are_both_accepted():
    """The release stores splits as a dict keyed by run id; iterating it
    directly would walk integer keys and raise."""
    country = _planted()
    as_list = gate.Country(
        name=country.name,
        graph=country.graph,
        labels=country.labels,
        splits=[country.splits[0]],
        traces={},
    )
    assert np.isclose(
        gate.score_nodepruning(country)["test_macro_f1"].iloc[0],
        gate.score_nodepruning(as_list)["test_macro_f1"].iloc[0],
    )


def test_gate_report_compares_against_the_published_target():
    report = gate.gate_report(_planted())
    assert report["target"] == gate.TARGETS.loc["venezuela", "nodepruning"]
    assert report["country"] == "venezuela"
    assert "within_2sd" in report


def test_targets_cover_every_country():
    assert list(gate.TARGETS.index) == list(gate.COUNTRIES)
    assert gate.TARGETS["nodepruning"].notna().all()


@pytest.mark.parametrize("country", gate.COUNTRIES)
def test_published_targets_are_plausible_macro_f1_percentages(country):
    value = gate.TARGETS.loc[country, "nodepruning"]
    assert 0 < value <= 100
