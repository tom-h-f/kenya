"""The reproduction gate: score `kma.coord2` against the IOHunter benchmark.

    uv run kma-iohunter-gate --data-dir ~/data/processed --country venezuela

The gate answers one question - did we copy the method correctly - by running
our detection against published datasets whose numbers someone else already
reported. It is not a Kenya measurement and must never be read as one.

## Which benchmark this is, and which it is not

The datasets are the IOHunter release (Minici, Luceri, Fabbri, Ferrara, AAAI
2025), CC-BY-4.0, https://zenodo.org/records/13357621. They are NOT the datasets
of Luceri et al. (WWW 2024, arXiv:2310.09884), whose method `kma.coord2` copies.
Same six countries and same construction, different inclusion rules: this
release has 533 positives for Venezuela where the WWW paper's Table 1 reports 33
drivers, and 275 for Russia against 3,487. So the WWW acceptance numbers (fused
unsupervised AUC 0.83 / F1 0.76) do NOT apply here and are not used.

The targets below are IOHunter's Table 2, Macro-F1, mean and standard deviation
over five seeds.

## What this gate does and does not test

Each country pickle ships the five trace networks already built (`coRT`,
`coURL`, `hashSeq`, `fastRT`, `tweetSim`) at a text-similarity threshold of 0.7,
plus the fused `graph`, the `labels` and five train/val/test `splits`.

So this exercises our FUSION and DETECTION against a known answer. It does NOT
exercise our trace construction, because the traces arrive pre-made. Those are
two separate claims about the copy and merging them would let a broken trace
builder pass a green gate.
"""

from __future__ import annotations

import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics

log = logging.getLogger(__name__)

COUNTRIES = ("UAE", "cuba", "russia", "venezuela", "iran", "china")

# IOHunter Table 2, Macro-F1 x 100, mean +/- std over five seeds. NodePruning is
# the unsupervised centrality baseline that `coord2.detect` reimplements;
# node2vec+RF is what `coord2.classify` reimplements.
TARGETS = pd.DataFrame(
    {
        "nodepruning": [84.66, 57.92, 87.65, 95.05, 60.83, 63.66],
        "nodepruning_std": [0.63, 2.07, 1.95, 1.47, 1.09, 0.70],
        "node2vec_rf": [96.97, 91.53, 83.43, 90.32, 80.50, 83.89],
        "node2vec_rf_std": [0.42, 1.11, 3.24, 2.14, 0.60, 1.06],
    },
    index=list(COUNTRIES),
)

# Their sweep: percentiles of the observed centrality distribution, not an
# absolute cut. This is the point the paper's prose obscures - it quotes 1e-2 as
# a conservative operating point, while the released code selects a percentile
# empirically. An absolute threshold does not survive a change of graph size,
# because eigenvector centrality is L2-normalised across nodes.
PERCENTILE_STEP = 0.5

# Isolated nodes are rewired to this many random non-isolated nodes before
# centrality is computed. Their value; it matters because eigenvector centrality
# is undefined-ish on a graph with disconnected singletons.
REWIRE_DEGREE = 5

DATASET_FILENAME = "0.7_datasets.pkl"


@dataclass(frozen=True)
class Country:
    name: str
    graph: nx.Graph
    labels: np.ndarray
    splits: dict | list
    traces: dict[str, nx.Graph]

    @property
    def positives(self) -> int:
        return int(self.labels.sum())


def load_country(data_dir: Path | str, country: str) -> Country:
    """Read one country's pickle from an unpacked IOHunter data directory."""
    path = Path(data_dir) / country / DATASET_FILENAME
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    traces = {k: payload[k] for k in ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim") if k in payload}
    return Country(
        name=country,
        graph=payload["graph"],
        labels=np.asarray(payload["labels"]).astype(int),
        splits=payload["splits"],
        traces=traces,
    )


def rewire_isolated(graph: nx.Graph, seed: int = 0) -> nx.Graph:
    """Connect every isolated node to `REWIRE_DEGREE` random non-isolated ones.

    Copied from the reference implementation rather than invented. Without it a
    fused similarity network is full of singletons, and the leading eigenvector
    is then computed over a graph whose components carry no relation to each
    other. Operates on a copy so the caller's graph is left alone.
    """
    out = graph.copy()
    rng = random.Random(seed)

    # Neighbour-based, not degree-based, matching the reference. A self-loop
    # contributes 2 to `degree`, so a node whose only edge is to itself looks
    # connected by degree and isolated by neighbours. It is isolated.
    isolated = []
    for node in out.nodes():
        neighbours = list(out.neighbors(node))
        if not neighbours or (len(neighbours) == 1 and neighbours[0] == node):
            isolated.append(node)
    connected = [node for node in out.nodes() if node not in set(isolated)]
    if not connected:
        return out

    for node in isolated:
        targets = rng.sample(connected, REWIRE_DEGREE) if len(connected) >= REWIRE_DEGREE else connected
        for target in targets:
            out.add_edge(node, target, weight=1.0)
        if out.has_edge(node, node):
            out.remove_edge(node, node)
    return out


def centrality_values(graph: nx.Graph, n_nodes: int) -> np.ndarray:
    """Unweighted eigenvector centrality, indexed by integer node id.

    Unweighted because the paper measured that weighting does not improve
    classification, and because the reference implementation calls
    `nx.eigenvector_centrality` whose `weight` default is None.
    """
    scores = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-8)
    out = np.zeros(n_nodes, dtype=float)
    for node, value in scores.items():
        out[int(node)] = value
    return out


def predictions_by_percentile(centrality: np.ndarray) -> list[np.ndarray]:
    """One label vector per swept percentile: 1 above the cut, 0 at or below."""
    out = []
    for percentile in np.arange(0, 100, PERCENTILE_STEP):
        cut = np.percentile(centrality, percentile)
        labels = np.ones(centrality.shape[0], dtype=int)
        labels[centrality <= cut] = 0
        out.append(labels)
    return out


def _macro_f1(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return float(metrics.f1_score(truth[mask], pred[mask], average="macro"))


def score_nodepruning(country: Country, seed: int = 0) -> pd.DataFrame:
    """Their protocol exactly: sweep the percentile, choose it on train+val,
    report on test, once per split.

    The threshold is chosen on held-out data rather than fixed, which is why
    this transfers across graphs of wildly different size. Selection uses
    train+val together because the baseline is unsupervised - there is nothing
    to train, so the two masks serve the same purpose.
    """
    graph = rewire_isolated(country.graph, seed=seed)
    centrality = centrality_values(graph, len(country.labels))
    candidates = predictions_by_percentile(centrality)

    rows = []
    # The release stores splits as a dict keyed by run id, not a list, so
    # iterating it directly would walk the integer keys.
    splits = country.splits
    ordered = [splits[k] for k in sorted(splits)] if isinstance(splits, dict) else list(splits)
    for split_id, split in enumerate(ordered):
        selection = np.logical_or(split["train"], split["val"])
        scores = [_macro_f1(country.labels, pred, selection) for pred in candidates]
        best = int(np.argmax(scores))
        rows.append(
            {
                "split": split_id,
                "percentile": float(np.arange(0, 100, PERCENTILE_STEP)[best]),
                "selection_macro_f1": scores[best],
                "test_macro_f1": _macro_f1(country.labels, candidates[best], split["test"]),
            }
        )
    return pd.DataFrame(rows)


def gate_report(country: Country, seed: int = 0) -> dict:
    """One country's result next to the published target."""
    per_split = score_nodepruning(country, seed=seed)
    got = per_split["test_macro_f1"].mean() * 100
    target = float(TARGETS.loc[country.name, "nodepruning"])
    tolerance = float(TARGETS.loc[country.name, "nodepruning_std"])
    return {
        "country": country.name,
        "nodes": country.graph.number_of_nodes(),
        "edges": country.graph.number_of_edges(),
        "positives": country.positives,
        "macro_f1": round(got, 2),
        "std": round(per_split["test_macro_f1"].std() * 100, 2),
        "target": target,
        "target_std": tolerance,
        "delta": round(got - target, 2),
        # Two published standard deviations is the band agreed before any result
        # was seen. Widening it afterwards would make the gate decorative.
        "within_2sd": bool(abs(got - target) <= 2 * tolerance),
        "median_percentile": float(per_split["percentile"].median()),
    }


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Score kma.coord2 against the IOHunter benchmark.")
    ap.add_argument("--data-dir", required=True, help="unpacked IOHunter data/processed directory")
    ap.add_argument("--country", action="append", choices=list(COUNTRIES), help="repeatable; default all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    countries = args.country or list(COUNTRIES)
    rows = []
    for name in countries:
        country = load_country(args.data_dir, name)
        log.info(
            "%s: %d nodes, %d edges, %d positives",
            name, country.graph.number_of_nodes(), country.graph.number_of_edges(), country.positives,
        )
        rows.append(gate_report(country, seed=args.seed))

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    passed = int(report["within_2sd"].sum())
    print(f"\nwithin 2 published SD: {passed}/{len(report)}")
    print(f"mean macro-F1 {report['macro_f1'].mean():.2f} against target {report['target'].mean():.2f}")


if __name__ == "__main__":
    main()
