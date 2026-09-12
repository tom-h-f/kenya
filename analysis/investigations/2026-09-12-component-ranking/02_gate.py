"""The three rankings of 01_structure.py through the IOHunter reproduction gate.

    cd analysis && uv run python investigations/2026-09-12-component-ranking/02_gate.py ~/data/iohunter/processed

Same protocol as `kma.iohunter_gate.score_nodepruning` - isolated nodes
rewired, percentile swept in 0.5 steps, chosen on train+val, macro-F1 on test,
mean over the five splits - with only the centrality swapped. The gate is the
evidence that `coord2` copies the paper's method, so a ranking that replaces
eigenvector centrality in production has to pass it too, or the Kenya output
stops being the paper's method.

Iran is judged against the reference code's own figure on the release (71.31
+/- 0.76), per the 2026-09-11 Iran finding, as well as the published 60.83.
"""

import argparse
import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from kma import iohunter_gate as gate

HERE = Path(__file__).parent
REFERENCE_IRAN = (71.31, 0.76)

spec = importlib.util.spec_from_file_location("structure", HERE / "01_structure.py")
structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(structure)


def as_array(scores: pd.Series, n_nodes: int) -> np.ndarray:
    out = np.zeros(n_nodes, dtype=float)
    for node, value in scores.items():
        out[int(node)] = value
    return out


def rankings(graph: nx.Graph, n_nodes: int) -> dict[str, np.ndarray]:
    component, _ = structure.component_centrality(graph)
    return {
        "global": gate.centrality_values(graph, n_nodes),
        "component": as_array(component, n_nodes),
        "pagerank": as_array(pd.Series(nx.pagerank(graph, alpha=0.85, weight=None)), n_nodes),
    }


def test_f1(country: gate.Country, scores: np.ndarray) -> tuple[float, float]:
    candidates = gate.predictions_by_percentile(scores)
    splits = country.splits
    ordered = [splits[k] for k in sorted(splits)] if isinstance(splits, dict) else list(splits)
    results = []
    for split in ordered:
        selection = np.logical_or(split["train"], split["val"])
        best = int(np.argmax([gate._macro_f1(country.labels, pred, selection) for pred in candidates]))
        results.append(gate._macro_f1(country.labels, candidates[best], split["test"]))
    return float(np.mean(results) * 100), float(np.std(results, ddof=1) * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--country", action="append", choices=list(gate.COUNTRIES))
    args = ap.parse_args()

    rows = []
    for name in args.country or list(gate.COUNTRIES):
        country = gate.load_country(args.data_dir, name)
        graph = gate.rewire_isolated(country.graph, seed=0)
        components = nx.number_connected_components(graph)
        target, band = float(gate.TARGETS.loc[name, "nodepruning"]), float(gate.TARGETS.loc[name, "nodepruning_std"])
        if name == "iran":
            target, band = REFERENCE_IRAN
        row = {"country": name, "components after rewiring": components, "target": target}
        for method, scores in rankings(graph, len(country.labels)).items():
            mean, _ = test_f1(country, scores)
            row[method] = round(mean, 2)
            row[f"{method} ok"] = abs(mean - target) <= 2 * band
        rows.append(row)
        print(pd.DataFrame([row]).to_string(index=False, header=len(rows) == 1), flush=True)

    report = pd.DataFrame(rows)
    print()
    for method in ("global", "component", "pagerank"):
        print(f"{method:<10} within 2 SD: {int(report[f'{method} ok'].sum())}/{len(report)}   "
              f"mean macro-F1 {report[method].mean():.2f}")


if __name__ == "__main__":
    main()
