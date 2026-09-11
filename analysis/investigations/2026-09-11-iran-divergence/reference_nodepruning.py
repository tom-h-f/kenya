"""IOHunter's own NodePruning baseline, run verbatim on the released pickles.

    uv run python investigations/2026-09-11-iran-divergence/reference_nodepruning.py \\
        <unpacked>/data/processed russia venezuela china iran

`handle_isolated_nodes` and the threshold/metric logic are copied as published in
mminici/InfoOpsGFM (src/my_utils.py 16e6a32, src/model_eval.py,
src/run_NodePruning.py a583560, seed 12121995). Left out: mlflow logging, and
the three metrics `get_best_threshold` computes but never argmaxes over.

This takes `kma.coord2` out of the loop entirely. If the reference code on the
reference data does not give the published number, the gap is in the benchmark,
not in our copy.
"""

import pickle
import random
import sys

import networkx as nx
import numpy as np
from sklearn import metrics

PUBLISHED = {
    "UAE": (84.66, 0.63), "cuba": (57.92, 2.07), "russia": (87.65, 1.95),
    "venezuela": (95.05, 1.47), "iran": (60.83, 1.09), "china": (63.66, 0.70),
}


def handle_isolated_nodes(graph):
    isolated_nodes = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if (len(neighbors) == 0) or (len(neighbors) == 1 and neighbors[0] == node):
            isolated_nodes.append(node)
    non_isolated_nodes = [node for node in graph.nodes() if node not in isolated_nodes]
    for isolated_node in isolated_nodes:
        if len(non_isolated_nodes) >= 5:
            random_nodes = random.sample(non_isolated_nodes, 5)
        else:
            random_nodes = non_isolated_nodes
        for target_node in random_nodes:
            graph.add_edge(isolated_node, target_node, weight=1.0)
        if graph.has_edge(isolated_node, isolated_node):
            graph.remove_edge(isolated_node, isolated_node)
    return isolated_nodes, graph


def get_best_threshold(ground_truth, pred_by_threshold, mask):
    return np.argmax([metrics.f1_score(ground_truth[mask], p[mask], average="macro") for p in pred_by_threshold])


def run(path, seed=12121995, num_splits=5):
    random.seed(seed)
    np.random.seed(seed)
    with open(path, "rb") as handle:
        datasets = pickle.load(handle)
    _, network = handle_isolated_nodes(datasets["graph"])
    centrality_values = nx.eigenvector_centrality(network)
    centrality_val_list = [-1] * network.number_of_nodes()
    for node_id in centrality_values:
        centrality_val_list[node_id] = centrality_values[node_id]
    centrality_val_list = np.array(centrality_val_list)

    predicted_labels_list = []
    for percentile in np.arange(0, 100, 0.5):
        centrality_threshold = np.percentile(centrality_val_list, percentile)
        predicted_labels = np.full(shape=network.number_of_nodes(), fill_value=1)
        predicted_labels[np.where(centrality_val_list <= centrality_threshold)[0]] = 0
        predicted_labels_list.append(np.copy(predicted_labels))

    labels = datasets["labels"]
    test = []
    for run_id in range(num_splits):
        split = datasets["splits"][run_id]
        mask = np.logical_or(split["train"], split["val"])
        best = get_best_threshold(labels, predicted_labels_list, mask)
        test.append(metrics.f1_score(labels[split["test"]], predicted_labels_list[best][split["test"]], average="macro"))
    return 100 * np.mean(test), 100 * np.std(test)


def main():
    data_dir, countries = sys.argv[1], sys.argv[2:] or list(PUBLISHED)
    print(f"{'country':10} {'reference code on release':>26} {'published':>16}")
    for country in countries:
        mean, std = run(f"{data_dir}/{country}/0.7_datasets.pkl")
        target, target_std = PUBLISHED[country]
        print(f"{country:10} {mean:18.2f} +/- {std:4.2f} {target:10.2f} +/- {target_std:4.2f}", flush=True)


if __name__ == "__main__":
    main()
