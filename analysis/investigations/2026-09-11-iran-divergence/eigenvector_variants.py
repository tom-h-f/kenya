"""Does the eigenvector computation move Iran?

    uv run python investigations/2026-09-11-iran-divergence/eigenvector_variants.py <unpacked>/data/processed

Every variant uses the gate's own data, rewiring, percentile sweep, selection
and macro-F1. Only the centrality computation changes:

  ours      (A+I) power iteration, max_iter 1000, tol 1e-8   - kma.iohunter_gate today
  nx_dflt   (A+I) power iteration, max_iter 100,  tol 1e-6  - the reference's call
  old_nx    A     power iteration, max_iter 100,  tol 1e-6  - plain adjacency
  exact     leading eigenvector by eigsh                     - no iteration at all

Eigenvector centrality is not unique on a disconnected graph, so what nodes in
small components receive is decided by the iteration rather than the maths -
the reason this was worth ruling out on Iran's 2,372 components. Measured
2026-09-11: it moves nothing (Iran 71.43-71.46 under every variant).
"""

import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from kma import iohunter_gate as gate

COUNTRIES = ("russia", "venezuela", "china", "iran", "UAE", "cuba")


def adjacency(graph, n):
    edges = np.array([(int(u), int(v)) for u, v in graph.edges()], dtype=np.int64)
    loops = edges[:, 0] == edges[:, 1]
    rows = np.concatenate([edges[~loops, 0], edges[~loops, 1], edges[loops, 0]])
    cols = np.concatenate([edges[~loops, 1], edges[~loops, 0], edges[loops, 1]])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def power(matrix, *, shift, max_iter, tol):
    """networkx's loop in matrix form: uniform start, L2 normalise, stop when the
    L1 change is below n * tol. (None, max_iter) where networkx would raise."""
    n = matrix.shape[0]
    x = np.full(n, 1.0 / n)
    for iteration in range(1, max_iter + 1):
        last = x
        x = matrix @ last + (last if shift else 0.0)
        x = x / (np.linalg.norm(x) or 1.0)
        if np.abs(x - last).sum() < n * tol:
            return x, iteration
    return None, max_iter


def exact(matrix):
    _, vectors = eigsh(matrix.astype(float), k=1, which="LA")
    vector = vectors[:, 0]
    return -vector if vector.sum() < 0 else vector


def macro_f1(centrality, country):
    candidates = gate.predictions_by_percentile(centrality)
    splits = country.splits
    ordered = [splits[k] for k in sorted(splits)] if isinstance(splits, dict) else list(splits)
    results = []
    for split in ordered:
        selection = np.logical_or(split["train"], split["val"])
        scores = [gate._macro_f1(country.labels, pred, selection) for pred in candidates]
        results.append(gate._macro_f1(country.labels, candidates[int(np.argmax(scores))], split["test"]))
    return 100 * float(np.mean(results))


def main(data_dir):
    print(f"{'country':10} {'target':>14}  {'ours':>12} {'nx_dflt':>12} {'old_nx':>12} {'exact':>7}")
    for name in COUNTRIES:
        country = gate.load_country(data_dir, name)
        n = len(country.labels)
        graph = gate.rewire_isolated(country.graph, seed=0)
        matrix = adjacency(graph, n)

        mine, _ = power(matrix, shift=True, max_iter=1000, tol=1e-8)
        if not np.allclose(mine, gate.centrality_values(graph, n), atol=1e-7):
            raise SystemExit(f"{name}: matrix-form (A+I) iteration does not match networkx - fix before reading")

        target = gate.TARGETS.loc[name, "nodepruning"]
        band = 2 * gate.TARGETS.loc[name, "nodepruning_std"]
        cells = []
        for shift, max_iter, tol in ((True, 1000, 1e-8), (True, 100, 1e-6), (False, 100, 1e-6)):
            vector, iters = power(matrix, shift=shift, max_iter=max_iter, tol=tol)
            if vector is None:
                cells.append(f"{'no conv':>8}@{iters:<3}")
                continue
            score = macro_f1(vector, country)
            cells.append(f"{score:6.2f}{'*' if abs(score - target) <= band else ' '}@{iters:<4}")
        score = macro_f1(exact(matrix), country)
        cells.append(f"{score:6.2f}{'*' if abs(score - target) <= band else ' '}")
        print(f"{name:10} {target:6.2f} +/-{band:4.2f}  " + " ".join(cells), flush=True)
    print("\n* = within 2 published SD of the target;  @N = iterations to converge")


if __name__ == "__main__":
    main(sys.argv[1])
