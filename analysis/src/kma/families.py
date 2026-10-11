"""Hub-star families: the grouping to use wherever similar items are clustered.

A connected component of a similarity graph is transitive: A~B and B~C put A and
C together though they need not resemble each other. On the text trace that
chained topic-similar posts into one 4,000-post "campaign" (measured 2026-10-11).
A hub-star family has no such chain: the item with most partners claims every
unclaimed partner, so every member is a direct match of the family's hub, and an
item belongs to exactly one family.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def hub_families(rows: np.ndarray, cols: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Family label per node and a mask of nodes that sit in an edge.

    `rows`/`cols` are edge endpoints in either direction. Hubs are taken in
    descending degree, ties by index, so the result is deterministic. Nodes in
    no edge get their own label."""
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    graph = ((graph + graph.T) > 0).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    labels = np.full(n, -1)
    for hub in np.lexsort((np.arange(n), -degree)):
        if degree[hub] == 0:
            break
        if labels[hub] >= 0:
            continue
        partners = graph.indices[graph.indptr[hub] : graph.indptr[hub + 1]]
        labels[hub] = hub
        labels[partners[labels[partners] < 0]] = hub
    linked = degree > 0
    labels[~linked] = np.arange(n)[~linked]
    return labels, linked


def cosine_hub_families(x: np.ndarray, tau: float) -> np.ndarray:
    """Hub-star family label per row of L2-normalised `x`, joining rows at
    cosine >= `tau`. The drop-in for `connected_components` over the same graph."""
    from sklearn.neighbors import radius_neighbors_graph

    g = radius_neighbors_graph(x, radius=float(np.sqrt(2 - 2 * tau)), mode="connectivity").tocoo()
    labels, _ = hub_families(g.row, g.col, x.shape[0])
    return labels
