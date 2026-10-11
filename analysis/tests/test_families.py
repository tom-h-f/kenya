import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import radius_neighbors_graph

from kma import stories as st
from kma.families import cosine_hub_families, hub_families


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _walk(n, dim=64, step=0.9, seed=0):
    """A random walk on the sphere: neighbours at cosine ~`step`, far ends unrelated."""
    rng = np.random.default_rng(seed)
    cur = _unit(rng.normal(size=dim))
    out = [cur]
    for _ in range(n - 1):
        noise = rng.normal(size=dim)
        noise -= noise @ cur * cur
        cur = _unit(step * cur + np.sqrt(1 - step**2) * _unit(noise))
        out.append(cur)
    return np.array(out, dtype="float32")


def test_hub_families_path_is_not_one_family():
    rows = np.arange(0, 9)
    labels, linked = hub_families(rows, rows + 1, 10)
    assert linked.all()
    assert len(set(labels)) > 1
    assert all(len(np.flatnonzero(labels == f)) <= 3 for f in set(labels))


def test_hub_families_star_and_isolates():
    labels, linked = hub_families(np.array([0, 0, 0]), np.array([1, 2, 3]), 6)
    assert set(labels[:4]) == {0}
    assert list(linked) == [True] * 4 + [False] * 2
    assert labels[4] != labels[5]


def test_hub_families_ignores_self_loops_and_directions():
    labels, linked = hub_families(np.array([0, 1, 2]), np.array([0, 0, 2]), 3)
    assert labels[0] == labels[1] and linked.tolist() == [True, True, False]


def test_chain_of_dispersed_posts_is_one_component_but_not_one_family():
    x = _walk(120)
    tau = 0.85
    g = radius_neighbors_graph(x, radius=float(np.sqrt(2 - 2 * tau)), mode="connectivity")
    _, comp = connected_components(g, directed=False)
    authors = np.array([f"a{i}" for i in range(len(x))])
    assert len(set(comp)) == 1
    assert st._reject_chaining_blobs(x, comp, authors).all()

    labels = cosine_hub_families(x, tau)
    for f in set(labels):
        m = x[labels == f]
        if len(m) > 1:
            assert (m @ x[f] >= tau - 1e-5).all()
    assert len(set(labels)) > 5


def test_adjudication_families_do_not_chain():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "investigations/2026-10-11-text-only-adjudication/01_packets.py"
    spec = importlib.util.spec_from_file_location("packets01", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    words = [f"w{i}" for i in range(40)]
    posts = {}
    for i in range(12):
        posts[f"p{i}"] = {"text": " ".join(words[i : i + 8]), "author_id": f"a{i}",
                          "created_at": f"2026-10-01T0{i % 10}:00:00Z"}
    pairs = [(f"p{i}", f"p{i + 1}", 0.9, "k") for i in range(11)]
    out = mod.families({"posts": posts}, pairs)
    assert out, "adjacent posts share 7 of 9 words"
    assert max(f["n_posts"] for f in out.values()) <= 3
    for root, fam in out.items():
        assert root in fam["posts"] and fam["hub"] == root
