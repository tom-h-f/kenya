"""Q2: does the IOHunter reproduction survive without the text trace?

    cd analysis && uv run python investigations/2026-10-10-text-trace-value/10_benchmark.py \
        --data-dir ~/Code/Data/iohunter/processed [--country venezuela]

Uses the repo's own gate (`kma.iohunter_gate.score_nodepruning`, the
unsupervised Macro-F1 that `coord2.detect` is gated on) unchanged. Only the
fused graph differs: rebuilt as the union of the released trace networks with
one trace left out, on the released graph's own node set. The all-five arm is
checked against the released fused graph edge for edge before anything is read.

Arms: all five; each trace left out (text = `tweetSim`); text alone.
Output: out/benchmark.csv, one row per (country, arm).
"""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import networkx as nx
import pandas as pd

from kma import iohunter_gate as gate

from common import OUT

TRACES = ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")


def union_graph(country: gate.Country, keep: tuple[str, ...],
                text_loops: bool = True, loops: bool = True) -> nx.Graph:
    """Union of the kept trace networks on the released node set. With
    `text_loops=False` the `tweetSim` self-loops are dropped: our own trace
    never links an account to itself (`users[i] != users[j]`), the release's
    does."""
    g = nx.Graph()
    g.add_nodes_from(country.graph.nodes())
    for name in keep:
        trace = country.traces.get(name)
        if trace is None:
            continue
        for a, b in trace.edges():
            if a == b and (not loops or (name == "tweetSim" and not text_loops)):
                continue
            g.add_edge(a, b)
    return g


def edge_set(g: nx.Graph) -> set:
    return {tuple(sorted((int(a), int(b)))) for a, b in g.edges() if a != b}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--country", action="append", choices=list(gate.COUNTRIES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("bench")

    rows = []
    for name in args.country or list(gate.COUNTRIES):
        country = gate.load_country(args.data_dir.expanduser(), name)
        present = [t for t in TRACES if t in country.traces]
        sizes = {t: country.traces[t].number_of_edges() for t in present}
        released = edge_set(country.graph)
        rebuilt = edge_set(union_graph(country, tuple(present)))
        log.info("%s: nodes %d, positives %d, released fused edges %d, trace edges %s; "
                 "union of traces == released graph: %s (released-only %d, trace-only %d)",
                 name, country.graph.number_of_nodes(), country.positives, len(released), sizes,
                 released == rebuilt, len(released - rebuilt), len(rebuilt - released))
        text_all = edge_set(union_graph(country, ("tweetSim",)))
        text_edges = {e for e in text_all if e[0] != e[1]}
        others = edge_set(union_graph(country, tuple(t for t in present if t != "tweetSim")))
        arms = {"all": present, "no_text": [t for t in present if t != "tweetSim"],
                "text_only": ["tweetSim"], "all_text_pairs_only": present,
                "text_pairs_only": ["tweetSim"], "noloops_all": present,
                "noloops_no_text": [t for t in present if t != "tweetSim"]}
        arms |= {f"no_{t}": [x for x in present if x != t] for t in present if t != "tweetSim"}
        for arm, keep in arms.items():
            g = union_graph(country, tuple(keep), text_loops="pairs_only" not in arm,
                            loops=not arm.startswith("noloops_"))
            per_split = gate.score_nodepruning(replace(country, graph=g), seed=args.seed)
            got = per_split["test_macro_f1"].mean() * 100
            target = float(gate.TARGETS.loc[name, "nodepruning"])
            sd = float(gate.TARGETS.loc[name, "nodepruning_std"])
            rows.append({
                "country": name, "arm": arm, "traces": "+".join(keep),
                "nodes": g.number_of_nodes(), "edges": len(edge_set(g)),
                "macro_f1": round(got, 2), "split_sd": round(per_split["test_macro_f1"].std() * 100, 2),
                "published": target, "published_sd": sd,
                "delta_vs_published": round(got - target, 2),
                "within_2sd": bool(abs(got - target) <= 2 * sd),
                "text_pair_edges": len(text_edges),
                "text_self_loops": (nx.number_of_selfloops(country.traces["tweetSim"])
                                    if "tweetSim" in country.traces else 0),
                "text_positive_only": bool(
                    "tweetSim" in country.traces and not text_edges),
                "text_pair_edges_not_in_other_traces": len(text_edges - others),
                "released_equals_union": released == rebuilt,
            })
            log.info("%s %-12s F1 %.2f (published %.2f)", name, arm, got, target)
        pd.DataFrame(rows).to_csv(OUT / "benchmark.csv", index=False)

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
