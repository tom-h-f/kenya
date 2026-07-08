"""Interactive coordination-network graphs (plotly).

Force-directed, pannable/zoomable account-account graphs where every node
carries its full triage context in the hover: handle, cluster, dominant
narrative, bot-suspicion, reach, and how many independent channels tie it to
its neighbours. Built for exploration - the static matplotlib graph in `viz`
is the print/export twin.

    from kma import coordination as co, netviz
    layers = co.build_layers(con, ["co_retweet", "text_sim"])
    members, summary = co.clusters(layers)
    attrs = netviz.node_attributes(con, members, co.cluster_names(con, members, summary))
    fig = netviz.cluster_network(co.aggregate_layers(layers), attrs)   # renders in marimo
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kma import viz

# Distinct hues for the largest clusters; everything past the cap folds into one
# muted "Other" so the palette never cycles (dataviz rule).
_CLUSTER_COLORS = viz.CATEGORICAL
_MAX_COLORED_CLUSTERS = len(_CLUSTER_COLORS)


def node_attributes(
    con,
    members: pd.DataFrame,
    cluster_names: pd.DataFrame | None = None,
    topics: pd.DataFrame | None = None,
    topic_names: pd.DataFrame | None = None,
    platform: str = "x",
) -> pd.DataFrame:
    """Per-account attributes for the graph hover, keyed by author_id: handle,
    cluster id + label, dominant narrative, suspicion, followers, posts."""
    from kma import coordination as co

    tbl = co.member_table(con, members, platform=platform, topics=topics)
    if cluster_names is not None and len(cluster_names):
        tbl = tbl.merge(cluster_names[["cluster_id", "name", "label"]], on="cluster_id", how="left")
    else:
        tbl["name"] = "cluster " + tbl["cluster_id"].astype(str)
        tbl["label"] = tbl["name"] + " (n=" + tbl.groupby("cluster_id")["author_id"].transform("size").astype(str) + ")"
    if topics is not None and topic_names is not None and "dominant_topic" in tbl.columns:
        tbl = tbl.merge(
            topic_names[["topic", "name"]].rename(
                columns={"topic": "dominant_topic", "name": "dominant_narrative"}
            ),
            on="dominant_topic",
            how="left",
        )
    if "dominant_narrative" not in tbl.columns:
        tbl["dominant_narrative"] = "-"
    tbl["dominant_narrative"] = tbl["dominant_narrative"].fillna("-")
    return tbl.set_index("author_id")


def _layout(edges: pd.DataFrame, seed: int = 0) -> dict:
    import networkx as nx

    g = nx.Graph()
    for r in edges.itertuples():
        g.add_edge(r.src, r.dst, weight=float(getattr(r, "weight", 1.0)))
    # k spreads nodes for readability; fewer iterations keep big graphs snappy.
    n = g.number_of_nodes()
    return nx.spring_layout(
        g, weight="weight", seed=seed, k=1.6 / np.sqrt(max(n, 1)),
        iterations=60 if n < 800 else 30,
    )


def _cluster_color_map(attrs: pd.DataFrame, node_ids: list) -> tuple[dict, list]:
    """cluster_id -> hue for the biggest clusters, the rest -> Other (gray)."""
    present = [c for c in attrs.loc[attrs.index.intersection(node_ids), "cluster_id"].dropna().unique()]
    sizes = attrs.loc[attrs.index.intersection(node_ids)].groupby("cluster_id").size()
    ordered = sizes.sort_values(ascending=False).index.tolist()
    cmap = {c: _CLUSTER_COLORS[i] for i, c in enumerate(ordered[:_MAX_COLORED_CLUSTERS])}
    legend = [(attrs.loc[attrs["cluster_id"] == c, "label"].iloc[0], cmap[c]) for c in cmap]
    if len(ordered) > _MAX_COLORED_CLUSTERS:
        legend.append((f"Other ({len(ordered) - _MAX_COLORED_CLUSTERS} clusters)", viz.DEEMPH))
    return cmap, legend


def cluster_network(
    aggregated: pd.DataFrame,
    attrs: pd.DataFrame,
    color_by: str = "cluster",
    focus_cluster=None,
    title: str = "Coordination network - drag to pan, scroll to zoom, hover a node for detail",
    height: int = 640,
    seed: int = 0,
):
    """Interactive force-directed graph of the validated multiplex.

    `aggregated` = aggregate_layers() output (src, dst, weight, channels,
    n_channels). `attrs` = node_attributes(). `color_by`:
    "cluster" (identity, default) or "suspicion" (sequential bot-likeness).
    `focus_cluster` restricts to one cluster's induced subgraph."""
    import plotly.graph_objects as go

    edges = aggregated
    if focus_cluster is not None:
        keep = set(attrs.index[attrs["cluster_id"] == focus_cluster])
        edges = edges[edges["src"].isin(keep) & edges["dst"].isin(keep)]
    if edges.empty:
        return go.Figure().add_annotation(
            text="No validated edges to draw on the current sample.",
            showarrow=False, font={"color": viz.INK_2},
        ).update_layout(height=height, paper_bgcolor=viz.SURFACE, plot_bgcolor=viz.SURFACE)

    pos = _layout(edges, seed)
    node_ids = list(pos)
    import networkx as nx

    g = nx.Graph()
    g.add_edges_from(zip(edges["src"], edges["dst"]))
    degree = dict(g.degree)

    # Edges: two traces so multi-channel (corroborated) edges stand out and can
    # be toggled. None separators break the polyline between segments.
    def _edge_xy(sub: pd.DataFrame):
        xs, ys = [], []
        for r in sub.itertuples():
            xs += [pos[r.src][0], pos[r.dst][0], None]
            ys += [pos[r.src][1], pos[r.dst][1], None]
        return xs, ys

    multi = edges[edges["n_channels"] >= 2]
    single = edges[edges["n_channels"] < 2]
    traces = []
    for sub, color, width, name in (
        (single, viz.DEEMPH, 1.0, "single-channel edge"),
        (multi, viz.VIOLET, 2.6, "corroborated edge (>= 2 channels)"),
    ):
        if len(sub):
            xs, ys = _edge_xy(sub)
            traces.append(
                go.Scatter(
                    x=xs, y=ys, mode="lines", hoverinfo="skip", name=name,
                    line={"color": color, "width": width}, opacity=0.55,
                )
            )

    sizes = np.array([8 + 3.2 * degree.get(n, 0) for n in node_ids])
    sizes = np.clip(sizes, 8, 42)

    def _hover(n) -> str:
        if n not in attrs.index:
            return f"<b>{n}</b>"
        a = attrs.loc[n]
        return (
            f"<b>@{a['handle']}</b><br>"
            f"cluster: {a['label']}<br>"
            f"narrative: {a['dominant_narrative']}<br>"
            f"suspicion: {a['suspicion']:.2f}   ·   anomaly: {a.get('anomaly_rank', float('nan')):.2f}<br>"
            f"followers: {int(a['followers_count']):,}   ·   posts: {int(a['n_posts'])}<br>"
            f"graph degree: {degree.get(n, 0)}"
            "<extra></extra>"
        )

    hover = [_hover(n) for n in node_ids]

    if color_by == "suspicion":
        vals = [float(attrs.loc[n, "suspicion"]) if n in attrs.index else 0.0 for n in node_ids]
        traces.append(
            go.Scatter(
                x=[pos[n][0] for n in node_ids], y=[pos[n][1] for n in node_ids],
                mode="markers", name="account", showlegend=False, hovertemplate=hover,
                marker={
                    "size": sizes, "color": vals, "colorscale": [[0, viz.SEQ_RAMP[0]], [1, viz.SEQ_RAMP[-1]]],
                    "cmin": 0, "cmax": 1,
                    "colorbar": {"title": "suspicion", "thickness": 12, "x": 1.02, "len": 0.8, "y": 0.5},
                    "line": {"color": viz.SURFACE, "width": 1.2},
                },
            )
        )
    else:
        cmap, _legend = _cluster_color_map(attrs, node_ids)
        colors = [
            cmap.get(attrs.loc[n, "cluster_id"], viz.DEEMPH) if n in attrs.index else viz.DEEMPH
            for n in node_ids
        ]
        # One marker trace per cluster hue so the legend names clusters and each
        # is independently toggleable.
        by_color: dict[str, list[int]] = {}
        for i, c in enumerate(colors):
            by_color.setdefault(c, []).append(i)
        label_for = {v: k for k, v in [(lbl, col) for lbl, col in _legend]}
        for c, idxs in by_color.items():
            traces.append(
                go.Scatter(
                    x=[pos[node_ids[i]][0] for i in idxs],
                    y=[pos[node_ids[i]][1] for i in idxs],
                    mode="markers", name=label_for.get(c, "Other"),
                    hovertemplate=[hover[i] for i in idxs],
                    marker={"size": sizes[idxs], "color": c,
                            "line": {"color": viz.SURFACE, "width": 1.2}},
                )
            )

    # suspicion mode puts a colorbar on the right, so lay the (2-entry) edge
    # legend horizontally along the top to avoid overlapping it.
    if color_by == "suspicion":
        legend = {"orientation": "h", "x": 0, "y": 1.06, "xanchor": "left"}
    else:
        legend = {"x": 1.01, "y": 1.0, "xanchor": "left"}
    fig = go.Figure(traces)
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": viz.INK}},
        height=height, hovermode="closest",
        paper_bgcolor=viz.SURFACE, plot_bgcolor=viz.SURFACE,
        font={"family": "sans-serif", "color": viz.INK_2},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 10}, **legend},
        margin={"l": 10, "r": 10, "t": 44, "b": 10},
        xaxis={"visible": False}, yaxis={"visible": False},
        dragmode="pan",
    )
    return fig
