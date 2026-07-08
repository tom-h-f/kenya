import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from kma import coordination as co
    from kma import netviz
    from kma import viz
    from kma.db import connect
    from kma.semantic import assign_topics, topic_summary

    viz.use_theme()
    con = connect()
    con.execute("SET enable_progress_bar=false")
    return assign_topics, co, con, mo, netviz, np, topic_summary, viz


@app.cell
def _(co, mo):
    mo.md(f"""
    # Coordination networks (CIB triage)

    Detecting **coordinated inauthentic behaviour**: groups of accounts acting
    in concert to amplify a narrative, where the coordination is concealed.

    **How the pipeline works, end to end:**

    1. **Traces** - every shared action becomes a link. Two accounts that
       retweet the same tweet, reply under the same post, or publish
       near-duplicate text are *co-acting*. Each action type is one *channel*.
    2. **Projection** - collapse those account x object links into an
       account x account graph per channel: edge weight = how many objects the
       pair both acted on.
    3. **Statistical validation** - keep only pairs whose overlap is
       *surprising* under a null model, not merely frequent. Sharing one viral
       tweet is not coordination; sharing eight obscure ones is. Two nulls run:
       the strict **Bonferroni** core and the sensitive **FDR** view.
    4. **Communities** - Leiden clustering finds dense groups; a group dense in
       **several independent channels** is far harder to explain as organic.
    5. **Characterization** - score each cluster with Phase 1 (bot-likeness) and
       Phase 2 (narrative) signals into a transparent triage index.
    6. **Evaluation** - a shuffled-data control (must find ~nothing) and a
       synthetic-injection test (must recover a planted cluster) prove the
       detector works before any cluster is trusted.

    **Sampling caveat:** {co.SAMPLING_CAVEAT}

    Coordination alone is **not** malicious - fan clubs and news outlets
    coordinate legitimately. The scorecard is a triage tool for human review,
    never an auto-label.
    """)
    return


@app.cell
def _(co, mo):
    mo.accordion({
        "Channels & metric glossary (what every column means)": mo.md(co.glossary_md())
    })
    return


@app.cell
def _(co, con, mo):
    _row = co.coverage(con).iloc[0]
    mo.vstack([
        mo.md(
            "### Data coverage\nWave B channels (co-hashtag / co-URL / co-mention) "
            "unlock as the share of posts carrying structured fields grows."
        ),
        mo.hstack(
            [
                mo.stat(value=f"{int(_row['posts']):,}", label="Posts", bordered=True),
                mo.stat(value=f"{_row['hashtag_share']:.1%}", label="With hashtags", bordered=True),
                mo.stat(value=f"{_row['url_share']:.1%}", label="With URLs", bordered=True),
                mo.stat(value=f"{_row['mention_share']:.1%}", label="With mentions", bordered=True),
            ],
            widths="equal",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Edge validation, per channel

    For each channel we test every account pair that shares at least
    `min_repetition` objects and keep only the statistically surprising
    ones. The bars below read left to right as **loose to strict**: all
    tested pairs, then those surviving FDR (sensitive), then Bonferroni
    (high-precision). The Jaccard table shows how much the SVN null and the
    naive top-percentile baseline agree - large divergence means the
    percentile filter is keeping popular-object noise the null rejects.
    """)
    return


@app.cell
def _(co, con):
    edge_frames = {
        _ch: co.validated_edges(con, _ch, min_repetition=2, tau=co.DEFAULT_TAU)
        for _ch in ["co_retweet", "co_reply", "text_sim"]
    }
    return (edge_frames,)


@app.cell
def _(co, edge_frames, mo, viz):
    _channels = list(edge_frames)
    _fig, _ax = viz.new_fig(9, 3.6)
    _fig.subplots_adjust(left=0.13)
    viz.grouped_hbars(
        _ax,
        _channels,
        [
            ("tested pairs", [len(edge_frames[c]) for c in _channels], viz.ORDINAL_3[0]),
            ("FDR-validated", [int(edge_frames[c]["sig_fdr"].sum()) for c in _channels], viz.ORDINAL_3[1]),
            ("Bonferroni", [int(edge_frames[c]["sig_bonferroni"].sum()) for c in _channels], viz.ORDINAL_3[2]),
        ],
        legend_loc="lower right",
    )
    _ax.set_title("Edges surviving each filter, loose to strict, per channel")
    _ax.set_xlabel("account pairs")
    mo.vstack([
        _fig,
        mo.md("**Edge-filter overlap, co-retweet** (Jaccard of surviving edge sets)"),
        co.edge_report(edge_frames["co_retweet"]),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why weight alone is not enough

    Each dot is a candidate pair: **x** = how many objects they share,
    **y** = how surprising that is (higher = smaller p-value). **Filled**
    dots survive FDR, **hollow** ones do not. Note that high weight does not
    guarantee significance - a pair can co-retweet many *popular* tweets and
    still be unsurprising, because the degree-corrected null already expects
    popular objects to be widely shared. Significance is the y-axis, not the
    x-axis.
    """)
    return


@app.cell
def _(edge_frames, mo, np, viz):
    import pandas as _pd

    _rows = [
        _e.assign(channel=_c) for _c, _e in edge_frames.items()
        if len(_e) and "p_value" in _e.columns
    ]
    if _rows:
        _all = _pd.concat(_rows, ignore_index=True)
        _all["nlp"] = -np.log10(_all["p_value"].clip(lower=1e-300))
        _fig, _ax = viz.new_fig(9, 4.6)
        _slots = {"co_retweet": viz.BLUE, "co_reply": viz.AQUA, "text_sim": viz.YELLOW}
        for _c, _g in _all.groupby("channel"):
            _sig = _g[_g["sig_fdr"]]
            _ns = _g[~_g["sig_fdr"]]
            _ax.scatter(_sig["weight"], _sig["nlp"], s=42, color=_slots[_c],
                        edgecolors=viz.SURFACE, linewidths=0.8, zorder=3)
            _ax.scatter(_ns["weight"], _ns["nlp"], s=42, facecolors="none",
                        edgecolors=_slots[_c], linewidths=1.4, zorder=2)
        viz.legend_swatches(
            _ax, [(_c, _col) for _c, _col in _slots.items()], loc="upper left"
        )
        _ax.set_title("Pair significance vs shared actions - filled marks survive FDR")
        _ax.set_xlabel("weight (shared objects / co-actions)")
        _ax.set_ylabel("-log10 p-value")
        _out = _fig
    else:
        _out = mo.md("_No tested pairs yet on the current sample._")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Does the detector actually work? Two falsification tests

    Before trusting any cluster we prove the method on data where we know
    the answer:

    - **Null baseline (false-positive control):** shuffle the real traces so
      all genuine coordination is destroyed, then re-run. The strict
      Bonferroni network **must be ~empty** - a non-empty result means the
      null model is broken, not that coordination was found.
    - **Synthetic injection (recovery test):** plant a known coordinated
      cluster (15 accounts co-retweeting 8 seed tweets within 60s) into the
      real data and measure how cleanly the pipeline recovers exactly those
      accounts. Precision/recall/F1 near 1.0 means it can find what it claims
      to find.
    """)
    return


@app.cell
def _(co, con):
    null_rt = co.null_baseline(con, "co_retweet", min_repetition=2)
    tr_inj, syn_ids = co.inject_synthetic(
        con, "co_retweet", k=15, n_seed_objects=8, window=60, seed=42
    )
    layers_inj = co.build_layers(con, ["co_retweet"], trace_table=tr_inj, min_repetition=2)
    members_inj, _ = co.clusters(layers_inj, min_size=3, resolution=co.DEFAULT_RESOLUTION)
    recovery = co.evaluate_recovery(members_inj, syn_ids)
    return null_rt, recovery


@app.cell
def _(mo, null_rt, recovery, viz):
    _fig, _ax = viz.new_fig(8, 2.8)
    _fig.subplots_adjust(left=0.13)
    viz.grouped_hbars(
        _ax,
        ["Bonferroni", "FDR"],
        [
            ("real traces", [null_rt.loc[0, "bonferroni"], null_rt.loc[0, "fdr"]], viz.BLUE),
            ("shuffled null", [null_rt.loc[1, "bonferroni"], null_rt.loc[1, "fdr"]], viz.DEEMPH),
        ],
        legend_loc="lower right",
    )
    _ax.set_title("False-positive control - shuffled input must give ~0 validated edges")
    _ax.set_xlabel("validated edges")
    mo.vstack([
        mo.md("### Evaluation: null baseline + synthetic injection"),
        _fig,
        mo.md("Recovery of a planted 15-account cluster (co-retweet, 8 seed tweets, 60s window):"),
        mo.hstack(
            [
                mo.stat(value=f"{recovery['precision']:.2f}", label="Precision", bordered=True),
                mo.stat(value=f"{recovery['recall']:.2f}", label="Recall", bordered=True),
                mo.stat(value=f"{recovery['f1']:.2f}", label="F1", bordered=True),
                mo.stat(value=f"{recovery['weighted_precision']:.2f}", label="Weighted precision", bordered=True),
            ],
            widths="equal",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Communities across the multiplex

    Each channel is one *layer* over the same set of accounts. We stack the
    validated layers, run **Leiden** community detection (CPM quality
    function), and drop singletons. The `resolution` (gamma) knob sets how
    dense a group must be to count - the sweep below shows which clusters
    are **robust** (persist across gamma) versus artefacts of one setting.
    Clusters supported by **two or more channels** are the high-confidence
    ones.
    """)
    return


@app.cell
def _(co, con):
    layers = co.build_layers(
        con,
        ["co_retweet", "text_sim"],
        min_repetition=2,
        tau=co.DEFAULT_TAU,
    )
    members, summary = co.clusters(
        layers, min_size=2, resolution=co.DEFAULT_RESOLUTION
    )
    return layers, members, summary


@app.cell
def _(co, layers):
    aggregated = co.aggregate_layers(layers)
    sweep = co.resolution_sweep(aggregated) if len(aggregated) else None
    return aggregated, sweep


@app.cell
def _(mo, sweep, viz):
    if sweep is not None and len(sweep):
        _fig, _ax = viz.new_fig(8, 3.4)
        _ax.plot(sweep["gamma"], sweep["clusters"], color=viz.BLUE, marker="o",
                 markersize=5.5, markeredgecolor=viz.SURFACE, markeredgewidth=1.2)
        _ax.plot(sweep["gamma"], sweep["largest"], color=viz.AQUA, marker="o",
                 markersize=5.5, markeredgecolor=viz.SURFACE, markeredgewidth=1.2)
        viz.legend_swatches(
            _ax, [("clusters found", viz.BLUE), ("largest cluster size", viz.AQUA)],
            loc="upper right",
        )
        _ax.set_xscale("log")
        _ax.set_title("Leiden resolution sweep - robust clusters persist across gamma")
        _ax.set_xlabel("CPM resolution (gamma, log)")
        _ax.set_ylabel("count")
        _out = _fig
    else:
        _out = mo.md("_No aggregated multiplex edges yet - sweep skipped._")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ### Explore the network

    The graph below is **interactive**: drag to pan, scroll to zoom, and
    hover any node to see the account handle, its cluster, dominant
    narrative, bot-suspicion, reach, and graph degree. Node **size** = graph
    degree (how many validated partners), node **colour** = cluster (or
    suspicion, via the toggle). **Violet edges** join accounts corroborated
    across two or more channels - the strongest evidence. Use the dropdown
    to isolate a single cluster's induced subgraph, and the legend to toggle
    edge types or individual clusters on and off.
    """)
    return


@app.cell
def _(cluster_names, mo):
    _opts = ["(all clusters)"]
    if cluster_names is not None and len(cluster_names):
        _opts += cluster_names.sort_values("label")["label"].tolist()
    focus_ui = mo.ui.dropdown(_opts, value="(all clusters)", label="Focus cluster")
    color_ui = mo.ui.radio(
        ["cluster", "suspicion"], value="cluster", label="Colour nodes by", inline=True
    )
    mo.hstack([focus_ui, color_ui], justify="start", gap=2)
    return color_ui, focus_ui


@app.cell
def _(cluster_names, con, members, netviz, topic_names, topics):
    node_attrs = (
        netviz.node_attributes(con, members, cluster_names, topics, topic_names)
        if len(members)
        else None
    )
    return (node_attrs,)


@app.cell
def _(aggregated, cluster_names, color_ui, focus_ui, mo, netviz, node_attrs):
    if node_attrs is not None and len(aggregated):
        _focus = None
        if focus_ui.value != "(all clusters)" and cluster_names is not None:
            _match = cluster_names.loc[cluster_names["label"] == focus_ui.value, "cluster_id"]
            _focus = _match.iloc[0] if len(_match) else None
        _out = netviz.cluster_network(
            aggregated, node_attrs, color_by=color_ui.value, focus_cluster=_focus
        )
    else:
        _out = mo.md("_No validated edges to draw yet on the current sample._")
    _out
    return


@app.cell
def _(assign_topics, co, con, layers, members, summary, topic_summary):
    topics = assign_topics(con, min_cluster_size=60)
    topic_names = topic_summary(topics)
    cluster_names = co.cluster_names(con, members, summary) if len(members) else None
    cards = co.scorecards(con, members, layers, topics=topics) if len(members) else None
    iv = (
        co.internal_validation(con, members, n_perm=200, min_size=2)
        if len(members)
        else None
    )
    return cards, cluster_names, iv, topic_names, topics


@app.cell
def _(cards, cluster_names, co, mo, summary, topic_names):
    _summary = (
        co.with_cluster_names(summary, cluster_names, drop_id=True)
        if cluster_names is not None and len(summary)
        else summary
    )
    _cards = (
        co.with_cluster_names(cards, cluster_names, drop_id=True)
        if cards is not None and cluster_names is not None and len(cards)
        else cards
    )
    if _cards is not None and len(_cards) and "dominant_topic" in _cards.columns:
        _cards = (
            _cards.merge(
                topic_names[["topic", "name"]].rename(
                    columns={"topic": "dominant_topic", "name": "dominant_narrative"}
                ),
                on="dominant_topic",
                how="left",
            ).drop(columns=["dominant_topic"])
        )
    _card_cols = [
        "name", "size", "n_channels", "suspicion_mean", "near_dup_rate",
        "dominant_narrative", "inauthenticity_index",
    ]
    mo.vstack([
        mo.md("### Detected clusters (co-retweet + text-sim, FDR-validated)"),
        _summary if len(summary) else mo.md("_No clusters >= min_size on current sample._"),
        mo.md("### Scorecards, ranked by inauthenticity index") if cards is not None else mo.md(""),
        _cards[_card_cols].head(15)
        if _cards is not None and len(_cards) and "dominant_narrative" in _cards.columns
        else (_cards.head(15) if _cards is not None and len(_cards) else mo.md("")),
    ])
    return


@app.cell
def _(cards, cluster_names, co, mo, np, viz):
    if cards is not None and len(cards) and cluster_names is not None:
        _c = co.with_cluster_names(
            cards.sort_values("inauthenticity_index", ascending=False).head(12),
            cluster_names,
        )
        _labels = _c["label"].tolist()
        _fig, _ax = viz.new_fig(9, max(2.4, 0.5 * len(_c)))
        _fig.subplots_adjust(left=0.18)
        _ys = np.arange(len(_c))[::-1]
        _left = np.zeros(len(_c))
        for _k, _color in zip(co.INAUTHENTICITY_WEIGHTS, viz.CATEGORICAL):
            _w = co.INAUTHENTICITY_WEIGHTS[_k] * _c[f"ix_{_k}"].to_numpy()
            _ax.barh(_ys, _w, left=_left, height=0.55, color=_color, linewidth=0)
            _left = _left + _w
        for _y, _v in zip(_ys, _left):
            _ax.text(_v + 0.012, _y, f"{_v:.2f}", va="center", fontsize=9, color=viz.INK_2)
        _ax.set_yticks(_ys, _labels)
        _ax.set_xlim(0, max(1.0, _left.max() * 1.12))
        _ax.set_ylim(-0.6, len(_c) - 0.4)
        _ax.grid(axis="y", visible=False)
        viz.legend_swatches(
            _ax,
            list(zip(co.INAUTHENTICITY_WEIGHTS, viz.CATEGORICAL)),
            loc="lower right",
        )
        _ax.set_title("Inauthenticity index by cluster - weighted component contributions")
        _ax.set_xlabel("index (weighted sum of percentile-ranked components)")
        _out = _fig
    else:
        _out = mo.md("_No clusters to score yet on the current sample._")
    _out
    return


@app.cell
def _(cluster_names, co, iv, mo, viz):
    if iv is not None and len(iv) and cluster_names is not None:
        _iv = co.with_cluster_names(iv, cluster_names)
        _fig, _ax = viz.new_fig(9, max(2.6, 0.55 * len(_iv)))
        _fig.subplots_adjust(left=0.16)
        viz.grouped_hbars(
            _ax,
            _iv["label"].tolist(),
            [
                ("suspicion", _iv["suspicion_effect"], viz.BLUE),
                ("narrative homogeneity", _iv["homogeneity_effect"], viz.AQUA),
            ],
            legend_loc="lower right",
        )
        _ax.set_title("Detected clusters vs random same-size groups - permutation effect size")
        _ax.set_xlabel("effect size, (observed - null mean) / null std")
        _out = _fig
    else:
        _out = mo.md("_No internal-validation results yet on the current sample._")
    _out
    return


@app.cell
def _(mo):
    persist = mo.ui.checkbox(label="Persist this run to R2", value=False)
    persist
    return (persist,)


@app.cell
def _(cluster_names, co, con, layers, members, mo, persist, summary):
    if persist.value and len(members):
        _keys = []
        for _ch, _edges in layers.items():
            for _method, _col in [
                ("svn_fdr", "sig_fdr"),
                ("svn_bonf", "sig_bonferroni"),
                ("pct", "sig_percentile"),
            ]:
                _subset = _edges[_edges[_col]]
                if len(_subset):
                    _keys.append(co.persist_edges(con, _subset, _ch, _method))
        _summary = (
            summary.merge(cluster_names, on="cluster_id", how="left")
            if cluster_names is not None
            else summary
        )
        _keys.append(co.persist_clusters(con, members, _summary))
        _out = mo.md("Persisted:\n\n" + "\n".join(f"- `{k}`" for k in _keys))
    elif persist.value:
        _out = mo.md("_Nothing to persist (no clusters detected)._")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(cluster_names, co, con, members, topic_names):
    if len(members) and cluster_names is not None:
        top_cluster = members["cluster_id"].value_counts().idxmax()
        top_label = cluster_names.loc[
            cluster_names["cluster_id"] == top_cluster, "label"
        ].iloc[0]
        drill = co.with_cluster_names(
            co.member_table(con, members[members["cluster_id"] == top_cluster]),
            cluster_names,
            drop_id=True,
        )
        if "dominant_topic" in drill.columns:
            drill = drill.merge(
                topic_names[["topic", "name"]].rename(
                    columns={"topic": "dominant_topic", "name": "dominant_narrative"}
                ),
                on="dominant_topic",
                how="left",
            ).drop(columns=["dominant_topic"])
    else:
        top_cluster, top_label, drill = None, None, None
    return drill, top_label


@app.cell
def _(drill, mo, top_label):
    (
        mo.vstack([
            mo.md(f"### Member drill-down: {top_label}"),
            drill,
        ])
        if drill is not None
        else mo.md("")
    )
    return


@app.cell
def _(drill, mo, viz):
    if drill is not None and len(drill):
        _fig, _ax = viz.new_fig(8, 4.6)
        _sc = _ax.scatter(
            drill["account_age_days"], drill["followers_count"].clip(lower=1),
            c=drill["suspicion"], cmap=viz.SEQ_CMAP, vmin=0, vmax=1,
            s=60, edgecolors=viz.SURFACE, linewidths=1.0,
        )
        for _, _r in drill.head(10).iterrows():
            _ax.annotate(
                _r["handle"], (_r["account_age_days"], max(_r["followers_count"], 1)),
                xytext=(7, 4), textcoords="offset points", fontsize=8, color=viz.INK_2,
            )
        _ax.set_yscale("log")
        _cb = _fig.colorbar(_sc, ax=_ax, pad=0.01)
        _cb.set_label("suspicion", color=viz.INK_2, fontsize=9)
        _cb.outline.set_visible(False)
        _ax.set_title("Cluster members - age vs followers, shaded by suspicion")
        _ax.set_xlabel("account age (days)")
        _ax.set_ylabel("followers")
        _out = _fig
    else:
        _out = mo.md("_No member cluster to drill into on the current sample._")
    _out
    return


if __name__ == "__main__":
    app.run()
