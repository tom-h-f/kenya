import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from kma import coordination as co
    from kma.db import connect
    from kma.semantic import assign_topics

    con = connect()
    con.execute("SET enable_progress_bar=false")
    return assign_topics, co, con, mo


@app.cell
def _(co, mo):
    mo.md(
        f"""
        # Coordination networks (CIB triage)

        Detect account clusters acting in concert across behavioural channels,
        validated against null models and corroborated across layers.

        **Sampling caveat:** {co.SAMPLING_CAVEAT}

        Coordination alone is not malicious - the scorecard ranks clusters for
        human review using Phase 1 authenticity + Phase 2 narrative signals.
        """
    )
    return


@app.cell
def _(co, con, mo):
    cov = co.coverage(con)
    _row = cov.iloc[0]
    mo.vstack([
        mo.md(
            f"""
            ### Data coverage ({int(_row['posts']):,} posts)

            Wave B channels unlock as structured fields accrue: hashtags
            **{_row['hashtag_share']:.1%}**, URLs **{_row['url_share']:.1%}**,
            mentions **{_row['mention_share']:.1%}**.
            """
        ),
        cov,
    ])
    return (cov,)


@app.cell
def _(co, con):
    _channels = ["co_retweet", "co_reply", "text_sim"]
    edge_frames = {}
    for ch in _channels:
        _e = co.validated_edges(con, ch, min_repetition=2, tau=co.DEFAULT_TAU)
        edge_frames[ch] = _e
        print(
            f"{ch}: tested={len(_e)} "
            f"fdr={int(_e['sig_fdr'].sum())} "
            f"bonf={int(_e['sig_bonferroni'].sum())}"
        )
    edge_report = co.edge_report(edge_frames["co_retweet"])
    return _channels, edge_frames, edge_report


@app.cell
def _(edge_report, mo):
    mo.vstack([
        mo.md("### Edge-filter overlap (co-retweet)"),
        edge_report,
    ])
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
    return layers_inj, members_inj, null_rt, recovery, syn_ids, tr_inj


@app.cell
def _(mo, null_rt, recovery):
    mo.vstack([
        mo.md("### Evaluation: null baseline + synthetic injection"),
        mo.md(
            "Shuffled-object Bonferroni edges should be ~0. "
            "Synthetic cluster recovery on injected co-retweet traces:"
        ),
        null_rt,
        recovery,
    ])
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
def _(assign_topics, co, con, layers, members, mo, summary):
    topics = assign_topics(con, min_cluster_size=60)
    cards = co.scorecards(con, members, layers, topics=topics) if len(members) else None
    iv = (
        co.internal_validation(con, members, n_perm=200, min_size=2)
        if len(members)
        else None
    )
    mo.vstack([
        mo.md("### Detected clusters (co-retweet + text-sim, FDR-validated)"),
        summary if len(summary) else mo.md("_No clusters >= min_size on current sample._"),
        mo.md("### Scorecards (ranked by inauthenticity index)") if cards is not None else mo.md(""),
        cards.sort_values("inauthenticity_index", ascending=False)[
            [
                "cluster_id", "size", "n_channels", "suspicion_mean", "near_dup_rate",
                "dominant_topic", "inauthenticity_index",
            ]
        ].head(15)
        if cards is not None and len(cards)
        else mo.md(""),
        mo.md("### Internal validation (permutation vs random groups)") if iv is not None else mo.md(""),
        iv if iv is not None and len(iv) else mo.md(""),
    ])
    return cards, iv, topics


@app.cell
def _(co, con, layers, members, mo, summary):
    persist = mo.ui.checkbox(label="Persist this run to R2", value=False)
    persist
    return (persist,)


@app.cell
def _(co, con, layers, members, mo, persist, summary):
    if persist.value and len(members):
        keys = []
        for ch, edges in layers.items():
            for method, col in [
                ("svn_fdr", "sig_fdr"),
                ("svn_bonf", "sig_bonferroni"),
                ("pct", "sig_percentile"),
            ]:
                subset = edges[edges[col]]
                if len(subset):
                    keys.append(co.persist_edges(con, subset, ch, method))
        keys.append(co.persist_clusters(con, members, summary))
        mo.md("Persisted:\n\n" + "\n".join(f"- `{k}`" for k in keys))
    elif persist.value:
        mo.md("_Nothing to persist (no clusters detected)._")
    return


@app.cell
def _(co, con, members, mo):
    if len(members):
        _top = members["cluster_id"].value_counts().idxmax()
        drill = co.member_table(con, members[members["cluster_id"] == _top])
        mo.vstack([
            mo.md(f"### Member drill-down: largest cluster `{_top}`"),
            drill,
        ])
    return


if __name__ == "__main__":
    app.run()
