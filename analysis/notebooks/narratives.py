import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from kma import viz
    from kma.db import connect
    from kma.semantic import assign_topics, search, topic_summary

    viz.use_theme()
    con = connect()
    con.execute("SET enable_progress_bar=false")
    return assign_topics, con, mo, search, topic_summary, viz


@app.cell
def _(mo):
    mo.md("""
    # Narrative layer

    What is being said, by whom, in what mood. Every post is turned into a
    768-dimension **embedding** (`paraphrase-multilingual-mpnet-base-v2`, which
    understands English, Swahili and Sheng in one shared space) so that meaning,
    not exact words, drives search and clustering.

    **What each tool below means:**

    - **Semantic search** - ranks posts by *meaning* similarity to your query
      (cosine of the embeddings), so "IEBC will rig it" also surfaces
      paraphrases that share no keywords. `sim` is that cosine, 0-1, higher =
      closer.
    - **Topics** - UMAP + HDBSCAN group the embeddings into narrative clusters;
      each cluster is auto-named by its most *distinctive* terms (c-TF-IDF).
      Topic `-1` = unclustered/noise.
    - **Sentiment** - a classifier labels each post positive / neutral /
      negative. The by-slice charts map this to a **-1..+1 mean** (all negative
      = -1, all positive = +1, 0 = balanced).
    - **Stance** - zero-shot supports / neutral / opposes *toward a named
      target*, which is different from sentiment: a post can be angry (negative
      sentiment) yet *support* the person it is angry on behalf of.

    Run `embed_new` to cover the full corpus before reading topics - clusters
    sharpen with more data.
    """)
    return


@app.cell
def _(mo):
    query = mo.ui.text(
        value="IEBC cannot be trusted, the election will be rigged",
        label="Semantic search",
        full_width=True,
    )
    query
    return (query,)


@app.cell
def _(con, query, search):
    search(con, query.value, k=12)[["sim", "author_handle", "text"]]
    return


@app.cell
def _(assign_topics, con):
    topics_df = assign_topics(con, min_cluster_size=60)
    return (topics_df,)


@app.cell
def _(mo, topic_summary, topics_df):
    _n = topics_df["topic"].nunique() - (1 if -1 in topics_df["topic"].values else 0)
    summary_df = topic_summary(topics_df)
    mo.vstack([
        mo.md(
            f"### Topics: **{_n}** clusters, "
            f"**{int((topics_df['topic'] == -1).sum())}** unclustered of {len(topics_df)}"
        ),
        summary_df[["name", "size", "terms"]],
    ])
    return (summary_df,)


@app.cell
def _(summary_df, viz):
    _s = summary_df.sort_values("size", ascending=False).head(15)
    _fig, _ax = viz.new_fig(9, max(3.0, 0.42 * len(_s)))
    _fig.subplots_adjust(left=0.30)
    viz.hbars(_ax, _s["label"], _s["size"])
    _ax.set_title("Narrative clusters by size")
    _ax.set_xlabel("posts")
    _fig
    return


@app.cell
def _(con):
    from kma.db import latest_labels

    labels_df = latest_labels(con, "x").df()
    return (labels_df,)


@app.cell
def _(labels_df, mo, viz):
    _sent = labels_df["sentiment"].value_counts().to_dict()
    _fig, _ax = viz.new_fig(9, 1.8)
    viz.diverging_stack(
        _ax,
        [_sent],
        [f"{len(labels_df):,} posts"],
    )
    _ax.set_title("Sentiment share, negative | neutral | positive")
    viz.legend_swatches(
        _ax,
        [("negative", viz.DIV_NEG), ("neutral", viz.NEUTRAL), ("positive", viz.DIV_POS)],
        loc="upper right",
    )
    mo.vstack([mo.md(f"### Sentiment / emotion over {len(labels_df):,} labelled posts"), _fig])
    return


@app.cell
def _(labels_df, viz):
    _emo = labels_df["emotion"].value_counts()
    _fig, _ax = viz.new_fig(9, 3.4)
    viz.hbars(_ax, _emo.index, _emo.values)
    _ax.set_title("Emotion distribution")
    _ax.set_xlabel("posts")
    _fig
    return


@app.cell
def _(con):
    from kma.deltas import TRIBE_DISCLAIMER, slice_sentiment

    region_df = slice_sentiment(con, "region", min_posts=3).df()
    community_df = slice_sentiment(con, "community", min_posts=3).df()
    return TRIBE_DISCLAIMER, community_df, region_df


@app.cell
def _(TRIBE_DISCLAIMER, community_df, mo, region_df):
    mo.vstack([
        mo.md("### Sentiment by region"),
        region_df,
        mo.md(f"### Sentiment by community\n\n_{TRIBE_DISCLAIMER}_"),
        community_df,
    ])
    return


@app.cell
def _(community_df, mo, region_df, viz):
    _figs = []
    for _df, _title in (
        (region_df, "Mean sentiment by region"),
        (community_df, "Mean sentiment by community, experimental proxy"),
    ):
        if not len(_df):
            continue
        _d = _df.sort_values("mean_sentiment", ascending=False)
        _fig, _ax = viz.new_fig(9, max(2.2, 0.45 * len(_d)))
        _fig.subplots_adjust(left=0.16)
        viz.hbars(
            _ax,
            [f"{s}  ({int(n):,})" for s, n in zip(_d["slice"], _d["posts"])],
            _d["mean_sentiment"],
            colors=[viz.DIV_POS if v >= 0 else viz.DIV_NEG for v in _d["mean_sentiment"]],
            tip_fmt=lambda v: f"{v:+.2f}",
        )
        _ax.set_title(_title)
        _ax.set_xlabel("mean sentiment, -1 to 1  (label = slice and post count)")
        _figs.append(_fig)
    mo.vstack(_figs) if _figs else mo.md("_No labelled slices yet._")
    return


@app.cell
def _(mo):
    target = mo.ui.dropdown(
        ["Ruto", "Raila", "Gachagua", "Kalonzo", "IEBC", "Sifuna", "Murkomen", "Goons"],
        value="Ruto",
        label="Stance toward",
    )
    target
    return (target,)


@app.cell
def _(con, target):
    from kma.classify import stance

    stance_df = stance(con, target.value, limit=60)
    return (stance_df,)


@app.cell
def _(mo, stance_df, target):
    mo.vstack([
        mo.md(f"### Stance toward **{target.value}**"),
        stance_df[["stance", "stance_score", "author_handle", "text"]],
    ])
    return


@app.cell
def _(stance_df, target, viz):
    _counts = stance_df["stance"].value_counts().to_dict()
    _fig, _ax = viz.new_fig(9, 1.8)
    viz.diverging_stack(
        _ax,
        [_counts],
        [f"{len(stance_df)} posts"],
        order=("opposes", "neutral", "supports"),
    )
    _ax.set_title(f"Stance toward {target.value}, opposes | neutral | supports")
    viz.legend_swatches(
        _ax,
        [("opposes", viz.DIV_NEG), ("neutral", viz.NEUTRAL), ("supports", viz.DIV_POS)],
        loc="upper right",
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
