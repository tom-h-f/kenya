import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from kma.db import connect
    from kma.semantic import assign_topics, search, topic_summary

    con = connect()
    con.execute("SET enable_progress_bar=false")
    return assign_topics, con, mo, search, topic_summary


@app.cell
def _(mo):
    mo.md("""
    # Narrative layer

    Semantic search + topic clustering over persisted post embeddings
    (`paraphrase-multilingual-mpnet-base-v2`, handles English / Swahili /
    Sheng). Run `embed_new` to cover the full corpus before reading topics -
    clusters sharpen with more data.
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
def _(assign_topics, con, mo, topic_summary):
    _df = assign_topics(con, min_cluster_size=60)
    _n = _df["topic"].nunique() - (1 if -1 in _df["topic"].values else 0)
    mo.vstack([
        mo.md(f"### Topics: **{_n}** clusters, **{int((_df['topic'] == -1).sum())}** unclustered of {len(_df)}"),
        topic_summary(_df)[["topic", "size", "terms"]],
    ])
    return


@app.cell
def _(con, mo):
    from kma.db import latest_labels

    _lab = latest_labels(con, "x").df()
    mo.vstack([
        mo.md(f"### Sentiment / emotion over {len(_lab):,} labelled posts"),
        mo.hstack([
            _lab["sentiment"].value_counts().rename("posts").to_frame(),
            _lab["emotion"].value_counts().rename("posts").to_frame(),
        ]),
    ])
    return


@app.cell
def _(con, mo):
    from kma.deltas import TRIBE_DISCLAIMER, slice_sentiment

    mo.vstack([
        mo.md("### Sentiment by region"),
        slice_sentiment(con, "region", min_posts=3).df(),
        mo.md(f"### Sentiment by community (EXPERIMENTAL)\n> {TRIBE_DISCLAIMER}"),
        slice_sentiment(con, "community", min_posts=3).df(),
    ])
    return


@app.cell
def _(mo):
    target = mo.ui.dropdown(
        ["Ruto", "Raila", "Gachagua", "Kalonzo", "IEBC", "Sifuna"],
        value="Ruto",
        label="Stance toward",
    )
    target
    return (target,)


@app.cell
def _(con, target):
    from kma.classify import stance

    stance(con, target.value, limit=60)[["stance", "stance_score", "author_handle", "text"]]
    return


if __name__ == "__main__":
    app.run()
