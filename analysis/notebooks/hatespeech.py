import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from kma import authenticity, deltas, semantic, viz
    from kma.db import (
        authors_source,
        connect,
        embeddings_source,
        hatespeech_source,
        incitement_source,
        labels_source,
        posts_source,
    )

    # House theme once; seaborn then draws onto viz.new_fig() axes with viz
    # colours (never sns.set_theme, which would clobber the house rcParams).
    viz.use_theme()
    con = connect()
    con.execute("SET enable_progress_bar=false")

    # Severity encoding, fixed across every chart: colour follows the class,
    # never its rank. neither = recessive grey, offensive = orange, hate = red.
    CLASS_ORDER = ["neither", "offensive", "hate"]
    CLASS_COLORS = {"neither": viz.DEEMPH, "offensive": viz.ORANGE, "hate": viz.RED}
    return (
        CLASS_COLORS,
        CLASS_ORDER,
        authenticity,
        authors_source,
        con,
        deltas,
        embeddings_source,
        hatespeech_source,
        incitement_source,
        labels_source,
        mo,
        np,
        pd,
        posts_source,
        semantic,
        sns,
        viz,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Hate & offensive speech in the Kenya 2027 stream

        Every collected post is scored by the fine-tuned afro-xlmr 3-class
        classifier (`neither` / `offensive` / `hate`) and joined here to author
        origin, dangerous-speech rhetoric (the NCIC coded-term lexicon),
        sentiment/emotion, reach, and account behaviour.

        The questions this notebook answers: **how much** toxic speech is in the
        discourse, **whether it is trending** toward the election, **where** it
        comes from, **what kind** of dangerous speech it is, **whether it
        spreads**, and **what it is about**.

        `label` is the argmax class; `hate_flag` is the deploy triage rule
        `p_hate >= 0.28` (an explicit threshold, not argmax - the taxonomy pushes
        most coded menace without a protected-group target to `offensive`, so
        read the two together). **Everything here is triage for a human analyst,
        never an automated verdict**, and the geographic/ethnic lenses are coarse
        author-origin proxies, not statements about who any post targets.
        """
    )
    return


@app.cell
def _(
    authors_source,
    con,
    deltas,
    hatespeech_source,
    incitement_source,
    labels_source,
    pd,
    posts_source,
):
    # One enriched frame, loaded once. Post (latest) INNER hatespeech (100%
    # coverage), LEFT the author-origin proxy, incitement rhetoric (81%) and
    # sentiment/emotion (100%). Everything downstream is pandas over this.
    df = con.sql(
        f"""
        WITH p AS (
            SELECT platform_post_id, author_id, author_handle, created_at, text,
                   like_count, reply_count, repost_count, quote_count,
                   hashtags, lang
            FROM {posts_source("x")}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        ), a AS (
            SELECT * FROM {authors_source("x")}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_user_id ORDER BY collected_at DESC
            ) = 1
        ), h AS (
            SELECT * FROM {hatespeech_source("x")}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY scored_at DESC
            ) = 1
        ), i AS (
            SELECT * FROM {incitement_source("x")}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY scored_at DESC
            ) = 1
        ), l AS (
            SELECT * FROM {labels_source("x")}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY labeled_at DESC
            ) = 1
        )
        SELECT
            p.platform_post_id, p.author_id, p.author_handle, p.created_at,
            p.text, p.hashtags, p.lang,
            (p.like_count + p.reply_count + p.repost_count + p.quote_count) AS engagement,
            p.like_count,
            h.label, h.p_neither, h.p_offensive, h.p_hate, h.hate_flag,
            {deltas.region_case("a.location")} AS region,
            {deltas.community_case("a.location")} AS community,
            i.lexicon_categories,
            i.dehumanisation_score, i.violence_call_score, i.othering_score,
            l.sentiment, l.emotion
        FROM p
        JOIN h USING (platform_post_id)
        LEFT JOIN a ON p.author_id = a.platform_user_id
        LEFT JOIN i ON p.platform_post_id = i.platform_post_id
        LEFT JOIN l ON p.platform_post_id = l.platform_post_id
        """
    ).df()

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["is_offensive"] = df["label"] == "offensive"
    df["is_hate"] = df["label"] == "hate"
    df["toxic"] = (df["label"] != "neither") | df["hate_flag"]
    return (df,)


@app.cell
def _(df, mo):
    _n = len(df)
    _c = df["label"].value_counts()
    _flag = int(df["hate_flag"].sum())
    _tox = int(df["toxic"].sum())
    mo.vstack([
        mo.md(f"**{_n:,}** scored posts."),
        mo.hstack(
            [
                mo.stat(value=f"{int(_c.get('neither', 0)):,}", label="neither", bordered=True),
                mo.stat(value=f"{int(_c.get('offensive', 0)):,}", label="offensive", bordered=True),
                mo.stat(value=f"{int(_c.get('hate', 0)):,}", label="hate (argmax)", bordered=True),
                mo.stat(value=f"{_flag:,}", label="hate_flag (p≥0.28)", bordered=True),
                mo.stat(value=f"{100 * _tox / _n:.1f}%", label="flagged for review", bordered=True),
            ],
            widths="equal",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## A. How severe, and how confident

        The classifier emits a probability per class. `p_hate` separates cleanly
        for `neither`, but `offensive` and `hate` overlap - which is exactly why
        the deploy rule flags on a **0.28 threshold** (dashed line) rather than
        the argmax winner. Posts to the right of the line reach the human queue.
        """
    )
    return


@app.cell
def _(CLASS_COLORS, CLASS_ORDER, df, sns, viz):
    _fig, _ax = viz.new_fig(9, 4)
    sns.violinplot(
        data=df, x="label", y="p_hate", order=CLASS_ORDER,
        hue="label", hue_order=CLASS_ORDER, palette=CLASS_COLORS,
        legend=False, cut=0, density_norm="width", linewidth=1.0, ax=_ax,
    )
    _ax.axhline(0.28, color=viz.INK_2, linewidth=1.2, linestyle="--")
    _ax.text(2.42, 0.30, "flag ≥ 0.28", color=viz.INK_2, fontsize=9, va="bottom", ha="right")
    _ax.set_title("Model hate-probability by predicted class")
    _ax.set_xlabel("")
    _ax.set_ylabel("p_hate")
    _ax.grid(axis="x", visible=False)
    _fig
    return


@app.cell
def _(df, mo):
    klass = mo.ui.dropdown(
        {"all": "all", "hate": "hate", "offensive": "offensive", "neither": "neither"},
        value="hate", label="Class",
    )
    sort_by = mo.ui.dropdown(
        {"p_hate": "p_hate", "p_offensive": "p_offensive", "engagement": "engagement"},
        value="p_hate", label="Sort by",
    )
    mo.hstack([klass, sort_by], justify="start", gap=2)
    return klass, sort_by


@app.cell
def _(df, klass, mo, sort_by):
    _df = df if klass.value == "all" else df[df["label"] == klass.value]
    _df = _df.sort_values(sort_by.value, ascending=False)
    _cols = [
        "author_handle", "label", "p_offensive", "p_hate", "hate_flag",
        "engagement", "region", "text",
    ]
    mo.ui.table(
        _df[_cols].head(400).round({"p_offensive": 3, "p_hate": 3}),
        selection=None, pagination=True,
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## B. Is toxicity trending toward the election?

        Daily share of posts the model calls `offensive` or `hate`, with a 7-day
        rolling mean (bold). A sustained rise in the run-up to the 2027 vote is
        the signal worth watching; single-day spikes usually track a specific
        rally, announcement or viral row. Restricted to 2026 (99.8% of the
        corpus; a handful of older posts are dropped from the time axis).
        """
    )
    return


@app.cell
def _(df, pd, viz):
    _d = df[df["created_at"].dt.year == 2026].copy()
    _d["date"] = _d["created_at"].dt.floor("D")
    _daily = _d.groupby("date").agg(
        n=("label", "size"),
        off=("is_offensive", "sum"),
        hate=("is_hate", "sum"),
    )
    _daily = _daily[_daily["n"] >= 20]  # drop thin days that make the rate jumpy
    _pct = pd.DataFrame(
        {
            "offensive": 100 * _daily["off"] / _daily["n"],
            "hate": 100 * _daily["hate"] / _daily["n"],
        }
    )
    _roll = _pct.rolling(7, min_periods=3).mean()

    _fig, _ax = viz.new_fig(9, 4)
    for _c, _color in (("offensive", viz.ORANGE), ("hate", viz.RED)):
        _ax.plot(_pct.index, _pct[_c], color=_color, linewidth=0.8, alpha=0.28)
        _ax.plot(_roll.index, _roll[_c], color=_color, linewidth=2.2)
        _last = _roll[_c].dropna()
        if len(_last):
            _ax.annotate(
                _c, (_last.index[-1], _last.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=_color, fontsize=9,
            )
    _ax.set_title("Daily prevalence of offensive / hate posts (7-day mean)")
    _ax.set_ylabel("% of posts")
    _ax.set_ylim(bottom=0)
    _ax.grid(axis="x", visible=False)
    _fig.autofmt_xdate(rotation=0, ha="center")
    _fig
    return


@app.cell
def _(df, viz):
    _d = df[(df["created_at"].dt.year == 2026) & df["toxic"]].copy()
    _wk = _d.set_index("created_at").resample("W").size()
    _fig, _ax = viz.new_fig(9, 3.2)
    _ax.bar(_wk.index, _wk.values, width=5.5, color=viz.ORANGE, linewidth=0)
    _ax.set_title("Weekly count of posts flagged for review")
    _ax.set_ylabel("posts")
    _ax.grid(axis="x", visible=False)
    _fig.autofmt_xdate(rotation=0, ha="center")
    _fig
    return


@app.cell
def _(df, np, pd, sns, viz):
    _d = df[df["created_at"].dt.year == 2026].copy()
    _d["hour"] = _d["created_at"].dt.hour
    _d["dow"] = _d["created_at"].dt.dayofweek
    _g = _d.groupby(["dow", "hour"]).agg(n=("label", "size"), tox=("toxic", "sum"))
    _rate = (100 * _g["tox"] / _g["n"]).where(_g["n"] >= 15)
    _pivot = _rate.reset_index().pivot(index="dow", columns="hour", values=0)
    _pivot = _pivot.reindex(index=range(7), columns=range(24))
    _days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    _fig, _ax = viz.new_fig(9, 3.8)
    sns.heatmap(
        _pivot, ax=_ax, cmap=viz.SEQ_CMAP, linewidths=0, cbar_kws={"label": "% toxic"},
        yticklabels=_days, xticklabels=[str(h) if h % 3 == 0 else "" for h in range(24)],
    )
    _ax.set_title("Toxic-speech rate by hour and weekday (UTC)")
    _ax.set_xlabel("hour (UTC)")
    _ax.set_ylabel("")
    _ax.tick_params(rotation=0)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## C. Where it comes from

        Prevalence by the **author's** region, derived from self-declared
        profile location. Two hard caveats: only ~a third of accounts carry a
        mappable location, and this is where the *poster* is from, **not who a
        post targets**. Read as a coarse aggregate signal, never about an
        individual.
        """
    )
    return


@app.cell
def _(df, mo):
    _cov = df["region"].notna().mean()
    mo.callout(
        mo.md(
            f"**Location coverage: {_cov:.0%} of scored posts.** The rest have no "
            "mappable profile location and are excluded from the regional slices. "
            "Regions reflect author origin, not the target of any post."
        ),
        kind="warn",
    )
    return


@app.cell
def _(df, sns, viz):
    _g = df.dropna(subset=["region"]).groupby("region").agg(
        n=("label", "size"), off=("is_offensive", "mean"), hate=("is_hate", "mean"),
    )
    _g = _g[_g["n"] >= 200]
    _g["toxic_pct"] = 100 * (_g["off"] + _g["hate"])
    _g = _g.sort_values("toxic_pct", ascending=False).reset_index()

    _fig, _ax = viz.new_fig(9, 4.2)
    sns.barplot(data=_g, y="region", x="toxic_pct", color=viz.ORANGE, ax=_ax)
    for _i, _r in _g.iterrows():
        _ax.text(_r["toxic_pct"] + 0.05, _i, f"{_r['toxic_pct']:.1f}%  (n={int(_r['n']):,})",
                 va="center", fontsize=8.5, color=viz.INK_2)
    _ax.set_title("Offensive + hate rate by author region")
    _ax.set_xlabel("% of the region's posts")
    _ax.set_ylabel("")
    _ax.set_xlim(right=_g["toxic_pct"].max() * 1.35)
    _ax.grid(axis="y", visible=False)
    _fig
    return


@app.cell
def _(deltas, mo):
    mo.callout(
        mo.md(
            f"**Experimental ethnic-community proxy. {deltas.TRIBE_DISCLAIMER}** "
            "It infers the *author's* likely community from profile location and "
            "is wrong for any individual, for mixed/urban/diaspora users, and says "
            "nothing about who a post targets. Shown aggregate-only, for direction "
            "not measurement."
        ),
        kind="danger",
    )
    return


@app.cell
def _(df, sns, viz):
    _g = df.dropna(subset=["community"]).groupby("community").agg(
        n=("label", "size"), off=("is_offensive", "mean"), hate=("is_hate", "mean"),
    )
    _g = _g[_g["n"] >= 200]
    _g["toxic_pct"] = 100 * (_g["off"] + _g["hate"])
    _g = _g.sort_values("toxic_pct", ascending=False).reset_index()

    _fig, _ax = viz.new_fig(9, 4.2)
    sns.barplot(data=_g, y="community", x="toxic_pct", color=viz.NEUTRAL, ax=_ax)
    for _i, _r in _g.iterrows():
        _ax.text(_r["toxic_pct"] + 0.05, _i, f"{_r['toxic_pct']:.1f}%  (n={int(_r['n']):,})",
                 va="center", fontsize=8.5, color=viz.INK_2)
    _ax.set_title("Offensive + hate rate by author community (experimental proxy)")
    _ax.set_xlabel("% of the group's posts")
    _ax.set_ylabel("")
    _ax.set_xlim(right=_g["toxic_pct"].max() * 1.4)
    _ax.grid(axis="y", visible=False)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## D. What kind of dangerous speech

        The incitement lexicon tags posts with NCIC/PeaceTech coded-term
        categories - the specific rhetoric that preceded past Kenyan election
        violence. The first chart is how often each category appears among
        flagged posts; the second cross-tabs the model's class against the
        lexicon category, showing where the fine-tuned model and the coded-term
        scan corroborate each other.
        """
    )
    return


@app.cell
def _(df, sns, viz):
    _f = df[df["toxic"] & df["lexicon_categories"].notna()].explode("lexicon_categories")
    _f = _f[_f["lexicon_categories"].notna()]
    _counts = _f["lexicon_categories"].value_counts().reset_index()
    _counts.columns = ["category", "n"]

    _fig, _ax = viz.new_fig(9, 3.6)
    sns.barplot(data=_counts, y="category", x="n", color=viz.RED, ax=_ax)
    for _i, _r in _counts.iterrows():
        _ax.text(_r["n"], _i, f"  {int(_r['n']):,}", va="center", fontsize=9, color=viz.INK_2)
    _ax.set_title("Dangerous-speech rhetoric among flagged posts (lexicon category)")
    _ax.set_xlabel("flagged posts with a lexicon hit")
    _ax.set_ylabel("")
    _ax.set_xlim(right=_counts["n"].max() * 1.15)
    _ax.grid(axis="y", visible=False)
    _fig
    return


@app.cell
def _(CLASS_ORDER, df, np, pd, sns, viz):
    _f = df[df["lexicon_categories"].notna()].explode("lexicon_categories")
    _f = _f[_f["lexicon_categories"].notna()]
    _ct = pd.crosstab(_f["label"], _f["lexicon_categories"])
    _ct = _ct.reindex(index=[c for c in CLASS_ORDER if c in _ct.index])
    # column-normalise: within each rhetoric category, the class split
    _norm = 100 * _ct.div(_ct.sum(axis=0), axis=1)

    _fig, _ax = viz.new_fig(9, 3.2)
    sns.heatmap(
        _norm, ax=_ax, cmap=viz.SEQ_CMAP, annot=_ct.values, fmt=",d",
        annot_kws={"fontsize": 8}, linewidths=2, linecolor=viz.SURFACE,
        cbar_kws={"label": "% of category"},
    )
    _ax.set_title("Model class × lexicon rhetoric (cell = post count)")
    _ax.set_xlabel("")
    _ax.set_ylabel("")
    _ax.tick_params(rotation=0)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## E. Does it spread, and who spreads it

        Two amplification questions. First, reach: on average, toxic posts are
        **not** the most engaged - inflammatory content is not automatically
        viral here - but a thin tail does break out, and those are the ones that
        matter. Second, authorship: comparing the account-suspicion distribution
        of toxic-post authors against everyone shows whether hate is
        disproportionately pushed by inauthentic / coordinated-looking accounts.
        """
    )
    return


@app.cell
def _(CLASS_COLORS, CLASS_ORDER, df, mo, sns, viz):
    _g = df.groupby("label")["engagement"].mean().reindex(CLASS_ORDER)
    _fig, _ax = viz.new_fig(9, 3.0)
    sns.barplot(
        x=_g.index, y=_g.values, hue=_g.index, hue_order=CLASS_ORDER,
        palette=CLASS_COLORS, legend=False, ax=_ax,
    )
    for _i, _v in enumerate(_g.values):
        _ax.text(_i, _v, f"{_v:.0f}", ha="center", va="bottom", fontsize=9, color=viz.INK_2)
    _ax.set_title("Mean engagement (likes + replies + reposts + quotes) by class")
    _ax.set_xlabel("")
    _ax.set_ylabel("mean engagement")
    _ax.grid(axis="x", visible=False)
    mo.vstack([
        _fig,
        mo.callout(
            mo.md(
                "Higher average reach for `neither` means toxic content is not the "
                "loudest by default - triage should weight the **viral tail**, not "
                "raw volume."
            ),
            kind="neutral",
        ),
    ])
    return


@app.cell
def _(df, mo):
    _top = (
        df[df["is_hate"]].sort_values("engagement", ascending=False)
        .head(15)[["author_handle", "engagement", "like_count", "p_hate", "region", "text"]]
        .round({"p_hate": 3})
    )
    mo.vstack([
        mo.md("**Most-amplified `hate` posts** - the breakout tail that reached the furthest."),
        mo.ui.table(_top, selection=None, pagination=False),
    ])
    return


@app.cell
def _(authenticity, con, df, mo, sns, viz):
    # Live per-author suspicion (isolation-forest + heuristics), keyed to the
    # post author. Triage signal, not a bot verdict.
    try:
        _susp = authenticity.authenticity_score(con).set_index("platform_user_id")["suspicion"]
        _d = df.assign(suspicion=df["author_id"].map(_susp)).dropna(subset=["suspicion"])

        _fig, _ax = viz.new_fig(9, 3.8)
        for _grp, _mask, _color in (
            ("all others", ~_d["toxic"], viz.DEEMPH),
            ("toxic-post authors", _d["toxic"], viz.RED),
        ):
            sns.kdeplot(x=_d.loc[_mask, "suspicion"], ax=_ax, color=_color, fill=True,
                        alpha=0.25, linewidth=2, label=_grp, clip=(0, 1))
        _ax.set_title("Account-suspicion of toxic-post authors vs everyone else")
        _ax.set_xlabel("author suspicion score (0-1)")
        _ax.set_ylabel("density")
        _ax.legend()
        _ax.grid(axis="x", visible=False)
        _out = _fig
    except Exception as _e:  # authenticity needs enough authors / features
        _out = mo.md(f"_Author-suspicion lens unavailable: {type(_e).__name__}: {_e}_")
    _out
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## F. What the hate is about

        Two narrative lenses over the flagged subset. Hashtags are the cheap,
        legible one - which campaign tags carry the most toxic replies. Topic
        clusters (UMAP → HDBSCAN on the multilingual embeddings, labelled by
        distinctive c-TF-IDF terms) surface the latent conversations that the
        hashtags miss.
        """
    )
    return


@app.cell
def _(df, sns, viz):
    _f = df[df["toxic"] & df["hashtags"].notna()].explode("hashtags")
    _f = _f[_f["hashtags"].notna() & (_f["hashtags"].str.len() > 0)]
    _top = _f["hashtags"].str.lower().value_counts().head(15).reset_index()
    _top.columns = ["hashtag", "n"]

    _fig, _ax = viz.new_fig(9, 4.6)
    sns.barplot(data=_top, y="hashtag", x="n", color=viz.RED, ax=_ax)
    for _i, _r in _top.iterrows():
        _ax.text(_r["n"], _i, f"  {int(_r['n']):,}", va="center", fontsize=9, color=viz.INK_2)
    _ax.set_title("Top hashtags among flagged (offensive + hate) posts")
    _ax.set_xlabel("flagged posts")
    _ax.set_ylabel("")
    _ax.set_xlim(right=_top["n"].max() * 1.15)
    _ax.grid(axis="y", visible=False)
    _fig
    return


@app.cell
def _(con, df, embeddings_source, mo, np, semantic, viz):
    # Topic clusters on the flagged subset only (~thousands of posts, fast). We
    # reuse semantic.topic_summary for the c-TF-IDF labels and replicate its tiny
    # UMAP->HDBSCAN recipe on the subset embeddings.
    _flag = df[df["toxic"]][["platform_post_id", "text", "p_hate"]]
    con.register("_flag_ids", _flag[["platform_post_id"]])
    _emb = con.sql(
        f"""
        SELECT e.platform_post_id, e.embedding FROM (
            SELECT platform_post_id, embedding FROM {embeddings_source("x")}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY embedded_at DESC
            ) = 1
        ) e JOIN _flag_ids f USING (platform_post_id)
        """
    ).df()
    con.unregister("_flag_ids")
    _e = _emb.merge(_flag, on="platform_post_id")

    from sklearn.cluster import HDBSCAN
    from umap import UMAP

    _x = np.asarray(_e["embedding"].tolist(), dtype="float32")
    _x = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42).fit_transform(_x)
    _e["topic"] = HDBSCAN(min_cluster_size=25, min_samples=1, metric="euclidean").fit_predict(_x)

    _summ = semantic.topic_summary(_e)
    _hate = _e[_e["topic"] != -1].groupby("topic")["p_hate"].mean()
    _summ = _summ.assign(mean_p_hate=_summ["topic"].map(_hate)).head(14)

    _fig, _ax = viz.new_fig(9, 5.2)
    _shade = (_summ["mean_p_hate"] / max(_summ["mean_p_hate"].max(), 1e-9)).to_numpy()
    _colors = [tuple(c) for c in viz.SEQ_CMAP(_shade)]
    viz.hbars(_ax, _summ["label"], _summ["size"], colors=_colors)
    _ax.set_title("Largest narrative clusters in flagged speech (darker = more hateful)")
    _ax.set_xlabel("flagged posts in cluster")
    mo.vstack([mo.md("Cluster label = distinctive terms; shade = mean `p_hate`."), _fig])
    return


if __name__ == "__main__":
    app.run()
