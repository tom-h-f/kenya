import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from kma import authenticity, deltas, measure, semantic, viz
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
        measure,
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
        origin, NCIC/PeaceTech coded-term rhetoric, sentiment/emotion, reach,
        and account behaviour.

        Rates below are **automated measurement**, not a human review queue.
        Two series, both excluding clearly off-domain (non-Kenya) posts:

        1. **Explicit** - model `offensive` / `hate` / `hate_flag` (`p_hate >= 0.28`)
        2. **Coded** - live NCIC lexicon hit corroborated by incitement NLI,
           gated by each term's `fp_risk` (high-fp terms like `nyoka` / `fukuza`
           need strong menace NLI and must beat `political_criticism`)

        Geographic/ethnic lenses remain coarse author-origin proxies, not
        statements about who any post targets.
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
    measure,
    pd,
    posts_source,
):
    # One enriched frame, loaded once. Post (latest) INNER hatespeech (100%
    # coverage), LEFT the author-origin proxy, incitement rhetoric (81%) and
    # sentiment/emotion (100%). Measurement columns (domain, coded_suspect,
    # explicit_toxic) are derived in pandas via kma.measure - not persisted.
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
            i.lexicon_hits, i.lexicon_categories,
            i.dehumanisation_score, i.violence_call_score, i.othering_score,
            i.political_criticism_score,
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
    df = measure.attach_measurement_columns(df)
    return (df,)


@app.cell
def _(df, mo):
    _n = len(df)
    _scope = df[df["in_kenya_scope"]]
    _n_scope = len(_scope)
    _off_n = int((~df["in_kenya_scope"]).sum())
    _c = df["label"].value_counts()
    _flag = int(df["hate_flag"].sum())
    _explicit = int(_scope["explicit_toxic"].sum())
    _coded = int(_scope["coded_suspect"].sum())
    _den = max(_n_scope, 1)
    mo.vstack([
        mo.md(f"**{_n:,}** scored posts; **{_n_scope:,}** in Kenya scope."),
        mo.hstack(
            [
                mo.stat(value=f"{int(_c.get('neither', 0)):,}", label="neither", bordered=True),
                mo.stat(value=f"{int(_c.get('offensive', 0)):,}", label="offensive", bordered=True),
                mo.stat(value=f"{int(_c.get('hate', 0)):,}", label="hate (argmax)", bordered=True),
                mo.stat(value=f"{_flag:,}", label="hate_flag (p≥0.28)", bordered=True),
                mo.stat(
                    value=f"{100 * _explicit / _den:.1f}%",
                    label="explicit toxic (Kenya)",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{100 * _coded / _den:.2f}%",
                    label="coded suspect (Kenya)",
                    bordered=True,
                ),
            ],
            widths="equal",
        ),
        mo.callout(
            mo.md(
                f"**{_off_n:,} off-domain posts excluded** from Kenya-scoped rates "
                "(US/Western markers without Kenya markers). Coded rate uses live "
                "lexicon scan ∩ NLI with `fp_risk` gates."
            ),
            kind="neutral",
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

        Daily share among **Kenya-scoped** posts: model `offensive` / `hate`,
        plus the separate `coded_suspect` series (lexicon ∩ NLI). Bold lines are
        7-day means. Restricted to 2026 (99.8% of the corpus).
        """
    )
    return


@app.cell
def _(df, pd, viz):
    _d = df[(df["created_at"].dt.year == 2026) & df["in_kenya_scope"]].copy()
    _d["date"] = _d["created_at"].dt.floor("D")
    _daily = _d.groupby("date").agg(
        n=("label", "size"),
        off=("is_offensive", "sum"),
        hate=("is_hate", "sum"),
        coded=("coded_suspect", "sum"),
    )
    _daily = _daily[_daily["n"] >= 20]  # drop thin days that make the rate jumpy
    _pct = pd.DataFrame(
        {
            "offensive": 100 * _daily["off"] / _daily["n"],
            "hate": 100 * _daily["hate"] / _daily["n"],
            "coded": 100 * _daily["coded"] / _daily["n"],
        }
    )
    _roll = _pct.rolling(7, min_periods=3).mean()

    _fig, _ax = viz.new_fig(9, 4)
    for _c, _color in (
        ("offensive", viz.ORANGE),
        ("hate", viz.RED),
        ("coded", viz.NEUTRAL),
    ):
        _ax.plot(_pct.index, _pct[_c], color=_color, linewidth=0.8, alpha=0.28)
        _ax.plot(_roll.index, _roll[_c], color=_color, linewidth=2.2)
        _last = _roll[_c].dropna()
        if len(_last):
            _ax.annotate(
                _c, (_last.index[-1], _last.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=_color, fontsize=9,
            )
    _ax.set_title("Daily prevalence (Kenya scope): offensive / hate / coded (7-day mean)")
    _ax.set_ylabel("% of Kenya-scoped posts")
    _ax.set_ylim(bottom=0)
    _ax.grid(axis="x", visible=False)
    _fig.autofmt_xdate(rotation=0, ha="center")
    _fig
    return


@app.cell
def _(df, viz):
    _d = df[
        (df["created_at"].dt.year == 2026)
        & df["in_kenya_scope"]
        & (df["explicit_toxic"] | df["coded_suspect"])
    ].copy()
    _wk = _d.set_index("created_at").resample("W").size()
    _fig, _ax = viz.new_fig(9, 3.2)
    _ax.bar(_wk.index, _wk.values, width=5.5, color=viz.ORANGE, linewidth=0)
    _ax.set_title("Weekly count of explicit-toxic or coded-suspect posts (Kenya scope)")
    _ax.set_ylabel("posts")
    _ax.grid(axis="x", visible=False)
    _fig.autofmt_xdate(rotation=0, ha="center")
    _fig
    return


@app.cell
def _(df, np, pd, sns, viz):
    # Heatmap keeps the explicit series for continuity with prior charts.
    _d = df[(df["created_at"].dt.year == 2026) & df["in_kenya_scope"]].copy()
    _d["hour"] = _d["created_at"].dt.hour
    _d["dow"] = _d["created_at"].dt.dayofweek
    _g = _d.groupby(["dow", "hour"]).agg(
        n=("label", "size"), tox=("explicit_toxic", "sum"),
    )
    _rate = (100 * _g["tox"] / _g["n"]).where(_g["n"] >= 15)
    _pivot = _rate.reset_index().pivot(index="dow", columns="hour", values=0)
    _pivot = _pivot.reindex(index=range(7), columns=range(24))
    _days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    _fig, _ax = viz.new_fig(9, 3.8)
    sns.heatmap(
        _pivot, ax=_ax, cmap=viz.SEQ_CMAP, linewidths=0,
        cbar_kws={"label": "% explicit toxic"},
        yticklabels=_days, xticklabels=[str(h) if h % 3 == 0 else "" for h in range(24)],
    )
    _ax.set_title("Explicit-toxic rate by hour and weekday (UTC, Kenya scope)")
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
    _g = df[df["in_kenya_scope"]].dropna(subset=["region"]).groupby("region").agg(
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
    _ax.set_title("Offensive + hate rate by author region (Kenya scope)")
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
    _g = df[df["in_kenya_scope"]].dropna(subset=["community"]).groupby("community").agg(
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
    _ax.set_title("Offensive + hate rate by author community (Kenya scope; experimental)")
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

        Live NCIC/PeaceTech lexicon categories among **`coded_suspect`** posts
        (lexicon ∩ NLI with `fp_risk` gates), Kenya-scoped. The crosstab is
        model class × live lexicon category on Kenya-scoped rows with a lexicon
        hit - where the fine-tuned model and the coded-term scan meet.
        """
    )
    return


@app.cell
def _(df, sns, viz):
    _f = df[df["coded_suspect_kenya"]].explode("lexicon_categories_live")
    _f = _f[_f["lexicon_categories_live"].notna()]
    _f = _f[_f["lexicon_categories_live"].map(lambda c: c is not None and str(c) not in ("", "nan"))]
    if _f.empty:
        _fig, _ax = viz.new_fig(9, 2.5)
        _ax.text(0.5, 0.5, "No coded_suspect posts with lexicon categories",
                 ha="center", va="center", transform=_ax.transAxes)
        _ax.set_axis_off()
    else:
        _counts = _f["lexicon_categories_live"].value_counts().reset_index()
        _counts.columns = ["category", "n"]
        _fig, _ax = viz.new_fig(9, 3.6)
        sns.barplot(data=_counts, y="category", x="n", color=viz.RED, ax=_ax)
        for _i, _r in _counts.iterrows():
            _ax.text(_r["n"], _i, f"  {int(_r['n']):,}", va="center", fontsize=9, color=viz.INK_2)
        _ax.set_title("Lexicon rhetoric among coded_suspect posts (Kenya scope)")
        _ax.set_xlabel("coded_suspect posts with a lexicon hit")
        _ax.set_ylabel("")
        _ax.set_xlim(right=_counts["n"].max() * 1.15)
        _ax.grid(axis="y", visible=False)
    _fig
    return


@app.cell
def _(CLASS_ORDER, df, pd, sns, viz):
    _f = df[df["in_kenya_scope"]].explode("lexicon_categories_live")
    _f = _f[_f["lexicon_categories_live"].notna()]
    if _f.empty:
        _fig, _ax = viz.new_fig(9, 2.5)
        _ax.text(0.5, 0.5, "No Kenya-scoped lexicon hits",
                 ha="center", va="center", transform=_ax.transAxes)
        _ax.set_axis_off()
    else:
        _ct = pd.crosstab(_f["label"], _f["lexicon_categories_live"])
        _ct = _ct.reindex(index=[c for c in CLASS_ORDER if c in _ct.index])
        _norm = 100 * _ct.div(_ct.sum(axis=0), axis=1)
        _fig, _ax = viz.new_fig(9, 3.2)
        sns.heatmap(
            _norm, ax=_ax, cmap=viz.SEQ_CMAP, annot=_ct.values, fmt=",d",
            annot_kws={"fontsize": 8}, linewidths=2, linecolor=viz.SURFACE,
            cbar_kws={"label": "% of category"},
        )
        _ax.set_title("Model class × live lexicon rhetoric (Kenya scope; cell = count)")
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

        Amplification on the Kenya-scoped slice. Mean engagement by model class;
        the viral tail table is `hate_flag` posts that are not off-domain (US/
        Western engagement pollution is dropped). Account-suspicion compares
        authors of explicit-toxic Kenya posts against everyone else.
        """
    )
    return


@app.cell
def _(CLASS_COLORS, CLASS_ORDER, df, mo, pd, sns, viz):
    _d = df[df["in_kenya_scope"]]
    _g = _d.groupby("label")["engagement"].mean().reindex(CLASS_ORDER)
    _fig, _ax = viz.new_fig(9, 3.0)
    sns.barplot(
        x=_g.index, y=_g.values, hue=_g.index, hue_order=CLASS_ORDER,
        palette=CLASS_COLORS, legend=False, ax=_ax,
    )
    for _i, _v in enumerate(_g.values):
        if pd.notna(_v):
            _ax.text(_i, _v, f"{_v:.0f}", ha="center", va="bottom", fontsize=9, color=viz.INK_2)
    _ax.set_title("Mean engagement by class (Kenya scope)")
    _ax.set_xlabel("")
    _ax.set_ylabel("mean engagement")
    _ax.grid(axis="x", visible=False)
    _off_hf = int((df["hate_flag"] & ~df["in_kenya_scope"]).sum())
    mo.vstack([
        _fig,
        mo.callout(
            mo.md(
                f"**{_off_hf:,} off-domain `hate_flag` posts excluded** from the "
                "amplification table below (engagement pollution from non-Kenya "
                "discourse)."
            ),
            kind="neutral",
        ),
    ])
    return


@app.cell
def _(df, mo):
    _top = (
        df[df["hate_flag"] & df["in_kenya_scope"]]
        .sort_values("engagement", ascending=False)
        .head(15)[["author_handle", "engagement", "like_count", "p_hate", "region", "domain", "text"]]
        .round({"p_hate": 3})
    )
    mo.vstack([
        mo.md(
            "**Most-amplified Kenya-scoped `hate_flag` posts** - breakout tail "
            "after dropping off-domain."
        ),
        mo.ui.table(_top, selection=None, pagination=False),
    ])
    return


@app.cell
def _(authenticity, con, df, mo, sns, viz):
    # Live per-author suspicion (isolation-forest + heuristics), keyed to the
    # post author. Measurement signal, not a bot verdict.
    try:
        _susp = authenticity.authenticity_score(con).set_index("platform_user_id")["suspicion"]
        _d = df.assign(suspicion=df["author_id"].map(_susp)).dropna(subset=["suspicion"])
        _d = _d[_d["in_kenya_scope"]]

        _fig, _ax = viz.new_fig(9, 3.8)
        for _grp, _mask, _color in (
            ("all others", ~_d["explicit_toxic"], viz.DEEMPH),
            ("explicit-toxic authors", _d["explicit_toxic"], viz.RED),
        ):
            sns.kdeplot(x=_d.loc[_mask, "suspicion"], ax=_ax, color=_color, fill=True,
                        alpha=0.25, linewidth=2, label=_grp, clip=(0, 1))
        _ax.set_title("Account-suspicion of explicit-toxic authors vs others (Kenya scope)")
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

        Narrative lenses over Kenya-scoped **explicit-toxic** posts. Hashtags
        are the cheap, legible cut; topic clusters (UMAP → HDBSCAN on the
        multilingual embeddings, labelled by distinctive c-TF-IDF terms)
        surface conversations the hashtags miss.
        """
    )
    return


@app.cell
def _(df, sns, viz):
    _f = df[df["explicit_toxic"] & df["hashtags"].notna()].explode("hashtags")
    _f = _f[_f["hashtags"].notna() & (_f["hashtags"].str.len() > 0)]
    _top = _f["hashtags"].str.lower().value_counts().head(15).reset_index()
    _top.columns = ["hashtag", "n"]

    _fig, _ax = viz.new_fig(9, 4.6)
    if _top.empty:
        _ax.text(0.5, 0.5, "No hashtags on explicit-toxic Kenya posts",
                 ha="center", va="center", transform=_ax.transAxes)
        _ax.set_axis_off()
    else:
        sns.barplot(data=_top, y="hashtag", x="n", color=viz.RED, ax=_ax)
        for _i, _r in _top.iterrows():
            _ax.text(_r["n"], _i, f"  {int(_r['n']):,}", va="center", fontsize=9, color=viz.INK_2)
        _ax.set_title("Top hashtags among explicit-toxic posts (Kenya scope)")
        _ax.set_xlabel("posts")
        _ax.set_ylabel("")
        _ax.set_xlim(right=_top["n"].max() * 1.15)
        _ax.grid(axis="y", visible=False)
    _fig
    return


@app.cell
def _(con, df, embeddings_source, mo, np, semantic, viz):
    # Topic clusters on Kenya-scoped explicit-toxic posts.
    _flag = df[df["explicit_toxic"]][["platform_post_id", "text", "p_hate"]]
    if _flag.empty:
        _out = mo.md("_No explicit-toxic Kenya-scoped posts to cluster._")
    else:
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
        _ax.set_title(
            "Largest narrative clusters in explicit-toxic speech "
            "(Kenya; darker = more hateful)"
        )
        _ax.set_xlabel("posts in cluster")
        _out = mo.vstack([
            mo.md("Cluster label = distinctive terms; shade = mean `p_hate`."),
            _fig,
        ])
    _out
    return


if __name__ == "__main__":
    app.run()
