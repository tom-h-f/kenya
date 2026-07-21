import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    from kma import viz

    viz.use_theme_538()
    OUT = Path(__file__).resolve().parent.parent / (
        "investigations/2026-07-17-manipulation-sweep/out"
    )
    THEME = viz.altair_theme_538()

    def styled(chart):
        return chart.configure(**THEME)

    def load(name):
        return pd.read_csv(OUT / name)

    return alt, load, mo, pd, styled


@app.cell
def _(mo):
    mo.md("""
    # Manipulation sweep

    A single-window read of the 2026-07 manipulation sweep
    (`investigations/2026-07-17-manipulation-sweep`). Every panel is drawn
    from a committed artifact in that folder's `out/`, so the notebook
    renders without touching R2.

    **These are triage signals, not verdicts.** Capture is a sample, not a
    census; coordination and cohort structure are evidence of *similarity*,
    not proof of malice; a suspicion score is not a bot label. Read the
    source posts before repeating any of this.
    """)
    return


@app.cell
def _(load, mo):
    _conv = load("09_convergence.csv")
    _flagged = load("10_flagged.csv")
    _bursts = load("03_paste_bursts.csv")
    _tail = load("10_nli_tail.csv")

    def tile(label, value, sub):
        return mo.stat(value=value, label=label, caption=sub, bordered=True)

    mo.hstack(
        [
            tile("Scored posts", "109K", "full corpus, incitement NLI"),
            tile(
                "Accounts, >=2 lenses",
                f"{int((_conv['n_lenses'] >= 2).sum())}",
                "convergence set",
            ),
            tile("Paste bursts", f"{len(_bursts)}", ">=3 authors in 60 min"),
            tile(
                "Lexicon+NLI flags",
                f"{len(_flagged)}",
                "coded-term incitement",
            ),
            tile(
                "NLI-only tail",
                f"{len(_tail)}",
                "violence-adjacent, no coded term",
            ),
        ],
        justify="start",
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. The "84.1% election violence" cascade

    A precise-sounding statistic ("Kenya faces an 84.1% chance of
    election-related violence", attributed to the Kofi Annan Foundation)
    was posted by several accounts inside one hour, then amplified by a
    retweet burst. Multiple *originals* of one decimal-precision scare stat
    in an hour is a seeding pattern, not organic pickup. Whether the
    foundation published any such figure is a concrete fact-check.
    """)
    return


@app.cell
def _(alt, pd, styled):
    # Reconstructed from the origin trace in findings.md (F4). Times UTC+1.
    _cascade = pd.DataFrame(
        [
            ("04:00", "Mike_Kutola", "seed", 1),
            ("04:36", "moneyacademyKE", "seed", 1),
            ("04:37", "Marleek97", "retweet", 1),
            ("04:39", "lance_ned", "retweet", 1),
            ("04:39", "jtale01", "retweet", 1),
            ("04:40", "kenmutuma_KE", "retweet", 1),
            ("04:41", "DeRiuki", "retweet", 1),
            ("04:44", "gathiru_", "retweet", 1),
            ("04:45", "clifba", "retweet", 1),
            ("04:47", "beccyqi", "retweet", 1),
            ("04:52", "BravinYuri", "seed", 1),
            ("04:53", "IAbdallahke", "retweet", 1),
        ],
        columns=["time", "handle", "kind", "n"],
    )
    _cascade["cum"] = range(1, len(_cascade) + 1)
    _base = alt.Chart(_cascade).encode(
        x=alt.X("time:O", title="minute (UTC+1, 2026-07-09)"),
        y=alt.Y("cum:Q", title="cumulative posts of the claim"),
    )
    _line = _base.mark_line(color="#2a78d6", strokeWidth=2, point=False)
    _pts = _base.mark_point(size=140, filled=True, stroke="#f0f0f0", strokeWidth=2).encode(
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["seed", "retweet"], range=["#e34948", "#2a78d6"]),
            legend=alt.Legend(title="post kind", orient="top-left"),
        ),
        tooltip=["time", "handle", "kind"],
    )
    styled(
        (_line + _pts)
        .properties(
            width=620,
            height=300,
            title=alt.Title(
                "One stat, three 'originals', eleven retweets in 53 minutes",
                subtitle="Red = fresh post of the claim; blue = retweet. Two later 'originals' re-seed after the RT wave.",
            ),
        )
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Coordinated paste bursts

    Near-duplicate text posted by three or more distinct authors inside a
    60-minute window (from `coordination.content_clusters`, then ordered by
    time to find who seeds). Bar length is the number of distinct authors;
    colour is how tight the burst was.
    """)
    return


@app.cell
def _(alt, load, styled):
    _b = load("03_paste_bursts.csv").copy()
    _b["label"] = _b["seed_handle"] + " -- " + _b["text"].str.slice(0, 46) + "..."
    _b = _b.sort_values("n_authors", ascending=False).head(12)
    _chart = (
        alt.Chart(_b)
        .mark_bar(cornerRadiusEnd=4, height=16)
        .encode(
            y=alt.Y("label:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=340)),
            x=alt.X("n_authors:Q", title="distinct authors in burst"),
            color=alt.Color(
                "span_min:Q",
                scale=alt.Scale(scheme="blues", reverse=True),
                legend=alt.Legend(title="burst span (min)"),
            ),
            tooltip=["seed_handle", "n_authors", "span_min",
                     "median_echo_lag_min", "text"],
        )
        .properties(
            width=560,
            height=alt.Step(22),
            title=alt.Title(
                "Paste bursts by author count",
                subtitle="Tighter (darker) bursts of many authors are the strongest coordination signal.",
            ),
        )
    )
    styled(_chart)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Habitual first repliers

    For every reply to a captured parent post, its rank in the thread's
    arrival order becomes a percentile. A genuine audience replies at
    random positions, so the mean percentile is ~0.5. Accounts far below
    that (negative z vs the uniform null) are *systematically early* across
    many different targets - a watch-and-pounce pattern. Every account
    shown also sits in an SVN-validated coordination cluster.
    """)
    return


@app.cell
def _(alt, load, styled):
    _f = load("04_fast_repliers.csv").sort_values("z_vs_uniform").head(14)
    _chart = (
        alt.Chart(_f)
        .mark_bar(cornerRadiusEnd=4, height=15, color="#e34948")
        .encode(
            y=alt.Y("author_handle:N", sort="x", title=None),
            x=alt.X("z_vs_uniform:Q",
                    title="z vs uniform thread-rank null (more negative = earlier)"),
            tooltip=["author_handle", "n", "mean_rank_pct", "median_lag_min",
                     "targets", "z_vs_uniform"],
        )
        .properties(
            width=560,
            height=alt.Step(21),
            title=alt.Title(
                "Systematically-early repliers",
                subtitle="z from a uniform-position null; all are coordination-cluster members.",
            ),
        )
    )
    styled(_chart)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Convergence matrix

    The deliverable ranking: accounts flagged by two or more *independent*
    lenses. A filled cell is a hit on that lens. Lenses share the capture,
    not their features, so multiple hits raise triage priority - they do
    not prove inauthenticity.
    """)
    return


@app.cell
def _(alt, load, styled):
    _c = load("09_convergence.csv")
    _lenses = ["coord_cluster", "fast_replier", "seeder", "awakened",
               "regular", "fringe_seed", "suspicion"]
    _top = _c[_c["n_lenses"] >= 2].sort_values(
        ["n_lenses", "followers_count"], ascending=[False, True]
    ).head(28)
    _long = _top.melt(
        id_vars=["handle", "n_lenses"], value_vars=_lenses,
        var_name="lens", value_name="hit",
    )
    _chart = (
        alt.Chart(_long)
        .mark_rect(stroke="#f0f0f0", strokeWidth=2)
        .encode(
            y=alt.Y("handle:N", sort=alt.EncodingSortField("n_lenses", order="descending"),
                    title=None),
            x=alt.X("lens:N", sort=_lenses, title=None,
                    axis=alt.Axis(labelAngle=-40, orient="top")),
            color=alt.Color(
                "hit:N",
                scale=alt.Scale(domain=[True, False], range=["#2a78d6", "#e6e6e3"]),
                legend=None,
            ),
            tooltip=["handle", "lens", "hit", "n_lenses"],
        )
        .properties(
            width=alt.Step(58),
            height=alt.Step(19),
            title=alt.Title(
                "Accounts flagged by >=2 lenses",
                subtitle="Filled = hit. senator047 (top) is the only 3-lens account.",
            ),
        )
    )
    styled(_chart)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Incitement over time

    Daily count of lexicon-confirmed coded-term posts (madoadoa, nyoka,
    kwekwe, ...) that also clear the NLI bar, split by category. Volume is
    low - this is a targeted triage lane, not a mass phenomenon. The
    larger, structurally-more-dangerous signal is the NLI-only tail below.
    """)
    return


@app.cell
def _(alt, load, styled):
    _t = load("10_trend.csv")
    _order = ["dehumanisation", "expulsion", "othering", "veiled_threat"]
    _chart = (
        alt.Chart(_t)
        .mark_bar(cornerRadiusEnd=3, width=16)
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("n:Q", title="flagged posts", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color(
                "lexicon_categories:N",
                sort=_order,
                scale=alt.Scale(domain=_order,
                                range=["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]),
                legend=alt.Legend(title="category", orient="top"),
            ),
            tooltip=["day", "lexicon_categories", "n"],
        )
        .properties(
            width=600,
            height=260,
            title=alt.Title(
                "Coded-term incitement, daily",
                subtitle="Lexicon hit AND max NLI >= 0.85. Direct-labelled categories; see table for text.",
            ),
        )
    )
    styled(_chart)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Incitement scores: lexicon hits vs everything else

    Each flagged post placed by its two strongest NLI scores. Lexicon-hit
    posts (red) cluster high on dehumanisation; the NLI-only tail (grey)
    stretches along the violence-call axis - accusation-driven
    fear-priming that carries no coded term, which is why the lexicon alone
    cannot see it.
    """)
    return


@app.cell
def _(alt, load, pd, styled):
    _flag = load("10_flagged.csv").assign(group="lexicon + NLI")
    _tail = load("10_nli_tail.csv").assign(group="NLI-only tail")
    _cols = ["author_handle", "dehumanisation_score", "violence_call_score",
             "othering_score", "text", "group"]
    _pts = pd.concat([_flag[_cols], _tail[_cols].head(400)], ignore_index=True)
    _chart = (
        alt.Chart(_pts)
        .mark_point(size=70, filled=True, opacity=0.7, stroke="#f0f0f0",
                    strokeWidth=0.6)
        .encode(
            x=alt.X("violence_call_score:Q", title="violence-call score",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("dehumanisation_score:Q", title="dehumanisation score",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=["lexicon + NLI", "NLI-only tail"],
                                range=["#e34948", "#8f8d86"]),
                legend=alt.Legend(title=None, orient="top-left"),
            ),
            tooltip=["author_handle", "dehumanisation_score",
                     "violence_call_score", "othering_score", "text"],
        )
        .properties(
            width=520,
            height=420,
            title=alt.Title(
                "Where the incitement signal lives",
                subtitle="Red coded-term posts sit high-left; the grey tail runs along the violence-call axis.",
            ),
        )
    )
    styled(_chart)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Flagged posts (read these)

    The 14 lexicon+NLI posts, and the top of the NLI-only tail. This is the
    human-review queue; the charts above are only its shape.
    """)
    return


@app.cell
def _(load, mo):
    _flag = load("10_flagged.csv")[
        ["author_handle", "created_at", "lexicon_hits", "dehumanisation_score",
         "violence_call_score", "othering_score", "text"]
    ]
    mo.ui.table(_flag, selection=None, page_size=14)
    return


@app.cell
def _(load, mo):
    _tail = load("10_nli_tail.csv")[
        ["author_handle", "created_at", "dehumanisation_score",
         "violence_call_score", "othering_score", "text"]
    ].head(40)
    mo.ui.table(_tail, selection=None, page_size=15)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    Source: `investigations/2026-07-17-manipulation-sweep/findings.md` and
    the `out/` artifacts. Re-generate any panel by re-running the matching
    `NN_*.py --full`. Caveats in the header apply to every panel.
    """)
    return


if __name__ == "__main__":
    app.run()
