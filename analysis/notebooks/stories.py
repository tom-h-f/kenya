import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from kma import stories as st
    from kma import viz
    from kma.db import connect

    viz.use_theme()
    con = connect()
    con.execute("SET enable_progress_bar=false")
    return con, mo, st, viz


@app.cell
def _(mo, st):
    mo.md(f"""
    # Story discovery & trusted-media triage

    Surface the **discrete claims** circulating right now, judge whether they
    look **amplified / uncorroborated**, and trace **origin + spread** - then hand
    a flagged story back to the collector for targeted collection.

    **How the pipeline works, end to end:**

    1. **Candidate stories** - cluster recent posts into claim-level groups
       (connected components of the cosine >= tau graph, at a story-level tau).
       One story = one claim, paraphrases included.
    2. **Corroboration** - compare each story's centroid to the trusted Kenyan
       outlets already collected + embedded in the same space
       ({", ".join(st.TRUSTED_SOURCES)}). A viral claim with **no close
       trusted-media match** is a corroboration gap.
    3. **Scorecard** - stack the corroboration gap with amplifier bot-likeness,
       coordination overlap, burstiness and source concentration into a
       transparent, percentile-ranked `story_suspicion_index`.
    4. **Origin / spread** - first-seen authors + the retweeter/reply diffusion.
    5. **Flag** - persist the flagged stories; the collector promotes their
       keywords as targeted search terms.

    > **{st.STORY_CAVEAT}**

    Triage for a human, **never** an auto-label of "false". Always read the
    nearest trusted post before judging a story.
    """)
    return


@app.cell
def _(mo, st):
    mo.accordion({"Metric glossary & caveats (what every column means)": mo.md(st.glossary_md())})
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Candidate stories, ranked by suspicion

    Each row is a claim-level cluster over the recent window. Read
    `story_suspicion_index` alongside `corrob_sim` and the nearest trusted post -
    a high index with a wide corroboration gap is the triage target, but the
    nearest trusted post is what tells you whether the gap is real or just an
    outlet lagging.
    """)
    return


@app.cell
def _(con, st):
    stories = st.candidate_stories(con, days=st.DEFAULT_DAYS, tau=st.DEFAULT_TAU,
                                   min_size=st.DEFAULT_MIN_SIZE)
    corrob = st.corroboration(con, stories, days=st.DEFAULT_DAYS)
    cards = st.story_scorecard(con, stories, corrob)
    return cards, stories


@app.cell
def _(cards, mo, st):
    if cards is not None and len(cards):
        _cols = [
            "story_id", "keywords", "hashtags", "size", "n_posts",
            "corrob_sim", "corroboration_gap", "nearest_handle",
            "story_suspicion_index",
        ]
        _tbl = cards[[c for c in _cols if c in cards.columns]].copy()
        _tbl["keywords"] = _tbl["keywords"].apply(lambda ts: " ".join(ts[:4]))
        _tbl["hashtags"] = _tbl["hashtags"].apply(lambda ts: " ".join(ts[:3]))
        _out = mo.vstack([
            mo.md(f"**{len(cards)} candidate stories** on the last "
                  f"{st.DEFAULT_DAYS} days (tau={st.DEFAULT_TAU})."),
            _tbl.round(3),
        ])
    else:
        _out = mo.md("_No candidate stories on the current window - widen `days` "
                     "or lower `min_size`, and confirm recent posts are embedded._")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Drill into one story

    Pick a story to see its first-seen authors (origin), how it diffused
    (amplifiers + volume timeline), and the trusted post nearest to it.
    """)
    return


@app.cell
def _(cards, mo):
    if cards is not None and len(cards):
        _opts = {
            f"[{r.story_id}] {' '.join(r.keywords[:5]) or r.representative_text[:60]}"
            f"  (index {r.story_suspicion_index:.2f})": int(r.story_id)
            for r in cards.itertuples()
        }
        story_pick = mo.ui.dropdown(_opts, value=list(_opts)[0], label="Story")
    else:
        story_pick = mo.ui.dropdown({"(none)": -1}, value="(none)", label="Story")
    story_pick
    return (story_pick,)


@app.cell
def _(cards, con, mo, st, stories, story_pick):
    _sid = story_pick.value
    if _sid is not None and _sid >= 0:
        _story = stories[stories["story_id"] == _sid]
        _card = cards[cards["story_id"] == _sid].iloc[0]
        _origin = st.origin(con, _story)
        _spread = st.spread(con, _story)
        _out = mo.vstack([
            mo.hstack([
                mo.stat(value=f"{int(_card['size'])}", label="Authors", bordered=True),
                mo.stat(value=f"{int(_card['n_posts'])}", label="Posts", bordered=True),
                mo.stat(value=f"{_card['corrob_sim']:.2f}", label="Corroboration sim", bordered=True),
                mo.stat(value=f"{_card['story_suspicion_index']:.2f}", label="Suspicion index", bordered=True),
            ], widths="equal"),
            mo.md("### Origin - earliest collected posts "
                  "(_earliest collected != patient-zero; capture is a sample_)"),
            _origin,
            mo.md("### Spread - top amplifiers (retweeters + repliers)"),
            _spread["amplifiers"].head(20) if len(_spread["amplifiers"])
            else mo.md("_No engagement/reply census reached these posts yet._"),
        ])
    else:
        _out = mo.md("_Select a story above._")
    _out
    return


@app.cell
def _(con, mo, st, stories, story_pick, viz):
    _sid = story_pick.value
    if _sid is not None and _sid >= 0:
        _tl = st.spread(con, stories[stories["story_id"] == _sid])["timeline"]
        if len(_tl) >= 2:
            _fig, _ax = viz.new_fig(9, 3.0)
            _ax.plot(_tl["hour"], _tl["n_posts"], color=viz.BLUE, marker="o", markersize=3)
            _ax.set_title("Story volume over time")
            _ax.set_xlabel("hour")
            _ax.set_ylabel("member posts")
            _fig.autofmt_xdate()
            _out = _fig
        else:
            _out = mo.md("_Too few time points to plot a timeline._")
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Corroboration panel

    The story vs the **nearest trusted-outlet post**. Judge the gap here: a low
    similarity with a clearly-unrelated trusted post is a genuine corroboration
    gap; a low similarity where the outlet simply has not covered it yet is not
    evidence of anything.
    """)
    return


@app.cell
def _(cards, mo, st, story_pick):
    _sid = story_pick.value
    if cards is not None and len(cards) and _sid is not None and _sid >= 0:
        _c = cards[cards["story_id"] == _sid].iloc[0]
        _near = _c.get("nearest_text")
        _out = mo.vstack([
            mo.callout(mo.md(f"**{st.STORY_CAVEAT}**"), kind="warn"),
            mo.md(f"**Story ({_c['size']} authors):** {_c['representative_text']}"),
            mo.md(
                f"**Nearest trusted post** — _{_c['nearest_handle']}_ "
                f"(sim **{_c['corrob_sim']:.2f}**):\n\n> \n{_near}\n"
                if _near else "_No trusted-outlet post in range - a maximal gap._"
            ),
        ])
    else:
        _out = mo.md("_Select a story above._")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Flag for targeted collection

    Persist the scored stories to R2 (`stories/` prefix). On its next promotion
    pass the collector reads the latest run and promotes flagged stories'
    keywords/hashtags as targeted search terms, so it chases the story.
    `min_index` drops low-suspicion stories from the write.
    """)
    return


@app.cell
def _(mo):
    flag_min = mo.ui.slider(0.0, 1.0, value=0.6, step=0.05,
                            label="min suspicion index to flag")
    flag_btn = mo.ui.run_button(label="Flag stories -> persist to R2")
    mo.hstack([flag_min, flag_btn], justify="start", gap=2)
    return flag_btn, flag_min


@app.cell
def _(cards, con, flag_btn, flag_min, mo, st):
    if flag_btn.value and cards is not None and len(cards):
        _key = st.persist_stories(con, cards, min_index=flag_min.value)
        _out = (
            mo.callout(mo.md(f"Wrote flagged stories to `{_key}`."), kind="success")
            if _key else mo.callout(mo.md("No story cleared the cutoff - nothing written."), kind="neutral")
        )
    else:
        _out = mo.md("_Set a cutoff and press the button to persist._")
    _out
    return


if __name__ == "__main__":
    app.run()
