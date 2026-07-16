import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from kma import coordination as co
    from kma import deltas
    from kma import framing as fr
    from kma import stories as st
    from kma import viz
    from kma.db import (
        connect,
        latest_coordination_clusters,
        latest_coordination_edges,
        latest_labels,
        latest_stories,
        authors_source,
    )
    from kma.semantic import assign_topics, topic_summary

    viz.use_theme()
    con = connect()
    con.execute("SET enable_progress_bar=false")
    return (
        assign_topics,
        authors_source,
        co,
        con,
        deltas,
        fr,
        latest_coordination_clusters,
        latest_coordination_edges,
        latest_labels,
        latest_stories,
        mo,
        pd,
        st,
        topic_summary,
        viz,
    )


@app.cell
def _(mo, st):
    mo.md(f"""
    # Desk brief — misinfo / disinfo investigation

    Investigator-facing composition of **claims**, **corroboration**,
    **amplifiers**, **framing**, **claim-scoped coordination**, and
    **region / community** aggregates.

    This is triage for humans, **not** an auto-verdict.

    > **{st.STORY_CAVEAT}**

    > **{st.SAMPLING_CAVEAT}**

    Bot-likeness is a suspicion score, not a bot label. Coordination is
    probabilistic co-action, not proof of malice.
    """)
    return


@app.cell
def _(mo, st):
    days = mo.ui.slider(1, 30, value=st.DEFAULT_DAYS, label="Window (days)")
    lane = mo.ui.dropdown(
        {"All tiers": "all", "Main triage": "main", "Thin evidence": "thin_evidence"},
        value="All tiers",
        label="Lane",
    )
    mo.hstack([days, lane], justify="start", gap=2)
    return days, lane


@app.cell
def _(con, days, st):
    stories = st.candidate_stories(
        con,
        days=int(days.value),
        tau=st.DEFAULT_TAU,
        min_size=st.DEFAULT_MIN_SIZE,
        include_thin=True,
    )
    if stories.empty:
        cards = stories
        corrob = stories
    else:
        corrob = st.corroboration(con, stories, days=int(days.value))
        cards = st.assign_tiers(stories, st.story_scorecard(con, stories, corrob))
    return cards, corrob, stories


@app.cell
def _(cards, lane, mo, st):
    if cards is None or not len(cards):
        filtered = cards
        _out = mo.md(
            "_No candidate stories on this window. Widen days, confirm embeddings, "
            "or lower thresholds in the stories method lab._"
        )
    else:
        if lane.value == "all":
            filtered = cards
        else:
            filtered = cards[cards["tier"] == lane.value]
        _n_main = int((cards["tier"] == st.TIER_MAIN).sum())
        _n_thin = int((cards["tier"] == st.TIER_THIN).sum())
        _n_hi = int(cards["high_suspicion"].sum()) if "high_suspicion" in cards.columns else 0
        _cols = [
            c for c in (
                "story_id", "stable_story_id", "tier", "high_suspicion", "keywords",
                "size", "n_posts", "corrob_sim", "corroboration_gap", "nearest_handle",
                "amplifier_botness", "coordination_overlap", "story_suspicion_index",
            ) if c in filtered.columns
        ]
        _tbl = filtered[_cols].copy()
        if "keywords" in _tbl.columns:
            _tbl["keywords"] = _tbl["keywords"].apply(
                lambda ts: " ".join(ts[:5]) if isinstance(ts, list) else ts
            )
        _out = mo.vstack([
            mo.md("## 1. Circulating claims"),
            mo.md(
                f"**{len(cards)}** scored stories "
                f"(main `{_n_main}`, thin `{_n_thin}`; "
                f"high_suspicion flag `{_n_hi}`). "
                f"Showing **{len(filtered)}** after lane filter."
            ),
            mo.md(
                "Thin-evidence rows are small high-gap claims — evidence to inspect, "
                "**not** auto-elevated suspicion."
            ),
            _tbl.round(3) if len(_tbl) else mo.md("_Lane filter empty._"),
        ])
    _out
    return (filtered,)


@app.cell
def _(filtered, mo):
    if filtered is not None and len(filtered):
        _opts = {
            f"[{r.tier}] [{r.story_id}] "
            f"{' '.join(r.keywords[:5]) if isinstance(r.keywords, list) else ''} "
            f"(gap {r.corroboration_gap:.2f}, ix {r.story_suspicion_index:.2f})": int(r.story_id)
            for r in filtered.itertuples()
        }
        story_pick = mo.ui.dropdown(_opts, value=list(_opts)[0], label="Focus story")
    else:
        story_pick = mo.ui.dropdown({"(none)": -1}, value="(none)", label="Focus story")
    story_pick
    return (story_pick,)


@app.cell
def _(cards, mo, st, story_pick):
    mo.md("## 2. Corroboration desk")
    _sid = story_pick.value
    if cards is not None and len(cards) and _sid is not None and _sid >= 0:
        _c = cards[cards["story_id"] == _sid].iloc[0]
        _near = _c.get("nearest_text")
        _out = mo.vstack([
            mo.callout(mo.md(f"**{st.STORY_CAVEAT}**"), kind="warn"),
            mo.md(f"**Claim exemplar** (`tier={_c['tier']}`):\n\n> {_c['representative_text']}"),
            mo.md(
                f"**Nearest trusted post** — _{_c.get('nearest_handle')}_ "
                f"(sim **{_c['corrob_sim']:.2f}**, gap **{_c['corroboration_gap']:.2f}**):\n\n"
                f"> {_near}"
                if _near else
                "_No trusted-outlet post cleared the claim/entity gate — maximal gap. "
                "Still not proof of falsity (outlet lag / sample)._"
            ),
        ])
    else:
        _out = mo.md("_Select a story._")
    _out
    return


@app.cell
def _(con, mo, st, stories, story_pick):
    mo.md("## 3. Amplifiers and origin")
    _sid = story_pick.value
    if _sid is not None and _sid >= 0 and stories is not None and len(stories):
        _story = stories[stories["story_id"] == _sid]
        _origin = st.origin(con, _story)
        _spread = st.spread(con, _story)
        focus_story = _story
        focus_spread = _spread
        _out = mo.vstack([
            mo.callout(mo.md(f"**{st.SAMPLING_CAVEAT}**"), kind="neutral"),
            mo.md("### Origin (earliest collected members)"),
            _origin,
            mo.md("### Spread (top amplifiers)"),
            _spread["amplifiers"].head(25) if len(_spread["amplifiers"])
            else mo.md("_No engagement/reply census reached these posts yet._"),
        ])
    else:
        focus_story = stories.iloc[0:0] if stories is not None else None
        focus_spread = {"amplifiers": None}
        _out = mo.md("_Select a story._")
    _out
    return focus_spread, focus_story


@app.cell
def _(mo):
    mo.md("""
    ## 4. Framing shifts

    Claim-anchored topic neighborhood + sentiment timeline. Topics are corpus-wide;
    the claim is mapped onto them (not the other way around). Stance stays live /
    target-parameterized in the narratives notebook.
    """)
    run_topics = mo.ui.run_button(label="Compute topics + framing (slow)")
    run_topics
    return (run_topics,)


@app.cell
def _(
    assign_topics,
    con,
    fr,
    latest_labels,
    mo,
    run_topics,
    stories,
    story_pick,
    topic_summary,
):
    _sid = story_pick.value
    if not run_topics.value:
        framing_bundle = None
        _out = mo.md("_Press the button to run UMAP/HDBSCAN topics + framing for this window._")
    elif stories is None or not len(stories) or _sid is None or _sid < 0:
        framing_bundle = None
        _out = mo.md("_Need stories + a focus story._")
    else:
        _topics = assign_topics(con, min_cluster_size=60)
        _summary = topic_summary(_topics)
        try:
            _labels = latest_labels(con).df()
        except Exception:
            _labels = None
        _story = stories[stories["story_id"] == _sid]
        framing_bundle = fr.story_framing(
            _story, topics_df=_topics, topic_summary=_summary, labels=_labels
        )
        _kw = framing_bundle["keywords"].get(int(_sid), [])
        _out = mo.vstack([
            mo.md(f"**Local keywords:** {', '.join(_kw) if _kw else '_(none)_'}"),
            mo.md("### Topic overlap"),
            framing_bundle["topics"] if len(framing_bundle["topics"])
            else mo.md("_No topic overlap (all noise or empty)._"),
            mo.md("### Sentiment timeline (member posts with labels)"),
            framing_bundle["sentiment_timeline"] if len(framing_bundle["sentiment_timeline"])
            else mo.md("_No labeled members in this claim window._"),
        ])
    _out
    return (framing_bundle,)


@app.cell
def _(
    co,
    con,
    focus_spread,
    focus_story,
    latest_coordination_clusters,
    latest_coordination_edges,
    mo,
    story_pick,
):
    mo.md("## 5. Claim-scoped coordination")
    _sid = story_pick.value
    if focus_story is None or not len(focus_story) or _sid is None or _sid < 0:
        coord_view = None
        _out = mo.md("_Select a story for claim-scoped coordination._")
    else:
        _amps = focus_spread.get("amplifiers") if focus_spread else None
        _accounts = co.story_account_set(focus_story, _amps)
        try:
            _edges = latest_coordination_edges(con).df()
        except Exception:
            _edges = None
        try:
            _clusters = latest_coordination_clusters(con).df()
        except Exception:
            _clusters = None
        coord_view = co.claim_coordination(_accounts, _edges, _clusters)
        _sum = coord_view["summary"]
        _out = mo.vstack([
            mo.md(
                f"Accounts in slice: **{_sum['n_accounts']}**. "
                f"Edges: **{_sum['n_edges']}**. "
                f"Clusters touched: **{_sum['n_clusters']}** "
                f"({_sum['cluster_ids']}). Channels: `{_sum['channels']}`."
            ),
            mo.callout(mo.md(f"**{_sum['note']}**"), kind="neutral"),
            mo.md("### Edges (claim-scoped)"),
            coord_view["edges"].head(50) if len(coord_view["edges"])
            else mo.md("_No validated edges among these accounts (or none persisted yet)._"),
            mo.md("### Cluster membership (claim-scoped)"),
            coord_view["clusters"].head(50) if len(coord_view["clusters"])
            else mo.md("_No cluster membership overlap._"),
        ])
    _out
    return (coord_view,)


@app.cell
def _(authors_source, con, deltas, focus_story, mo, pd, story_pick):
    mo.md("## 6. Community and region lens")
    _sid = story_pick.value
    if focus_story is None or not len(focus_story) or _sid is None or _sid < 0:
        _out = mo.md("_Select a story._")
    else:
        _ids = focus_story["author_id"].dropna().unique().tolist()
        if not _ids:
            _out = mo.md("_No authors on this story._")
        else:
            _id_list = ", ".join(f"'{i}'" for i in _ids)
            try:
                _auth = con.sql(
                    f"""
                    SELECT platform_user_id AS author_id, location
                    FROM (
                        SELECT * FROM {authors_source('x')}
                        QUALIFY row_number() OVER (
                            PARTITION BY platform, platform_user_id
                            ORDER BY collected_at DESC
                        ) = 1
                    )
                    WHERE platform_user_id IN ({_id_list})
                    """
                ).df()
            except Exception as exc:
                _auth = pd.DataFrame(columns=["author_id", "location"])
                _err = str(exc)
            else:
                _err = None
            _region = deltas.slice_claim(_auth, "region")
            _community = deltas.slice_claim(_auth, "community")
            _disc = (
                _community["disclaimer"].iloc[0]
                if len(_community) and "disclaimer" in _community.columns
                else deltas.TRIBE_DISCLAIMER
            )
            _insuff = bool(_community["insufficient_location_signal"].iloc[0]) if len(_community) else True
            _parts = [
                mo.callout(mo.md(f"**{_disc}**"), kind="warn"),
                mo.md(
                    "_Insufficient location signal — treat breakdowns as exploratory only._"
                    if _insuff else
                    "_Coverage above threshold — still aggregate-only; never attach to a person._"
                ),
                mo.md("### Region (aggregate)"),
                _region if len(_region) else mo.md("_No mappable region signal._"),
                mo.md("### Community proxy (aggregate)"),
                _community if len(_community) else mo.md("_No mappable community signal._"),
            ]
            if _err:
                _parts.insert(0, mo.callout(mo.md(f"Author lookup issue: `{_err}`"), kind="danger"))
            _out = mo.vstack(_parts)
    _out
    return


@app.cell
def _(cards, con, latest_stories, mo, pd):
    mo.md("## 7. What’s new vs last persisted run")
    if cards is None or not len(cards) or "stable_story_id" not in cards.columns:
        _out = mo.md("_No current scorecard with stable_story_id._")
    else:
        try:
            _prev = latest_stories(con).df()
        except Exception:
            _prev = pd.DataFrame()
        if _prev.empty or "stable_story_id" not in _prev.columns:
            _out = mo.md(
                "_No prior `stories/` run with `stable_story_id` yet. "
                "Persist from the stories notebook after upgrading, then re-open._"
            )
        else:
            _cur = set(cards["stable_story_id"].dropna().astype(str))
            _old = set(_prev["stable_story_id"].dropna().astype(str))
            _new = _cur - _old
            _gone = _old - _cur
            _tbl = cards[cards["stable_story_id"].astype(str).isin(_new)][
                [c for c in ("stable_story_id", "tier", "keywords", "corroboration_gap",
                             "story_suspicion_index") if c in cards.columns]
            ]
            _out = mo.vstack([
                mo.md(
                    f"**New since last persist:** {len(_new)}. "
                    f"**No longer in window/scorecard:** {len(_gone)}."
                ),
                _tbl if len(_tbl) else mo.md("_No new stable ids vs last run._"),
            ])
    _out
    return


@app.cell
def _(cards, focus_story, mo, st, story_pick):
    mo.md("## 8. Per-story deep dive / follow-ups")
    _sid = story_pick.value
    if cards is None or not len(cards) or _sid is None or _sid < 0:
        _out = mo.md("_Select a story._")
    else:
        _c = cards[cards["story_id"] == _sid].iloc[0]
        _members = (
            focus_story[["author_handle", "text", "created_at"]].head(20)
            if focus_story is not None and len(focus_story) else None
        )
        _qs = [
            "Does the nearest trusted post address the *same claim*, or only the topic?",
            "Are amplifiers overlapping a known coordination cluster, or organic fans?",
            "Is the thin-lane signal still alone (2 accounts) or starting to recruit?",
            "Does region/community aggregate skew look like geography of the event, or of the amp network?",
            "What would a journalist ask PesaCheck / AfricaCheck / newsroom next?",
        ]
        _out = mo.vstack([
            mo.md(f"### Story `{_c.get('stable_story_id', _sid)}`"),
            mo.md(
                f"- tier: `{_c['tier']}` · high_suspicion: `{_c.get('high_suspicion')}`\n"
                f"- size `{int(_c['size'])}` · posts `{int(_c['n_posts'])}`\n"
                f"- gap `{_c['corroboration_gap']:.2f}` · suspicion index "
                f"`{_c['story_suspicion_index']:.2f}`\n"
                f"- botness `{_c.get('amplifier_botness', float('nan')):.2f}` · "
                f"coord overlap `{_c.get('coordination_overlap', 0):.2f}`"
            ),
            mo.md("### Member sample"),
            _members if _members is not None else mo.md("_No members._"),
            mo.md("### Suggested human follow-ups"),
            mo.md("\n".join(f"{i}. {q}" for i, q in enumerate(_qs, 1))),
            mo.accordion({"Metric glossary & caveats": mo.md(st.glossary_md())}),
        ])
    _out
    return


if __name__ == "__main__":
    app.run()
