import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from kma import viz
    from kma.authenticity import WEIGHTS, authenticity_score
    from kma.db import connect

    viz.use_theme()
    df = authenticity_score(connect())
    return WEIGHTS, df, mo, viz


@app.cell
def _(WEIGHTS, df, mo):
    _w = " · ".join(f"{k} {v}" for k, v in WEIGHTS.items())
    mo.md(
        f"""
        # Account authenticity triage

        **{len(df):,}** accounts scored. No ground-truth labels: this ranks how
        bot-like an account *looks*, it does not prove inauthenticity. It feeds
        the coordination scorecard (Phase 3) as the `bot_likeness` component.

        **`suspicion`** (0-1) is a weighted blend of transparent red flags -
        weights: {_w}. What each signal captures:

        - **age** - very new accounts are cheap to mass-create.
        - **follower/following ratio** - following thousands while followed by
          few is a classic amplifier pattern.
        - **rate** - superhuman posting cadence.
        - **dup** - high share of near-duplicate text (copypasta).
        - **bio / img / handle** - empty bio, default avatar, digit-suffix
          handle (`Brian4893555414`) - the throwaway-account fingerprint.

        **`anomaly_rank`** (0-1 percentile) is a second, *unsupervised* lens
        (isolation forest). It flags statistical outliers on all features at
        once - which includes genuine mega-influencers - so read it **alongside**
        `suspicion`, never alone. An account high on both is the strong signal.
        """
    )
    return


@app.cell
def _(df):
    _cols = [
        "handle", "account_age_days", "followers_count", "following_count",
        "tweet_rate", "duplicate_text_ratio", "n_posts", "suspicion", "anomaly_rank",
    ]
    df.sort_values("suspicion", ascending=False)[_cols].head(25)
    return


@app.cell
def _(df, viz):
    _fig, _ax = viz.new_fig(9, 3.6)
    _p90 = df["suspicion"].quantile(0.9)
    _ax.hist(df["suspicion"], bins=40, color=viz.BLUE, linewidth=0, rwidth=0.92)
    _ax.axvline(_p90, color=viz.INK_2, linewidth=1)
    _ax.text(_p90, _ax.get_ylim()[1] * 0.95, " p90", va="top", fontsize=9, color=viz.INK_2)
    _ax.set_title("Heuristic suspicion across all accounts")
    _ax.set_xlabel("suspicion")
    _ax.set_ylabel("accounts")
    _ax.grid(axis="x", visible=False)
    _fig
    return


@app.cell
def _(df, viz):
    _fig, _ax = viz.new_fig(9, 5)
    _sc = _ax.scatter(
        df["account_age_days"], df["followers_count"].clip(lower=1),
        c=df["suspicion"], cmap=viz.SEQ_CMAP, vmin=0, vmax=df["suspicion"].max(),
        s=26, edgecolors=viz.SURFACE, linewidths=0.7,
    )
    _ax.set_yscale("log")
    _cb = _fig.colorbar(_sc, ax=_ax, pad=0.01)
    _cb.set_label("suspicion", color=viz.INK_2, fontsize=9)
    _cb.outline.set_visible(False)
    _ax.set_title("Account age vs followers, shaded by suspicion")
    _ax.set_xlabel("account age (days)")
    _ax.set_ylabel("followers")
    _fig
    return


@app.cell
def _(df, viz):
    _fig, _ax = viz.new_fig(8, 5.5)
    _p90 = df["suspicion"].quantile(0.9)
    _hot = (df["suspicion"] > _p90) & (df["anomaly_rank"] > 0.9)
    for _mask, _color, _z in ((~_hot, viz.DEEMPH, 2), (_hot, viz.RED, 3)):
        _g = df[_mask]
        _ax.scatter(
            _g["suspicion"], _g["anomaly_rank"], s=24, color=_color, zorder=_z,
            edgecolors=viz.SURFACE, linewidths=0.7,
        )
    for _i, (_, _r) in enumerate(df[_hot].nlargest(3, "suspicion").iterrows()):
        _ax.annotate(
            _r["handle"], (_r["suspicion"], _r["anomaly_rank"]),
            xytext=(10, -14 - 13 * _i), textcoords="offset points",
            fontsize=8, color=viz.INK_2,
            arrowprops={"arrowstyle": "-", "color": viz.BASELINE, "linewidth": 0.8},
        )
    _ax.axhline(0.9, color=viz.GRID, linewidth=1)
    _ax.axvline(_p90, color=viz.GRID, linewidth=1)
    viz.legend_swatches(
        _ax,
        [("flagged by both lenses", viz.RED), ("everyone else", viz.DEEMPH)],
        loc="lower right",
    )
    _ax.set_title("Suspicion vs anomaly rank - two independent lenses")
    _ax.set_xlabel("heuristic suspicion")
    _ax.set_ylabel("isolation-forest anomaly rank")
    _ax.grid(visible=False)
    _fig
    return


@app.cell
def _(df, viz):
    _fig, _ax = viz.new_fig(9, 3.6)
    _created = df["created_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    _ax.hist(_created, bins=40, color=viz.BLUE, linewidth=0, rwidth=0.92)
    _ax.set_title("Account-creation dates - bursty windows are a CIB signal")
    _ax.set_xlabel("account created")
    _ax.set_ylabel("accounts")
    _ax.grid(axis="x", visible=False)
    _fig.autofmt_xdate(rotation=0, ha="center")
    _fig
    return


@app.cell
def _(df, mo):
    _known = [
        "WilliamsRuto", "RailaOdinga", "MarthaKarua", "NationAfrica",
        "StandardKenya", "rigathi", "skmusyoka", "UDAKenya", "TheODMparty",
    ]
    _sanity = df[df["handle"].isin(_known)][
        ["handle", "account_age_days", "followers_count", "following_count", "suspicion", "anomaly_rank"]
    ].sort_values("suspicion")
    mo.vstack([
        mo.md("### Sanity check: known-real accounts should score low on `suspicion`"),
        _sanity,
    ])
    return


if __name__ == "__main__":
    app.run()
