import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns

    from kma.authenticity import WEIGHTS, authenticity_score
    from kma.db import connect

    sns.set_theme(style="whitegrid", palette="rocket")
    df = authenticity_score(connect())
    return WEIGHTS, df, mo, plt, sns


@app.cell
def _(WEIGHTS, df, mo):
    _w = " · ".join(f"{k} {v}" for k, v in WEIGHTS.items())
    mo.md(
        f"""
        # Account authenticity triage

        **{len(df):,}** accounts scored. No ground-truth labels: this ranks how
        bot-like an account looks, it does not prove inauthenticity.

        **Heuristic weights:** {_w}. `suspicion` is in [0,1], age +
        follower/following ratio dominant.

        **`anomaly_rank`** is an unsupervised isolation-forest second lens. It
        flags statistical outliers - which includes genuine mega-influencers -
        so read it alongside `suspicion`, never alone.
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
def _(df, plt, sns):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df["suspicion"], bins=40, ax=_ax, color="#b5179e")
    _ax.set(title="Distribution of heuristic suspicion", xlabel="suspicion", ylabel="accounts")
    _fig
    return


@app.cell
def _(df, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="account_age_days", y="followers_count", hue="suspicion",
        palette="rocket", size="suspicion", sizes=(10, 120), ax=_ax, legend=False,
    )
    _ax.set(
        yscale="log", title="Account age vs followers (hue = suspicion)",
        xlabel="account age (days)", ylabel="followers",
    )
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
