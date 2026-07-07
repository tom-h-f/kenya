import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns

    from kma import connect_quack

    sns.set_theme(style="whitegrid", palette="rocket")
    con = connect_quack()

    def remote(sql: str):
        """Run SQL on the tf1 server. Views: posts, latest_posts, metrics, authors, latest_authors."""
        return con.sql(f"FROM kenya.query('{sql.replace(chr(39), chr(39) * 2)}')")

    return mo, plt, remote, sns


@app.cell
def _(mo, remote):
    _counts = remote(
        """
        SELECT 'posts' v, count(*) n FROM posts
        UNION ALL SELECT 'latest_posts', count(*) FROM latest_posts
        UNION ALL SELECT 'authors', count(*) FROM latest_authors
        UNION ALL SELECT 'metrics', count(*) FROM metrics
        """
    ).df()
    _t = {r.v: int(r.n) for r in _counts.itertuples()}
    mo.md(
        f"""
        # Kenya 2027 monitor
        Queried live through the **tf1 quack server**.

        **{_t.get("latest_posts", 0):,}** unique posts · **{_t.get("posts", 0):,}** snapshots ·
        **{_t.get("authors", 0):,}** authors · **{_t.get("metrics", 0):,}** metric snapshots
        """
    )
    return


@app.cell
def _(plt, remote, sns):
    _vol = remote(
        """
        SELECT date_trunc('hour', created_at) AS hour, count(*) AS posts
        FROM latest_posts WHERE platform = 'x' AND created_at > now() - INTERVAL 14 DAY
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    _fig, _ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=_vol, x="hour", y="posts", ax=_ax, color="#b5179e")
    _ax.fill_between(_vol["hour"], _vol["posts"], alpha=0.15, color="#b5179e")
    _ax.set(title="Post volume per hour (by tweet creation time)", xlabel="", ylabel="posts")
    _fig.autofmt_xdate()
    _fig
    return


@app.cell
def _(plt, remote, sns):
    _eng = remote("SELECT like_count FROM latest_posts WHERE platform = 'x' AND like_count > 0").df()
    _fig, _ax = plt.subplots(figsize=(9, 4))
    sns.histplot(data=_eng, x="like_count", log_scale=True, bins=30, ax=_ax, color="#7209b7")
    _ax.set(title="Distribution of likes per tweet (log scale)", xlabel="likes")
    _fig
    return


@app.cell
def _(plt, remote, sns):
    _top = remote(
        """
        SELECT author_handle, count(*) AS posts, sum(like_count) AS likes
        FROM latest_posts WHERE platform = 'x'
        GROUP BY 1 ORDER BY likes DESC LIMIT 15
        """
    ).df()
    _fig, _ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=_top, y="author_handle", x="likes", hue="likes", palette="rocket", legend=False, ax=_ax)
    _ax.set(title="Top authors by total likes", ylabel="", xlabel="likes")
    _fig
    return


@app.cell
def _(plt, remote, sns):
    _traj = remote(
        """
        WITH top AS (
            SELECT platform_post_id FROM latest_posts WHERE platform = 'x'
            ORDER BY like_count DESC LIMIT 6
        )
        SELECT p.platform_post_id AS id, p.collected_at, p.like_count
        FROM posts p JOIN top USING (platform_post_id)
        ORDER BY p.collected_at
        """
    ).df()
    _fig, _ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=_traj, x="collected_at", y="like_count", hue="id", marker="o", ax=_ax, legend=False)
    _ax.set(title="Engagement trajectory of the top 6 tweets (likes as resampled)", xlabel="", ylabel="likes")
    _fig.autofmt_xdate()
    _fig
    return


@app.cell
def _(plt, remote, sns):
    _fl = remote(
        """
        SELECT a.handle, a.followers_count AS followers, a.verified,
               sum(p.like_count) AS likes, count(*) AS posts
        FROM latest_posts p JOIN latest_authors a ON p.author_id = a.platform_user_id
        WHERE a.platform = 'x' AND a.followers_count > 0
        GROUP BY 1, 2, 3
        """
    ).df()
    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=_fl, x="followers", y="likes", size="posts", hue="verified", sizes=(30, 300), ax=_ax
    )
    _ax.set(xscale="log", yscale="log", title="Author reach vs engagement", xlabel="followers", ylabel="likes")
    _fig
    return


@app.cell
def _(remote):
    _df = remote(
        """
        SELECT *
        FROM latest_posts p
        WHERE contains(text, 'If the goal is simply to beat Ruto, they’ll settle for Gachagua')
        """
    ).df()

    _df
    return


if __name__ == "__main__":
    app.run()
