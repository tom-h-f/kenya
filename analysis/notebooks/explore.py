import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from kma import connect_quack
    from kma import viz

    viz.use_theme()
    con = connect_quack()

    def remote(sql: str):
        """Run SQL on the tf1 server. Views: posts, latest_posts, metrics, authors, latest_authors."""
        return con.sql(f"FROM kenya.query('{sql.replace(chr(39), chr(39) * 2)}')")

    return mo, np, remote, viz


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
    mo.vstack([
        mo.md(
            "# Kenya 2027 monitor\n"
            "The **raw-data overview**: how much has been collected, who posts, "
            "when, and what gets engagement - before any modelling. Queried live "
            "through the **tf1 quack server**.\n\n"
            "- **Unique posts** - distinct tweets (deduped to their latest state).\n"
            "- **Snapshots** - every collection of a post over time; the ratio to "
            "unique posts shows how often we re-measure engagement.\n"
            "- **Authors** - distinct accounts seen.\n"
            "- **Metric snapshots** - engagement re-measurements of hot posts."
        ),
        mo.hstack(
            [
                mo.stat(value=f"{_t.get('latest_posts', 0):,}", label="Unique posts", bordered=True),
                mo.stat(value=f"{_t.get('posts', 0):,}", label="Snapshots", bordered=True),
                mo.stat(value=f"{_t.get('authors', 0):,}", label="Authors", bordered=True),
                mo.stat(value=f"{_t.get('metrics', 0):,}", label="Metric snapshots", bordered=True),
            ],
            widths="equal",
        ),
    ])
    return


@app.cell
def _(remote, viz):
    _vol = remote(
        """
        SELECT date_trunc('hour', created_at) AS hour, count(*) AS posts
        FROM latest_posts WHERE platform = 'x' AND created_at > now() - INTERVAL 14 DAY
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    _fig, _ax = viz.new_fig(9, 3.8)
    _ax.plot(_vol["hour"], _vol["posts"], color=viz.BLUE)
    _ax.fill_between(_vol["hour"], _vol["posts"], alpha=0.10, color=viz.BLUE, linewidth=0)
    _ax.set_title("Post volume per hour, last 14 days")
    _ax.set_ylabel("posts")
    _ax.set_ylim(bottom=0)
    _ax.grid(axis="x", visible=False)
    _fig.autofmt_xdate(rotation=0, ha="center")
    _fig
    return


@app.cell
def _(np, remote, viz):
    _eng = remote(
        "SELECT like_count FROM latest_posts WHERE platform = 'x' AND like_count > 0"
    ).df()
    _fig, _ax = viz.new_fig(9, 3.8)
    _bins = np.geomspace(1, max(_eng["like_count"].max(), 10), 30)
    _ax.hist(_eng["like_count"], bins=_bins, color=viz.BLUE, linewidth=0, rwidth=0.92)
    _ax.set_xscale("log")
    _ax.set_title("Likes per tweet (log scale)")
    _ax.set_xlabel("likes")
    _ax.set_ylabel("tweets")
    _ax.grid(axis="x", visible=False)
    _fig
    return


@app.cell
def _(remote, viz):
    _top = remote(
        """
        SELECT author_handle, count(*) AS posts, sum(like_count) AS likes
        FROM latest_posts WHERE platform = 'x'
        GROUP BY 1 ORDER BY likes DESC LIMIT 15
        """
    ).df()
    _fig, _ax = viz.new_fig(9, 5.5)
    _fig.subplots_adjust(left=0.18)
    viz.hbars(_ax, _top["author_handle"], _top["likes"])
    _ax.set_title("Top authors by total likes")
    _ax.set_xlabel("likes")
    _fig
    return


@app.cell
def _(remote, viz):
    _traj = remote(
        """
        WITH top AS (
            SELECT platform_post_id FROM latest_posts WHERE platform = 'x'
            ORDER BY like_count DESC LIMIT 100
        )
        SELECT p.platform_post_id AS id, p.collected_at, p.like_count
        FROM posts p JOIN top USING (platform_post_id)
        ORDER BY p.collected_at
        """
    ).df()
    _fig, _ax = viz.new_fig(9, 4.2)
    _stats = _traj.groupby("id").agg(snaps=("like_count", "size"), peak=("like_count", "max"))
    _tracked = _stats[_stats["snaps"] >= 3]
    _lead = (_tracked if len(_tracked) else _stats)["peak"].idxmax()
    for _id, _g in _traj.groupby("id"):
        if _id == _lead:
            continue
        _ax.plot(_g["collected_at"], _g["like_count"], color=viz.DEEMPH,
                 linewidth=1.1, alpha=0.6, zorder=2)
    _g = _traj[_traj["id"] == _lead]
    _ax.plot(
        _g["collected_at"], _g["like_count"], color=viz.BLUE, linewidth=2.2,
        marker="o", markersize=5, markeredgecolor=viz.SURFACE,
        markeredgewidth=1.2, zorder=3,
    )
    _ax.annotate(
        viz.compact(_g["like_count"].iloc[-1]),
        (_g["collected_at"].iloc[-1], _g["like_count"].iloc[-1]),
        xytext=(8, 0), textcoords="offset points",
        va="center", fontsize=9, color=viz.INK_2,
    )
    viz.legend_swatches(
        _ax,
        [("most-liked tracked tweet", viz.BLUE), ("other top-100 tweets", viz.DEEMPH)],
        loc="lower left",
    )
    _ax.set_yscale("log")
    _ax.set_title("Engagement trajectories of the top 100 tweets, as resampled")
    _ax.set_ylabel("likes")
    _ax.grid(axis="x", visible=False)
    _fig.autofmt_xdate(rotation=0, ha="center")
    _fig
    return


@app.cell
def _(remote, viz):
    _fl = remote(
        """
        SELECT a.handle, a.followers_count AS followers, a.verified,
               sum(p.like_count) AS likes, count(*) AS posts
        FROM latest_posts p JOIN latest_authors a ON p.author_id = a.platform_user_id
        WHERE a.platform = 'x' AND a.followers_count > 0
        GROUP BY 1, 2, 3
        """
    ).df()
    _fl = _fl[_fl["likes"] > 0]
    _fig, _ax = viz.new_fig(9, 5)
    for _verified, _color, _z in ((False, viz.DEEMPH, 2), (True, viz.BLUE, 3)):
        _g = _fl[_fl["verified"] == _verified]
        _ax.scatter(
            _g["followers"], _g["likes"], s=22, color=_color, zorder=_z,
            edgecolors=viz.SURFACE, linewidths=0.8,
        )
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    viz.legend_swatches(_ax, [("verified", viz.BLUE), ("unverified", viz.DEEMPH)], loc="upper left")
    _ax.set_title("Author reach vs engagement")
    _ax.set_xlabel("followers")
    _ax.set_ylabel("total likes")
    _fig
    return


@app.cell
def _(remote, viz):
    _cadence = remote(
        """
        SELECT extract('hour' FROM created_at) AS hour, count(*) AS posts
        FROM latest_posts WHERE platform = 'x'
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    _cadence = _cadence.set_index("hour").reindex(range(24), fill_value=0).reset_index()
    _fig, _ax = viz.new_fig(9, 3.5)
    viz.vbars(_ax, [f"{int(h):02d}" for h in _cadence["hour"]], _cadence["posts"])
    _ax.set_title("Posting cadence by hour of day")
    _ax.set_xlabel("hour (UTC)")
    _ax.set_ylabel("posts")
    _fig
    return


@app.cell
def _(remote, viz):
    _lang = remote(
        """
        SELECT lang, count(*) AS posts FROM latest_posts
        WHERE platform = 'x' AND lang IS NOT NULL
        GROUP BY 1 ORDER BY posts DESC LIMIT 12
        """
    ).df()
    _fig, _ax = viz.new_fig(9, 4.5)
    viz.hbars(_ax, _lang["lang"], _lang["posts"])
    _ax.set_title("Post volume by detected language")
    _ax.set_xlabel("posts")
    _fig
    return


@app.cell
def _(remote, viz):
    from kma.deltas import region_case

    _region = remote(
        f"""
        SELECT {region_case('a.location')} AS region, count(*) AS posts
        FROM latest_posts p JOIN latest_authors a ON p.author_id = a.platform_user_id
        WHERE p.platform = 'x'
        GROUP BY 1 HAVING region IS NOT NULL ORDER BY posts DESC
        """
    ).df()
    _fig, _ax = viz.new_fig(9, 4)
    _fig.subplots_adjust(left=0.14)
    viz.hbars(_ax, _region["region"], _region["posts"])
    _ax.set_title("Post volume by author region, profile-location proxy")
    _ax.set_xlabel("posts")
    _fig
    return


if __name__ == "__main__":
    app.run()
