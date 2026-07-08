from __future__ import annotations

import asyncio
import logging

import typer

from kenya_monitor.config import PlatformTargets, R2Config, load_accounts, load_targets
from kenya_monitor.storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

app = typer.Typer(help="Kenya 2027 social media monitor", no_args_is_help=True)
accounts_app = typer.Typer(help="Manage the X account pool", no_args_is_help=True)
app.add_typer(accounts_app, name="accounts")


def _storage() -> Storage:
    return Storage(R2Config.from_env())


@app.command()
def check() -> None:
    """Round-trip a probe through R2 to verify auth + Parquet write/read."""
    n = _storage().healthcheck()
    typer.echo(f"read back {n} probe row(s) from R2 -> auth + round-trip OK")


@app.command()
def query(sql: str) -> None:
    """Run a DuckDB SQL query against R2. Use the `posts` glob helper via {posts}."""
    storage = _storage()
    sql = sql.format(posts=storage.posts_view())
    storage.query(sql).show()


@app.command()
def targets() -> None:
    """Print the configured collection targets."""
    for platform, t in load_targets().items():
        typer.echo(f"[{platform}] {len(t.accounts)} accounts, {len(t.keywords)} keywords")


@accounts_app.command("add")
def accounts_add() -> None:
    """Load config/accounts.yaml into the twscrape pool and log in."""
    from kenya_monitor.collectors.x import build_api
    from kenya_monitor.accounts import sync_accounts

    async def _run() -> None:
        api = build_api()
        result = await sync_accounts(api, load_accounts())
        info = await api.pool.accounts_info()
        typer.echo(
            f"synced pool: +{result.added} new, {result.updated} updated; "
            f"{result.active} active / {len(info)} total"
        )
        for a in info:
            if not a["active"]:
                typer.echo(f"  @{a['username']}  active=False  err={a['error_msg']}")

    asyncio.run(_run())


@accounts_app.command("sync")
def accounts_sync(
    relogin: bool = typer.Option(True, help="attempt relogin on failed accounts"),
) -> None:
    """Refresh proxy/cookie changes from yaml and maintain the pool."""
    from kenya_monitor.accounts import pool_health, sync_accounts
    from kenya_monitor.collectors.x import build_api

    async def _run() -> None:
        api = build_api()
        result = await sync_accounts(api, load_accounts(), relogin_failed=relogin)
        health = await pool_health(api.pool)
        typer.echo(
            f"sync: +{result.added} new, {result.updated} updated, "
            f"{result.relogin_attempted} relogin attempts; "
            f"{health.active} active, {health.inactive} inactive, {health.locked} locked"
        )

    asyncio.run(_run())


@accounts_app.command("stats")
def accounts_stats() -> None:
    """Show pool health and rotation order."""
    from kenya_monitor.accounts import pool_health
    from kenya_monitor.collectors.x import build_api

    async def _run() -> None:
        api = build_api()
        health = await pool_health(api.pool)
        typer.echo(f"pool: {health.active} active / {health.total} total ({health.locked} locked)")
        typer.echo(f"rotation: {api.pool._order_by}")
        for a in await api.pool.accounts_info():
            used = a["last_used"].isoformat() if a["last_used"] else "never"
            typer.echo(
                f"  @{a['username']:20} active={a['active']}  reqs={a['total_req']:4}  last={used}"
            )

    asyncio.run(_run())


@accounts_app.command("list")
def accounts_list() -> None:
    """Show accounts currently in the pool."""
    from kenya_monitor.collectors.x import build_api

    async def _run() -> None:
        info = await build_api().pool.accounts_info()
        for a in info:
            typer.echo(f"@{a['username']}  active={a['active']}  last_used={a['last_used']}")

    asyncio.run(_run())


@app.command()
def collect(
    platform: str = typer.Argument("x"),
    keywords: bool = typer.Option(False, "--keywords", help="collect configured keyword searches"),
    accounts: bool = typer.Option(False, "--accounts", help="collect configured account timelines"),
    query: str = typer.Option("", "--query", help="one-off search query (test mode)"),
    handle: str = typer.Option("", "--handle", help="one-off account timeline (test mode)"),
    limit: int = typer.Option(50, help="max posts per target"),
) -> None:
    """Run a one-off X collection and write results to R2."""
    if platform != "x":
        raise typer.BadParameter("only 'x' is wired so far")
    from kenya_monitor.collectors.x import full_window
    from kenya_monitor.config import SEARCH_MIN_FAVES
    from kenya_monitor.runner import build_x_collector, collect_x

    targets = load_targets().get("x", PlatformTargets())
    if query or handle:
        targets = PlatformTargets(
            accounts=[handle] if handle else [],
            keywords=[query] if query else [],
        )
        keywords, accounts = bool(query), bool(handle)

    async def _run() -> None:
        storage = _storage()
        collector = await build_x_collector(load_accounts())
        counts = await collect_x(
            collector,
            storage,
            targets,
            search_windows=full_window(),
            min_faves=SEARCH_MIN_FAVES,
            window_limit=limit,
            timeline_limit=limit,
            keywords=keywords,
            accounts=accounts,
        )
        typer.echo(f"collected: {counts}")

    asyncio.run(_run())


@app.command()
def run(
    once: bool = typer.Option(False, "--once", help="run a single post-collection pass and exit"),
    limit: int = typer.Option(50, help="max posts per target per run"),
) -> None:
    """Scheduled worker: post collection (~6x/day) + metrics refresh (~hourly), randomized."""
    from kenya_monitor.scheduler import run_once, run_scheduler

    if once:
        counts = asyncio.run(run_once(limit, include_backfill=True))
        typer.echo(f"collected: {counts}")
    else:
        asyncio.run(run_scheduler(limit))


@app.command()
def backfill(
    days: int = typer.Option(14, help="look-back window in days (daily granularity)"),
    limit: int = typer.Option(20, help="max posts per keyword per day-window"),
) -> None:
    """One-time deep backfill to even out historical temporal coverage (engaged tweets only)."""
    from kenya_monitor.scheduler import run_backfill_once

    counts = asyncio.run(run_backfill_once(days=days, window_limit=limit))
    typer.echo(f"backfill: {counts}")


@app.command()
def snowball(
    top_retweeted: int = typer.Option(None, help="hot reposted objects to census (default from env)"),
    top_conversations: int = typer.Option(None, help="hot conversations to hydrate (default from env)"),
    retweeters_limit: int = typer.Option(None, help="max retweeters per object (default from env)"),
) -> None:
    """One snowball pass: retweeter lists, reply threads, referenced-original hydration."""
    from kenya_monitor.scheduler import run_snowball_once

    overrides = {
        k: v
        for k, v in {
            "top_retweeted": top_retweeted,
            "top_conversations": top_conversations,
            "retweeters_limit": retweeters_limit,
        }.items()
        if v is not None
    }
    counts = asyncio.run(run_snowball_once(**overrides))
    typer.echo(f"snowball: {counts}")


@app.command()
def adapt(
    dry_run: bool = typer.Option(False, "--dry-run", help="compute promotions without saving"),
) -> None:
    """Run one adaptive-promotion pass (bursting hashtags + cluster accounts)."""
    from kenya_monitor import adaptive

    storage = _storage()
    entries = adaptive.promote(
        storage.con,
        storage.posts_view(platform="x"),
        storage.clusters_view(platform="x"),
        storage.authors_view(platform="x"),
        dry_run=dry_run,
    )
    for e in entries:
        typer.echo(f"{e.kind:8} {e.value}  ({e.source}, confirmed {e.last_confirmed})")
    typer.echo(f"{len(entries)} dynamic target(s){' (dry run, not saved)' if dry_run else ''}")


@app.command()
def follows(
    handle: list[str] = typer.Option(None, "--handle", help="explicit handles (else flagged clusters)"),
    limit: int = typer.Option(None, help="max edges per direction per account (default from env)"),
) -> None:
    """Fetch follower/following edges for flagged-cluster members."""
    from kenya_monitor.config import FOLLOW_FETCH_LIMIT
    from kenya_monitor.scheduler import run_follows_once

    counts = asyncio.run(
        run_follows_once(handles=list(handle) or None, limit=limit or FOLLOW_FETCH_LIMIT)
    )
    typer.echo(f"follows: {counts}")


@app.command()
def metrics(
    days: int = typer.Option(5, help="look-back window in days"),
    top_pct: float = typer.Option(0.05, help="top fraction by engagement to refresh"),
    max_posts: int = typer.Option(200, help="rate-limit safety cap on posts per pass"),
) -> None:
    """Run one metrics-refresh pass: top `top_pct` of last `days` days by likes+quotes+reposts."""
    from kenya_monitor.scheduler import run_metrics_once

    counts = asyncio.run(run_metrics_once(since_days=days, top_pct=top_pct, max_posts=max_posts))
    typer.echo(f"metrics: {counts}")


if __name__ == "__main__":
    app()
