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
def stats(
    platform: str = typer.Option("x", help="platform prefix to summarise"),
    hours: int = typer.Option(24, help="recent activity window in hours"),
    days: int = typer.Option(7, help="daily breakdown window in days"),
) -> None:
    """Summarise total and recent data collected in R2."""
    from kenya_monitor.stats import format_stats, gather_stats

    summary = gather_stats(_storage(), platform=platform, recent_hours=hours, daily_days=days)
    typer.echo(format_stats(summary))


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
    """Always-on worker: continuous cycles of posts -> snowball -> metrics ->
    follow crawl, throttled only by per-account pacing + twscrape rate limits."""
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


@app.command("hate-seek")
def hate_seek_cmd(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print the rendered queries; makes no network requests"
    ),
    terms: list[str] = typer.Option(None, "--terms", help="explicit register terms (else rotated)"),
    limit: int = typer.Option(None, help="max posts per term per window (default from env)"),
    max_terms: int = typer.Option(None, help="terms per pass (default from env)"),
) -> None:
    """One hate-seeking search pass over the coded-term register.

    The register (config/hate_terms.yaml) holds documented NCIC/PeaceTech coded
    incitement markers. A hit is a triage seed, never a verdict - most of these
    words have innocent everyday senses, which is what fp_risk anchoring is for.
    Results land in quarantined partitions excluded from prevalence stats.
    """
    from kenya_monitor import hate_seek as hs
    from kenya_monitor.collectors.x import recent_windows
    from kenya_monitor.config import (
        HATE_SEEK_MAX_TERMS,
        HATE_SEEK_RECENT_DAYS,
        HATE_SEEK_WINDOW_LIMIT,
        HATE_TERMS_PATH,
        load_hate_terms,
    )

    register = load_hate_terms(HATE_TERMS_PATH)
    if dry_run:
        wanted = {t.lower() for t in (terms or [])}
        if wanted:
            everything = hs.select_terms(register, max_terms=10_000, max_targets=10_000)
            queries = [q for q in everything if q.term.lower() in wanted]
        else:
            queries = hs.select_terms(
                register, state=hs.load_state(), max_terms=max_terms or HATE_SEEK_MAX_TERMS
            )
        windows = recent_windows(HATE_SEEK_RECENT_DAYS)
        since, until = windows[0]
        for q in queries:
            typer.echo(f"[{q.source}/{q.fp_risk}] {q.term} -> {q.target_type}")
            typer.echo(f"    {hs.render(q, register.anchors, since=since, until=until)}")
        typer.echo(f"\n{len(queries)} quer(ies), {len(windows)} window(s); nothing requested.")
        return

    from kenya_monitor.scheduler import run_hate_seek_once

    counts = asyncio.run(
        run_hate_seek_once(
            terms=list(terms) if terms else None,
            limit=limit or HATE_SEEK_WINDOW_LIMIT,
            max_terms=max_terms or HATE_SEEK_MAX_TERMS,
        )
    )
    typer.echo(f"hate-seek: {counts}")


@app.command("hate-mine")
def hate_mine_cmd(
    dry_run: bool = typer.Option(False, "--dry-run", help="show candidates without saving"),
    cohort: int = typer.Option(30, help="hate-seed accounts to mine vocabulary from"),
    limit: int = typer.Option(25, help="max candidates to report"),
    approve: list[str] = typer.Option(None, "--approve", help="mark terms human-approved"),
) -> None:
    """Mine emergent coded terms from the hate-seed cohort's vocabulary.

    The classifier cannot see the 2026 coded register, so the cohort is located
    by explicit toxicity and coordination and its *language* is mined by
    contrast against the corpus. Candidates are never searched until a human
    approves them (--approve) or HATE_MINE_AUTOPROMOTE is set.
    """
    from kenya_monitor import hate_signal as hsig
    from kenya_monitor import mined_terms as mt
    from kenya_monitor.config import HATE_MINE_AUTOPROMOTE
    from kenya_monitor.scheduler import _hate_seed_handles

    storage = _storage()
    if approve:
        terms = mt.load()
        wanted = {a.lower() for a in approve}
        hit = [t for t in terms if t.term.lower() in wanted]
        for t in hit:
            t.approved = True
        mt.save(terms)
        typer.echo(f"approved {len(hit)} term(s): {', '.join(t.term for t in hit)}")
        missing = wanted - {t.term.lower() for t in hit}
        if missing:
            typer.echo(f"not found in candidates: {', '.join(sorted(missing))}")
        return

    handles = _hate_seed_handles(storage, cohort)
    posts_view = storage.posts_view(platform="x")
    candidates = hsig.mine_terms(
        storage.con, posts_view, storage.authors_view(platform="x"), handles, n=limit
    )
    if not candidates:
        typer.echo("no candidate terms cleared the floors")
        return
    examples = {
        c["term"]: hsig.term_examples(storage.con, posts_view, c["term"]) for c in candidates
    }
    typer.echo(f"cohort: {len(handles)} accounts\n")
    for c in candidates:
        typer.echo(
            f"{c['term']:24} novelty={c['novelty']:6.2f} lift={c['lift']:6.2f} "
            f"cohort={c['c_in']:4} authors={c['a_in']:3} corpus={c['c_all']:6}"
        )
        for ex in examples[c["term"]][:2]:
            typer.echo(f"    {ex[:150].replace(chr(10), ' ')}")
    if dry_run:
        typer.echo(f"\n{len(candidates)} candidate(s) (dry run, not saved)")
        return
    merged = mt.merge(mt.load(), candidates, examples)
    mt.save(merged)
    n_active = len(mt.active(merged, autopromote=HATE_MINE_AUTOPROMOTE))
    typer.echo(
        f"\nsaved {len(merged)} candidate(s); {n_active} will be searched "
        f"(autopromote={'on' if HATE_MINE_AUTOPROMOTE else 'off'}). "
        "Approve with: monitor hate-mine --approve <term>"
    )


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
    max_accounts: int = typer.Option(None, help="cap accounts per run (default from env)"),
    top_suspicious: int = typer.Option(
        None, "--top-suspicious", help="fetch follows for top-N suspicion-ranked accounts"
    ),
    top_hate: int = typer.Option(
        None, "--top-hate", help="fetch follows for top-N hate-network seed accounts"
    ),
) -> None:
    """Fetch follower/following edges for flagged-cluster members."""
    from kenya_monitor.config import FOLLOW_FETCH_LIMIT, FOLLOW_MAX_ACCOUNTS
    from kenya_monitor.scheduler import run_follows_once

    counts = asyncio.run(
        run_follows_once(
            handles=handle,
            limit=limit or FOLLOW_FETCH_LIMIT,
            max_accounts=max_accounts or (top_hate or top_suspicious or FOLLOW_MAX_ACCOUNTS),
            top_suspicious=top_suspicious,
            top_hate=top_hate,
        )
    )
    typer.echo(f"follows: {counts}")


@app.command("crawl-follows")
def crawl_follows_cmd(
    handle: list[str] = typer.Option(None, "--handle", help="seed handles to start from"),
    limit: int = typer.Option(None, help="max edges per direction per account"),
    max_accounts: int = typer.Option(None, help="accounts to crawl this run"),
    refresh_days: int = typer.Option(None, help="re-crawl after this many days"),
    top_suspicious: int = typer.Option(
        None, "--top-suspicious", help="also seed from top-N suspicion-ranked accounts"
    ),
    top_hate: int = typer.Option(
        None, "--top-hate", help="also seed from top-N hate-network seed accounts"
    ),
    no_edges: bool = typer.Option(
        False, "--no-edges", help="do not queue uncrawled accounts already seen in follows/"
    ),
    status: bool = typer.Option(False, "--status", help="print crawl state summary and exit"),
) -> None:
    """Recursively crawl follower/following graphs; tracks crawl time per account."""
    from kenya_monitor.config import (
        FOLLOW_CRAWL_MAX_PER_RUN,
        FOLLOW_CRAWL_REFRESH_DAYS,
        FOLLOW_FETCH_LIMIT,
    )
    from kenya_monitor.follow_crawl import (
        crawl_summary,
        load_crawl_state,
        pending_count,
    )
    from kenya_monitor.scheduler import run_follow_crawl_once
    from kenya_monitor.storage import Storage
    from kenya_monitor.config import R2Config

    storage = Storage(R2Config.from_env())
    entries = load_crawl_state()
    if status:
        summary = crawl_summary(entries)
        pending = pending_count(storage, refresh_days or FOLLOW_CRAWL_REFRESH_DAYS)
        typer.echo(f"tracked: {summary['tracked']} (ok={summary['ok']}, failed={summary['failed']})")
        typer.echo(f"pending from edges: {pending}")
        if summary["latest_crawl"]:
            typer.echo(f"latest crawl: {summary['latest_crawl']}")
        return

    counts = asyncio.run(
        run_follow_crawl_once(
            seed_handles=handle,
            limit=limit or FOLLOW_FETCH_LIMIT,
            max_accounts=max_accounts or FOLLOW_CRAWL_MAX_PER_RUN,
            refresh_days=refresh_days or FOLLOW_CRAWL_REFRESH_DAYS,
            from_edges=not no_edges,
            top_suspicious=top_suspicious,
            top_hate=top_hate,
        )
    )
    typer.echo(f"crawl-follows: {counts}")


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
