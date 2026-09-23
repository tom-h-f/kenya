"""Modal app: plant the weekly canary. NOT DEPLOYED - deploying it is Tom's call.

    uv run --with modal modal run modal_canary.py --dry-run     # plan, write nothing
    uv run --with modal modal run modal_canary.py               # plant once
    uv run --with modal modal deploy modal_canary.py            # Mondays 05:00 UTC

Plants a group shaped from the UAE operation (see `kma.canary`) into the
canary prefix, and writes the leads contract's expectation row with a 36-hour
deadline. 05:00 UTC Monday is before the 06:30 leads pass, so the daily pass
that runs at or before it picks the group up.

The daily v2 pass has to UNION the planted posts into its window for any of
this to be detected: `canary.load_active(con)` then `canary.inject_view(con,
view, posts)` before building networks, and `canary.text_edges(posts)` added
to the text trace. That wiring belongs in `modal_v2_daily.py` (workstream A)
and is left to the merge; until it is in, a planted canary is never seen and
the poller sends `[CANARY] MISSING` - which is the failure it exists to raise.

The leads pass (workstream B) should build a canary community's dossier with
`canary.packet(load_active(con), members, cluster_id)` instead of
`dossier.build`, which reads the archive and finds nothing.
"""

import modal

app = modal.App("kma-canary")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas>=2", "pyarrow>=18", "numpy>=1.26", "networkx>=3.3",
                 "duckdb>=1.1", "python-dotenv>=1.0")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
)

SIZE = 20
ACTIVITY = "normal"
DAYS = 7


def _plant(dry_run: bool) -> dict:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from kma import canary, coord2
    from kma.db import connect

    con = connect()
    con.execute("SET memory_limit='4GB'")
    view = (f"SELECT * FROM ({coord2.posts_view()}) "
            f"WHERE created_at >= now() - INTERVAL {DAYS} DAY")
    p = canary.new(con, view, size=SIZE, activity=ACTIVITY, days=DAYS)
    out = {"canary_id": p.canary_id, "accounts": len(p.accounts), "posts": len(p.posts),
           "deadline": p.deadline.isoformat()}
    if not dry_run:
        out["keys"] = canary.persist(con, p)
    return out


@app.function(image=image, secrets=[modal.Secret.from_name("kenya-r2")],
              cpu=2, memory=8192, timeout=60 * 30,
              schedule=modal.Cron("0 5 * * 1"))
def weekly() -> dict:
    return _plant(dry_run=False)


@app.function(image=image, secrets=[modal.Secret.from_name("kenya-r2")],
              cpu=2, memory=8192, timeout=60 * 30)
def once(dry_run: bool = True) -> dict:
    return _plant(dry_run)


@app.local_entrypoint()
def main(dry_run: bool = False):
    print(once.remote(dry_run))
