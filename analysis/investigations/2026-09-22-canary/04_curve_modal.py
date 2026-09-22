"""The full-scale detection-strength curve on Modal. UNRUN: written 2026-09-22
while the workspace was over its spend limit.

    cd analysis && uv run --with modal modal run --detach \\
        investigations/2026-09-22-canary/04_curve_modal.py --snapshot 2026-09-14-deep500 --days 0

`03_curve.py`'s points over the whole snapshot (`--days 0`) instead of one
week, with the snapshot's persisted text trace, one container per activity
level. Results go to the `iohunter-bench` volume under `canary/`, never to R2.
"""

import modal

app = modal.App("kma-canary-curve")
vol = modal.Volume.from_name("iohunter-bench")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas>=2", "pyarrow>=18", "numpy>=1.26", "networkx>=3.3", "scipy>=1.13",
                 "scikit-learn>=1.5", "duckdb>=1.1", "python-dotenv>=1.0", "python-igraph>=0.11",
                 "leidenalg>=0.10", "sentence-transformers>=5.0", "torch>=2.4")
    .env({"PYTHONPATH": "/root/src:/root/inv"})
    .add_local_dir("src", remote_path="/root/src", ignore=["**/__pycache__/**", "*.pyc"])
    .add_local_file("investigations/2026-09-22-canary/03_curve.py", "/root/inv/curve.py")
)


@app.function(image=image, secrets=[modal.Secret.from_name("kenya-r2")], volumes={"/data": vol},
              cpu=8, memory=32768, timeout=60 * 60 * 6, max_containers=2)
def curve(snapshot: str, days: int, activity: str, sizes: str, seeds: str) -> list[dict]:
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import pandas as pd
    from sentence_transformers import SentenceTransformer

    import curve as c
    from kma import bench, canary, coord2, coord2_run
    from kma.db import connect
    from kma.semantic import MODEL

    con = connect()
    view = coord2.posts_view(bench.pinned_source(bench.load(snapshot, con=con), "posts"))
    if days:
        view = f"SELECT * FROM ({view}) WHERE created_at >= now() - INTERVAL {int(days)} DAY"
    canary.check_tags_unused(con, view)
    text = coord2_run.load_textsim(con, snapshot, 0.85, coord2.TEXT_MIN_OVERLAP)
    start = pd.Timestamp(con.sql(f"SELECT max(created_at) FROM ({view})").fetchone()[0])
    start = (start - pd.Timedelta(days=7)).to_pydatetime()
    encoder = SentenceTransformer(MODEL, device="cpu")
    rows = [c.point(con, view, text, encoder, size=int(s), activity=activity, seed=int(k),
                    start=start)
            for s in sizes.split(",") for k in seeds.split(",")]
    out = Path(f"/data/canary/curve-{snapshot}-{activity}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, default=str, indent=1))
    vol.commit()
    return rows


@app.local_entrypoint()
def main(snapshot: str = "2026-09-14-deep500", days: int = 0, sizes: str = "3,5,10,20,50",
         seeds: str = "0,1,2"):
    for rows in curve.starmap([(snapshot, days, a, sizes, seeds) for a in ("low", "normal")]):
        for r in rows:
            print(r)
