"""Download the public explorer bundle files for the communities under study into out/bundle/.

    uv run python investigations/2026-10-11-text-only-adjudication/00_fetch.py
"""

import urllib.request
from pathlib import Path

RUN = "20261011T033003Z"
BASE = f"https://kenya.tomfr.net/data/{RUN}"
CIDS = [0, 8, 35, 41, 42, 45, 77, 90, 101]
OUT = Path(__file__).parent / "out" / "bundle"


def get(path: str) -> None:
    target = OUT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    request = urllib.request.Request(f"{BASE}/{path}", headers={"User-Agent": "kma-investigation"})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())
    print(f"fetched {path} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    get("communities.json")
    for cid in CIDS:
        get(f"evidence/c{cid}.json")
        get(f"graph/c{cid}.json")
