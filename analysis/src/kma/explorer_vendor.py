"""Host the DuckDB-WASM binaries in the dashboard bucket, versioned and immutable.

    uv run --frozen --no-dev python -m kma.explorer_vendor            # upload what is missing
    uv run --frozen --no-dev python -m kma.explorer_vendor --check    # list, upload nothing

The wasm is 35.7 MB, over the 25 MiB a Workers asset may be, so the site serves it
from `explorer/v1/vendor/` through its data route. Each file is stored under a name
that carries the package version (`duckdb-eh-1.29.0.wasm`; the parquet extension, which
DuckDB-WASM 1.29.0 (core 1.1.1) otherwise fetches from extensions.duckdb.org and the site's
CSP does not allow, as `parquet.eh-1.1.1.wasm`) with
`cache-control: public, max-age=31536000, immutable`, because a name that carries a
version never changes in place. The pinned sha256 of every file is in
`explorer/vendor.json`; a download that differs is refused, so an upgrade is a
reviewed change to that file. Uses the dashboard key and refuses the raw bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.request
from pathlib import Path

from kma import db, explorer_export

log = logging.getLogger("kma.explorer")

PIN_PATH = db.MONOREPO_ROOT / "explorer" / "vendor.json"
VENDOR_PREFIX = f"{explorer_export.PREFIX}/vendor/"
CACHE = explorer_export.RUN_FILE_CACHE
CONTENT_TYPE = {".wasm": "application/wasm", ".js": "text/javascript"}


def load_pin(path: Path = PIN_PATH) -> dict:
    pin = json.loads(path.read_text())
    if not pin.get("version") or not pin.get("files"):
        raise ValueError(f"{path}: needs version and files")
    return pin


def target_name(source_name: str, version: str, entry: dict | None = None) -> str:
    """The versioned name a file is stored under. A pin entry may carry its own
    `name` (the parquet extension must start with `parquet.` because DuckDB derives
    its init symbol from the text before the first dot) and its own `version`."""
    if entry and entry.get("name"):
        return entry["name"]
    stem, dot, ext = source_name.rpartition(".")
    if source_name.endswith(".worker.js"):
        return f"{source_name[:-len('.js')]}-{version}.js"
    return f"{stem}-{version}{dot}{ext}"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def sync(client, bucket: str, pin: dict, *, download=fetch, check_only: bool = False) -> list[dict]:
    if bucket == db.BUCKET:
        raise ValueError(f"{bucket!r} is the raw data bucket; vendor files go to the dashboard bucket")
    out = []
    for source_name, expected in sorted(pin["files"].items()):
        name = target_name(source_name, pin["version"], expected)
        key = VENDOR_PREFIX + name
        try:
            head = client.head_object(Bucket=bucket, Key=key)
            present = True
            size = int(head["ContentLength"])
        except client.exceptions.ClientError:
            present, size = False, None
        entry = {"file": name, "key": key, "present": present, "bytes": size, "uploaded": False}
        if not present and not check_only:
            data = download(expected.get("url") or pin["url"].format(
                version=pin["version"], file=source_name))
            digest = hashlib.sha256(data).hexdigest()
            if digest != expected["sha256"]:
                raise ValueError(f"{source_name}: sha256 {digest} is not the pinned {expected['sha256']}")
            client.put_object(Bucket=bucket, Key=key, Body=data, CacheControl=CACHE,
                              ContentType=CONTENT_TYPE[Path(name).suffix])
            entry.update(present=True, bytes=len(data), uploaded=True)
            log.info("uploaded %s (%d bytes)", key, len(data))
        out.append(entry)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Host the pinned DuckDB-WASM files in the bucket.")
    ap.add_argument("--check", action="store_true", help="report what is missing, upload nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from kma import dashboard

    client = explorer_export.r2_client()
    print(json.dumps(sync(client, dashboard.DASHBOARD_BUCKET, load_pin(), check_only=args.check),
                     indent=2))


if __name__ == "__main__":
    main()
