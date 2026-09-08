"""Frozen corpus snapshots: name a corpus once, read the same one forever.

    from kma.bench import snapshot, load, pinned_source
    snapshot("2026-09-05-promotion-off")          # writes the manifest
    m = load("2026-09-05-promotion-off")
    con.sql(f"SELECT count(*) FROM {pinned_source(m, 'posts', type='search')}")

The corpus is append-only and the collector's own parameters change what lands
in it, so any measurement not pinned to a snapshot is confounded by collection
that happened while it ran. A snapshot is a MANIFEST over immutable per-run
objects, never a copy: R2 objects are written once and never rewritten, so
listing them by path is enough to freeze what a read can see.

Row counts come from parquet footers, not from scanning: one range request per
object rather than a full read.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from datetime import datetime, timezone

import duckdb
import pandas as pd

from kma.db import BUCKET, connect

log = logging.getLogger(__name__)

# Every prefix that carries corpus state. `bench/` is deliberately absent: a
# snapshot never covers other snapshots.
PREFIXES = (
    "posts",
    "authors",
    "engagements",
    "follows",
    "metrics",
    "embeddings",
    "labels",
    "incitement",
    "hatespeech",
    "topics",
    "coordination",
    "census_runs",
    # The census TTL ledger: per-object selection truth, and the only source
    # that distinguishes "not selected" from "selected and empty". Captured
    # since 2026-09-08; a snapshot without it cannot support per-id replay.
    "census_ttl",
    "collection_runs",
    "stories",
)

SNAPSHOT_PREFIX = "bench"


def _r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _r2_uri(key: str, bucket: str = BUCKET) -> str:
    return f"r2://{bucket}/{key}"


def list_objects(client, bucket: str, prefix: str) -> Iterator[dict]:
    """Every object under `prefix`, paginated. Yields key, size, etag, modified."""
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": f"{prefix}/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for obj in page.get("Contents", []):
            yield {
                "key": obj["Key"],
                "size": int(obj["Size"]),
                # boto3 hands back the etag quoted; the quotes are not part of it.
                "etag": str(obj.get("ETag", "")).strip('"'),
                "last_modified": obj.get("LastModified"),
            }
        if not page.get("IsTruncated"):
            return
        token = page.get("NextContinuationToken")


def partition_values(key: str) -> dict[str, str]:
    """Hive partition values from a key: `a=1/b=2/f.parquet` -> {a: 1, b: 2}."""
    out: dict[str, str] = {}
    for part in key.split("/")[:-1]:
        name, sep, value = part.partition("=")
        if sep:
            out[name] = value
    return out


def _row_counts(
    con: duckdb.DuckDBPyConnection, prefix: str, uri: Callable[[str], str]
) -> pd.DataFrame:
    """One row per file: footer `num_rows`. Empty frame when the prefix has no
    parquet under it - a prefix that was never written is not an error."""
    glob = uri(f"{prefix}/**/*.parquet")
    empty = pd.DataFrame({"file_name": pd.Series(dtype="object"), "rows": pd.Series(dtype="int64")})
    try:
        return con.sql(
            f"SELECT file_name, num_rows AS rows FROM parquet_file_metadata('{glob}')"
        ).df()
    except duckdb.Error as exc:
        log.info("no parquet under %s (%s)", prefix, str(exc).split("\n")[0])
        return empty


def _code_versions(con: duckdb.DuckDBPyConnection, uri: Callable[[str], str]) -> tuple[str, str]:
    """The collector `code_version` range the snapshot spans. Series aggregated
    across a parameter change are wrong, so the range is recorded up front."""
    glob = uri("census_runs/**/*.parquet")
    try:
        row = con.sql(
            f"""SELECT min(code_version), max(code_version)
                FROM read_parquet('{glob}', union_by_name=true)
                WHERE code_version IS NOT NULL AND code_version <> ''"""
        ).fetchone()
    except duckdb.Error:
        return ("", "")
    return (row[0] or "", row[1] or "") if row else ("", "")


def snapshot(
    name: str,
    *,
    prefixes: tuple[str, ...] = PREFIXES,
    rows: bool = True,
    con: duckdb.DuckDBPyConnection | None = None,
    client=None,
    bucket: str = BUCKET,
    uri: Callable[[str], str] | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Freeze what the corpus currently is, and write the manifest to R2.

    `rows=False` skips footer reads: fast, but the manifest then cannot be used
    to check that a pinned read is stable, which is the point of taking one.
    """
    con = con or connect()
    client = client or _r2_client()
    uri = uri or (lambda key: _r2_uri(key, bucket))

    frames = []
    for prefix in prefixes:
        listing = pd.DataFrame(list(list_objects(client, bucket, prefix)))
        if listing.empty:
            log.info("%s: no objects", prefix)
            continue
        listing["prefix"] = prefix
        listing["path"] = [uri(k) for k in listing["key"]]
        parts = pd.DataFrame([partition_values(k) for k in listing["key"]])
        listing["platform"] = parts.get("platform", pd.Series(index=listing.index, dtype="object"))
        listing["type"] = parts.get("type", pd.Series(index=listing.index, dtype="object"))
        listing["kind"] = parts.get("kind", pd.Series(index=listing.index, dtype="object"))
        listing["dt"] = parts.get("dt", pd.Series(index=listing.index, dtype="object"))
        listing["model"] = parts.get("model", pd.Series(index=listing.index, dtype="object"))

        if rows:
            counts = _row_counts(con, prefix, uri)
            listing = listing.merge(counts, how="left", left_on="path", right_on="file_name")
            listing = listing.drop(columns=["file_name"])
        else:
            listing["rows"] = pd.NA

        log.info("%s: %d objects, %s rows", prefix, len(listing), listing["rows"].sum(skipna=True))
        frames.append(listing)

    if not frames:
        raise RuntimeError(f"snapshot {name!r} found no objects; refusing to write an empty manifest")

    manifest = pd.concat(frames, ignore_index=True)
    manifest["snapshot"] = name
    lo, hi = _code_versions(con, uri)
    manifest["created_at"] = datetime.now(timezone.utc)
    manifest["code_version_min"] = lo
    manifest["code_version_max"] = hi
    manifest["kma_sha"] = os.getenv("GIT_SHA", "")

    if write:
        key = f"{SNAPSHOT_PREFIX}/snapshot={name}/manifest.parquet"
        con.register("_manifest", manifest)
        try:
            con.execute(f"COPY _manifest TO '{uri(key)}' (FORMAT parquet, COMPRESSION zstd)")
        finally:
            con.unregister("_manifest")
        log.info("wrote %s (%d objects)", key, len(manifest))

    return manifest


def load(
    name: str,
    con: duckdb.DuckDBPyConnection | None = None,
    bucket: str = BUCKET,
    uri: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Read back a manifest written by `snapshot`."""
    con = con or connect()
    uri = uri or (lambda key: _r2_uri(key, bucket))
    key = f"{SNAPSHOT_PREFIX}/snapshot={name}/manifest.parquet"
    return con.sql(f"SELECT * FROM read_parquet('{uri(key)}')").df()


def pinned_source(manifest: pd.DataFrame, prefix: str, **filters: str) -> str:
    """A `read_parquet(...)` expression over exactly the objects in the
    manifest - an explicit path list, never a glob.

    A glob re-resolves at read time and would pick up everything collected
    since, which is the confound the snapshot exists to remove. `**filters`
    match hive partition columns, e.g. `type="search"`.
    """
    sel = manifest[manifest["prefix"] == prefix]
    for column, value in filters.items():
        if column not in sel.columns:
            raise ValueError(f"unknown partition column {column!r} for prefix {prefix!r}")
        sel = sel[sel[column] == value]
    paths = sorted(sel["path"].dropna().tolist())
    if not paths:
        raise ValueError(f"snapshot holds no objects for prefix {prefix!r} with {filters}")
    listed = ", ".join(f"'{p}'" for p in paths)
    return f"read_parquet([{listed}], union_by_name=true, hive_partitioning=true)"


def diff(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Objects added between two snapshots, counted by prefix.

    Only additions are reported as growth; anything MISSING from the later
    snapshot is reported too, because per-run objects are supposed to be
    immutable and a disappearance means something deleted history.
    """
    b = set(before["path"])
    a = set(after["path"])
    added = after[after["path"].isin(a - b)]
    removed = before[before["path"].isin(b - a)]
    out = (
        pd.concat(
            [
                added.groupby("prefix").agg(added=("path", "size"), added_rows=("rows", "sum")),
                removed.groupby("prefix").agg(removed=("path", "size")),
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    return out


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Freeze the corpus as a named snapshot.")
    ap.add_argument("name", help="snapshot name, e.g. 2026-09-05-promotion-off")
    ap.add_argument("--no-rows", action="store_true", help="skip parquet footer reads")
    args = ap.parse_args()

    manifest = snapshot(args.name, rows=not args.no_rows)
    by_prefix = manifest.groupby("prefix").agg(objects=("path", "size"), rows=("rows", "sum"))
    print(by_prefix.to_string())
    print(f"\ntotal: {len(manifest)} objects, {int(manifest['rows'].sum(skipna=True))} rows")


if __name__ == "__main__":
    main()
