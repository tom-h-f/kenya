"""Push new leads to a phone. Runs on tf1, beside ntfy.

    uv run kma-leads-notify --dry-run             # print what would be sent
    uv run kma-leads-notify --loop                # the tf1 service

The reader runs on Modal and writes verdicts to R2; ntfy is on tf1 and
tailnet-only, so Modal cannot reach it. This closes the gap from the other
side: it reads verdicts back out of R2 and posts the ones worth a look. It
needs R2 read access and an ntfy token, and never an Anthropic key.

## What it sends

- A **lead**: a production verdict with `worth_a_look` set
  (`kma.leads_run.worth_a_look` decides that, not this).
- A **canary**: any verdict carrying a `canary_id`, whatever the reader made
  of it. The canary tests the chain, so its arriving is the signal; the
  verdict rides along in the body. The title always starts `[CANARY]` and is
  never rewritten.
- A **missing canary**: an expectation under `kind=canary_expected` whose
  deadline passed with no verdict for it. A canary that fails silently would
  make the chain look healthy while it is broken, so its absence is an alert.

## Never twice

Every alert has a key - `run_id:candidate_id` for verdicts, `missing:<id>` for
a missing canary - recorded in a state file right after ntfy accepts it. With
no state file (a fresh host, a lost volume) only verdicts from the last
`--since-hours` are considered, so losing state re-sends at most a day or two
rather than the whole history.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

from kma.db import BUCKET, connect

log = logging.getLogger("kma")

LEADS_PREFIX = "leads/platform=x"
TOPIC = os.getenv("NTFY_TOPIC", "kenya-leads")
# ntfy listens on the tf1 loopback (127.0.0.1:8090); the tailnet URL does not
# resolve from tf1 itself, measured 2026-09-22.
NTFY_URL = os.getenv("NTFY_URL", "http://127.0.0.1:8090")
STATE = Path(os.getenv("LEADS_STATE", "/state/leads-notify.json"))
LOOKBACK_DAYS = 7
SINCE_HOURS = 48

PRIORITY_DEFAULT = 3
PRIORITY_HIGH = 4

CANARY_TAG = "[CANARY]"


def load_state(path: Path = STATE) -> set[str] | None:
    """Sent keys, or None when there is no state yet."""
    if not path.exists():
        return None
    return set(json.loads(path.read_text()).get("sent", []))


def save_state(sent: set[str], path: Path = STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sent": sorted(sent)}))
    tmp.replace(path)


def _read(con: duckdb.DuckDBPyConnection, kind: str, days: int) -> pd.DataFrame:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date()
    glob = f"r2://{BUCKET}/{LEADS_PREFIX}/kind={kind}/**/*.parquet"
    try:
        return con.sql(
            f"SELECT * FROM read_parquet('{glob}', union_by_name=true, hive_partitioning=true) "
            f"WHERE dt >= DATE '{since}'"
        ).df()
    except duckdb.Error as exc:
        if "No files found" in str(exc):
            return pd.DataFrame()
        raise


def verdict_key(row) -> str:
    return f"{row['run_id']}:{row['candidate_id']}"


def pending(
    verdicts: pd.DataFrame,
    expected: pd.DataFrame,
    sent: set[str] | None,
    now: datetime,
    since_hours: int = SINCE_HOURS,
) -> list[dict]:
    """Every alert not yet sent, oldest first."""
    out: list[dict] = []
    if not verdicts.empty:
        v = verdicts
        if "mode" in v.columns:
            v = v[v["mode"] == "production"]
        if sent is None:
            cutoff = now - timedelta(hours=since_hours)
            v = v[pd.to_datetime(v["adjudicated_at"], utc=True) >= cutoff]
        canary = v["canary_id"].notna() if "canary_id" in v.columns else False
        v = v[v["worth_a_look"].fillna(False).astype(bool) | canary]
        for _, row in v.sort_values("adjudicated_at").iterrows():
            key = verdict_key(row)
            if sent is None or key not in sent:
                out.append({"key": key, **message(row)})

    if not expected.empty:
        seen = (set(verdicts["canary_id"].dropna().astype(str))
                if not verdicts.empty and "canary_id" in verdicts.columns else set())
        due = expected[pd.to_datetime(expected["deadline"], utc=True) < now]
        for _, row in due.iterrows():
            cid = str(row["canary_id"])
            key = f"missing:{cid}"
            if cid in seen or (sent is not None and key in sent):
                continue
            out.append({"key": key, **missing_message(row)})
    return out


def _clean(text) -> str:
    """ntfy headers are latin-1; titles are kept to plain ASCII."""
    return str(text).encode("ascii", "replace").decode("ascii")


def message(row) -> dict:
    canary = row.get("canary_id")
    canary = None if canary is None or pd.isna(canary) else str(canary)
    kind = row.get("cluster_type") or "unclear"
    what = (f"#{row['label']}" if row.get("source") == "trend" and row.get("label")
            else f"{int(row['size'])} accounts" if pd.notna(row.get("size")) else "group")
    title = f"Kenya lead: {kind} ({row.get('status')} {row.get('source')}, {what})"
    if canary:
        title = f"{CANARY_TAG} {canary}: {title}"

    lines = [
        f"kenya_relevant={row.get('kenya_relevant')} confidence={row.get('confidence')}",
        str(row.get("rationale") or ""),
        f"Would change: {row.get('what_would_change_this') or '-'}",
        _match_line(row),
        f"Dossier: r2://{BUCKET}/{row.get('dossier_key')}" if row.get("dossier_key") else "Dossier: none",
        f"Show: uv run kma-leads show {row['run_id']} {row['candidate_id']}",
    ]
    if canary:
        lines.insert(0, f"Canary {canary} reached the reader. worth_a_look={row.get('worth_a_look')}")
    priority = PRIORITY_HIGH if kind == "influence_operation" and not canary else PRIORITY_DEFAULT
    return {"title": _clean(title), "body": "\n".join(lines), "priority": priority}


def _match_line(row) -> str:
    if row.get("source") == "trend":
        return f"Match: {row.get('status')} trend tag"
    if row.get("status") == "new":
        return "Match: new, no predecessor above the Jaccard cut"
    j, prev, joined = row.get("jaccard"), row.get("previous_size"), row.get("joined")
    return (f"Match: changed, Jaccard {j:.2f} against a predecessor of {int(prev)}, "
            f"{int(joined)} joined") if pd.notna(j) and pd.notna(prev) and pd.notna(joined) \
        else f"Match: {row.get('status')}"


def missing_message(row) -> dict:
    cid = str(row["canary_id"])
    return {
        "title": _clean(f"{CANARY_TAG} MISSING {cid}"),
        "body": (f"Canary {cid} injected {row.get('injected_at')} had no verdict by "
                 f"{row.get('deadline')}. Some stage between detection and this poller "
                 "is not running: check the daily v2 run, the leads pass, then this service."),
        "priority": PRIORITY_HIGH,
    }


def send(alert: dict, token: str, url: str = NTFY_URL, topic: str = TOPIC) -> None:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/{topic}",
        data=alert["body"].encode("utf-8"),
        method="POST",
        headers={
            "Title": alert["title"],
            "Priority": str(alert["priority"]),
            "Authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"ntfy returned {resp.status}")


def once(con: duckdb.DuckDBPyConnection, *, dry_run: bool, since_hours: int = SINCE_HOURS,
         state: Path = STATE) -> list[dict]:
    now = datetime.now(timezone.utc)
    sent = load_state(state)
    alerts = pending(_read(con, "verdicts", LOOKBACK_DAYS),
                     _read(con, "canary_expected", LOOKBACK_DAYS), sent, now, since_hours)
    if dry_run:
        for a in alerts:
            print(f"--- would send (priority {a['priority']}) to {TOPIC}\n{a['title']}\n{a['body']}\n")
        print(f"{len(alerts)} alert(s); state {'absent' if sent is None else f'{len(sent)} sent'}")
        return alerts
    token = os.environ.get("NTFY_TOKEN")
    if not token:
        raise RuntimeError("NTFY_TOKEN is absent; add it to ../.env on tf1 (see docs/analysis/leads.md)")
    done = set(sent or ())
    for a in alerts:
        send(a, token)
        done.add(a["key"])
        save_state(done, state)
        log.info("sent %s", a["key"])
    if sent is None:
        # Record an empty state even when nothing was sent, so the cutoff for a
        # fresh host applies once and not on every pass.
        save_state(done, state)
    return alerts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Post new leads from R2 to ntfy.")
    ap.add_argument("--dry-run", action="store_true", help="print alerts; send nothing, record nothing")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=600, help="seconds between passes with --loop")
    ap.add_argument("--since-hours", type=int, default=SINCE_HOURS,
                    help="with no state file, only verdicts this recent are considered")
    args = ap.parse_args()

    con = connect()
    while True:
        try:
            once(con, dry_run=args.dry_run, since_hours=args.since_hours)
        except Exception:
            if not args.loop:
                raise
            log.exception("leads notify pass failed")
        if not args.loop:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
