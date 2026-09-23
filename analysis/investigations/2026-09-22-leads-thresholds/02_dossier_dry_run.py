"""The whole leads pass on real R2 data with the reader stubbed out.

    cd analysis && uv run python investigations/2026-09-22-leads-thresholds/02_dossier_dry_run.py

Everything up to the model call runs for real - the diff on v1's two latest
runs, the queue, member resolution for communities and trend tags, and dossier
assembly - and each packet's evidence counts are printed, so an empty dossier
shows up here rather than as an "unclear" verdict. Nothing is persisted and
nothing is spent.
"""

from __future__ import annotations

import time

from kma import adjudicate, leads_run
from kma.db import connect

seen: dict = {}


def stub(packets, model=None):
    for p in packets:
        seen[p["cluster_id"]] = p
    return [{"cluster_id": p["cluster_id"], "cluster_type": "unclear", "kenya_relevant": True,
             "confidence": "low", "rationale": "stub", "what_would_change_this": "stub"}
            for p in packets]


def main() -> None:
    t = time.time()
    s = leads_run.execute(connect(), source="v1", limit=5, judge=stub, persist=False,
                          mode=leads_run.MODE_VALIDATION)
    print(f"{time.time() - t:.0f}s; notes {s['notes']}; skipped {s['skipped']}")
    for v in s.get("verdicts", []):
        print(f"  {v['candidate_id']:<24} {v['source']:<9} {v['status']:<7} size {v['size']} "
              f"jaccard {v['jaccard']} worth {v['worth_a_look']}")
    print()
    for cid, p in seen.items():
        print(f"  dossier {cid}: size {p['size']} objects {len(p['shared_objects'])} "
              f"texts {len(p['shared_texts'])} posts {len(p['representative_posts'])} "
              f"targets {len(p['amplification_targets'])} provenance {bool(p.get('provenance'))} "
              f"kenya {p.get('kenya')}")
    trend = [p for c, p in seen.items() if c >= leads_run.TREND_ID_BASE]
    if trend:
        print("\n--- trend dossier as the reader sees it ---\n" + adjudicate.render(trend[0])[:2000])


if __name__ == "__main__":
    main()
