"""Ground-truth eval - does the story pipeline surface the known disinfo cases?

    cd analysis && uv run python run_eval.py

Runs each labeled case in kma.eval.GROUND_TRUTH through the live pipeline and prints
a red/green report with the first drop-out stage. Needs a working analysis env
(uv sync), R2 creds in ../.env, and recent posts embedded (kma.enrich).
"""

import pandas as pd

from kma import eval as gt
from kma import stories as st
from kma.db import connect

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

DAYS = 14
con = connect()

report = gt.evaluate(con, days=DAYS, tau=st.DEFAULT_TAU, min_size=st.DEFAULT_MIN_SIZE)

cols = ["case", "expect", "n_present", "n_embedded", "story_id", "story_authors",
        "is_blob", "rank", "triage_cut", "n_stories", "suspicion", "surfaced",
        "pass", "drop_stage"]
print(f"\n=== ground-truth report (days={DAYS}, tau={st.DEFAULT_TAU}, "
      f"min_size={st.DEFAULT_MIN_SIZE}, triage_fraction={gt.TOP_FRACTION}) ===")
print(report[cols].to_string(index=False))

required = report[report["expect"] == gt.EXPECT_SURFACE]
n_pass = int(required["pass"].sum())
print(f"\n{n_pass}/{len(required)} required (surface) cases passed. "
      f"blob_authors_cut={gt.BLOB_AUTHORS}")
for _, r in report.iterrows():
    case = next(c for c in gt.GROUND_TRUTH if c.name == r["case"])
    if r["expect"] == gt.EXPECT_KNOWN_LIMITATION:
        state = "surfaced" if r["surfaced"] else f"not surfaced (drops at {r['drop_stage']})"
        print(f"  NOTE {r['case']}: known limitation, {state} - {case.note}")
    elif not r["pass"]:
        print(f"  RED  {r['case']}: dropped at {r['drop_stage']} - {case.note}")
