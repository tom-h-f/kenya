"""Phase 4 smoke test - run each stage against live R2 and print what to eyeball.

    cd analysis && uv run python verify_stories.py

Needs: working analysis env (uv sync), R2 creds in ../.env, and embeddings for
recent posts INCLUDING the trusted outlets (run `uv run python -m kma.enrich`
first if [1] shows few/none).
"""

import pandas as pd

from kma import stories as st
from kma.db import connect, embeddings_source

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 200)

DAYS = 14  # widen to reach older stories; narrow (7) for just the fresh window
con = connect()

# [1] Are embeddings present, and are the trusted outlets among them?
n = con.sql(f"SELECT count(*) FROM {embeddings_source('x')}").fetchone()[0]
trusted = st._trusted_posts(con, DAYS, "x", st.MODEL)
seen = sorted(trusted["author_handle"].str.lower().unique()) if len(trusted) else []
print(f"[1] embeddings in R2: {n:,}")
print(f"    trusted-outlet posts embedded (last {DAYS}d): {len(trusted)}")
print(f"    trusted handles present: {seen or 'NONE - corroboration will be blind'}")

# [2] Candidate stories (the clustering step)
s = st.candidate_stories(con, days=DAYS, tau=st.DEFAULT_TAU, min_size=st.DEFAULT_MIN_SIZE)
print(f"\n[2] {s['story_id'].nunique()} stories / {len(s)} member posts "
      f"(tau={st.DEFAULT_TAU}, min_size={st.DEFAULT_MIN_SIZE})")
for sid, g in s.groupby("story_id"):
    print(f"    story {sid}: {g['author_id'].nunique()} authors, {len(g)} posts "
          f"| e.g. {g['text'].iloc[0][:80]!r}")

if s.empty:
    raise SystemExit("no stories - widen DAYS, lower min_size, or embed more posts")

# [3] Corroboration (the novel signal) - eyeball BOTH directions
c = st.corroboration(con, s, days=DAYS)
print("\n[3] corroboration (high = trusted media echoes it; low = gap):")
print(c[["story_id", "corrob_sim", "nearest_handle"]].round(3).to_string(index=False))

# [4] Scorecard - the ranked triage table
cards = st.story_scorecard(con, s, c)
print("\n[4] ranked scorecard:")
cols = ["story_id", "size", "n_posts", "corrob_sim", "corroboration_gap",
        "amplifier_botness", "coordination_overlap", "story_suspicion_index"]
print(cards[[x for x in cols if x in cards.columns]].round(3).to_string(index=False))
print("    top story keywords:", cards.iloc[0]["keywords"])
print("    top story text    :", cards.iloc[0]["representative_text"][:120])

# [5] Origin + spread of the most-suspicious story
top = int(cards.iloc[0]["story_id"])
story = s[s["story_id"] == top]
print(f"\n[5] origin of story {top} (earliest COLLECTED != patient-zero):")
print(st.origin(con, story)[
    ["created_at", "author_handle", "suspicion", "in_coordination_cluster"]
].to_string(index=False))
sp = st.spread(con, story)
print(f"    amplifiers reached: {len(sp['amplifiers'])} | timeline points: {len(sp['timeline'])}")

# [6] Persist one run (uncomment to actually write to R2 stories/ and feed the collector)
# key = st.persist_stories(con, cards, min_index=0.6)
# print("\n[6] wrote:", key)
