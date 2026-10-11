"""What a lexical Tanzania guard would change, on the labelled sample and community 8's member posts.

The guard: a post matching a Tanzania-specific term and no `measure._KENYA_RE` anchor is not Kenyan.
Circular on the keyword-selected half of the sample by construction; the c8_member half is the
independent test.
"""
import re

import pandas as pd

from kma import measure

TZ = re.compile(r"tanzaniamassacre|tundu\s*lissu|\blissu\b|samia\s*suluhu|samiamuuaji|chadema|\bmo29\b|"
                r"updates_kesi_ya_uhaini|katibampya|freetundulissu|magufuli|\bdodoma\b|oktoba\s*29|"
                r"bunge\s+la\s+tanzania|tanzania|watanzania|ccm", re.I)
k = pd.read_parquet("out/tz_labelled.parquet")
k["kw"] = k.text.map(lambda t: bool(TZ.search(t)) and not measure._KENYA_RE.search(t))
k["model"] = k.p >= 0.5
k["guard"] = k.model & ~k.kw
for src, g in k.groupby("source"):
    t, kk = g[g.label == "tanzania_only"], g[g.label == "kenya"]
    print(src, "tz_only", len(t), "model FP", int(t.model.sum()), "after guard", int(t.guard.sum()),
          "| kenya", len(kk), "model TP", int(kk.model.sum()), "after guard", int(kk.guard.sum()))
m = pd.read_parquet("out/c8_member_posts.parquet").dropna(subset=["p"])
m["kw"] = m.text.map(lambda t: bool(TZ.search(t)) and not measure._KENYA_RE.search(t))
print("c8 member posts scored", len(m), "model share", round((m.p >= .5).mean(), 3),
      "after guard", round(((m.p >= .5) & ~m.kw).mean(), 3))
