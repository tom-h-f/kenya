# Opus v4 follow-up: mix, threshold, errors

**Goal:** Recover 2013 unanimous without killing Opus challenge gain; tune
hate thresholds for deployment; mine hard misses against validated Opus labels.

**Assumption (operator 2026-07-24):** all 1,500 Opus v4 labels are human-validated.

**Order:**
1. Update `STATE.md` caveat (validated Opus).
2. Error analysis (4): download r3 error CSVs; score 14 known coded posts;
   write hard-miss pack (no label edits unless a clear train bug appears).
3. Threshold (3): full hate sweep (`11_hate_sweep.py`) on baseline + best r3
   seed for `val2026_opus-v4`, `challenge_opus-v4`, `test_unanimous`.
4. Mix sweep (2): Modal train `extra-repeat` Opus ∈ {1,2,3,5}, 3 seeds each
   (or 1 seed smoke then 3 for winners). Prefer 1 seed first for all four
   repeats, then 3-seed the Pareto candidates.

**Promote only if:** unan within ~2.1pt of shipped **and** challenge ≥ r3-×5
mean (0.610) within noise, or clear operator-accepted tradeoff.
