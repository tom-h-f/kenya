# 00 — Design (locked decisions)

**Goal:** Record product and method decisions so implementation does not re-litigate scope.

## Locked decisions

1. **Approach B:** Rebuild methods first, then journalist desk brief.
2. **Audience:** Investigator / journalist. Brief may be extensive.
3. **Questions the brief answers:**
   1. What claims are circulating?
   2. Is this corroborated? (gap ≠ false; nearest trusted always shown)
   3. Who is amplifying?
   4. How is framing shifting?
   5. Where is coordination densest on this claim?
   6. What’s new since last look?
   7. Region + community aggregate lens (instructive in Kenya 2027 context; careful)
4. **Spine:** Stories/claims. Topics, coordination, authenticity, region/community are lenses.
5. **Notebook:** `analysis/notebooks/desk_brief.py`. Phase notebooks remain method labs.
6. **Tiers:**
   - `main` — existing amp-weighted triage (`min_size`, scorecard)
   - `thin_evidence` — small / low-amp clusters with high corroboration gap; never auto-elevated to “high suspicion”
7. **Community policy:** Use location→community proxy for aggregate insight. Always show `TRIBE_DISCLAIMER`. Never attach community to an individual in desk exports.

## Non-goals

- Auto-label false / bot network / inauthentic campaign
- Individual ethnicity inference
- Blocking on Wave B co-hashtag/URL/mention density

## Tone rules (copy into notebook markdown)

- A corroboration gap is a triage flag, not proof of falsity.
- Coordination is probabilistic co-action, not proof of malice.
- Bot-likeness is a suspicion score, not a bot verdict.
- Capture is a sample, not a census (origin/spread bounded).

## Acceptance criteria

- [ ] This file committed under `docs/plans/2026-07-16-misinfo-desk-brief/`
- [ ] No unresolved product forks (approach, audience, tiers, community policy)
- [ ] `docs/analysis/README.md` will link here as Phase 5 after notebook ships (tracked in 05/docs task)
