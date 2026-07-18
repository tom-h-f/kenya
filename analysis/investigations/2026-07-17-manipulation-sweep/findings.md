# Manipulation Sweep Findings - 2026-07-17

Corpus: ~102k latest posts, ~48k dated authors, X/Twitter only, capture window
approx. Jul 2026. All artifacts re-runnable: `uv run python
investigations/2026-07-17-manipulation-sweep/NN_x.py --full` from `analysis/`.
Coordination layers were computed fresh (R2 `coordination/` had no persisted
runs) and cached locally in `out/00_coordination_*.parquet`; nothing was
persisted to R2 - persisting would feed the collector's adaptive targeting,
which this sweep deliberately avoided pending a decision.

Every claim below is triage material for human review, not a verdict. See
caveats footer.

## F1. Ol Kalou by-election astroturf messaging (high confidence)

**Claim:** A set of low-follower accounts pushed synchronized, PR-styled
praise for by-election candidate "Samuel Muchina Nyaga / Muchina Wa UDA"
in near-simultaneous bursts.

**Evidence** (`03_paste_bursts.csv`, `03_components.csv`):
- component 36686: 4 authors posting near-duplicate candidate advocacy inside
  0.8 minutes (seed `naledifusion`, 15.5k followers).
- component 14709: 8 posts from 4 authors inside 3.0 minutes praising the
  candidate (seed `KinyoiKe`).
- Member texts read as campaign copy ("I support Nyandarua Youth League...",
  "Muchina Wa UDA kept the discussion centred on practical development"),
  repeated with entity swaps across accounts.
- Same milieu shows a 5-author echo of "Kiambu engagements reinforce...
  accessible leadership" seeded by `theweeknd6699` (component 1698, 33 min).

**Alternatives:** legitimate paid comms by a campaign (likely - but
undisclosed coordinated posting is exactly what this monitor triages);
journalists quoting the same speech (does not fit the sub-minute spread of
non-news accounts).

## F2. Voter-register doubt narrative, fringe-to-mainstream (medium-high)

**Claim:** A narrative that IEBC voter details are missing/deleted circulated
as an entity-swapped Swahili template and separately jumped from a
62-follower account to national media within 2 hours.

**Evidence:**
- `03_templates.csv`: skeleton "ebu pitieni huko <u> mwangalie kama your voter
  details ziko. naona iebc wameanua zangu!" - 5 distinct texts, 5 authors
  (`Gitz__`, `vybzmartel_`, `GrieyKing`, `Zephrteve`, `nursemoraa`).
- `06_story_launder.csv` story 635: "@IEBCKenya Why are voter details not
  showing... WANTAM" seeded by `JakeReaper3` (62 followers), carried by
  `citizentvkenya` 1.8h later.
- Two independent lenses (template detection, story laundering) hit the same
  narrative.

**Why it matters:** voter-roll distrust is a classic election-manipulation
lever; whether the details are truly missing is checkable against IEBC.
**Alternatives:** genuine users hitting a real IEBC portal bug and copying a
viral phrasing organically. Absence of corroboration is not falsity - this
needs a human check of the IEBC portal claim.

## F3. Reply-brigade workers inside coordination clusters (high)

**Claim:** A set of accounts is habitually among the first repliers across
many principals' threads, and every one of the top ten sits in an
SVN-validated coordination cluster.

**Evidence** (`04_fast_repliers.csv`):
- Uniform-null thread-rank test: `senator047` mean rank pct 0.18 over 21
  threads, z = -5.1, median lag 3.3 min across 11 distinct targets;
  `comogolla` z = -4.4; `Victor35167777` z = -4.2; all 10 most-extreme
  accounts have `in_coord_cluster = True`.
- `senator047` reply texts are uniform hostile one-liners ("Taka taka nkt",
  "KK government very clueless") aimed at commentators.
- Follow-graph corroboration (`08_follow_density.csv`): coordination-cluster
  members hold 35 internal follow edges vs 6.1 expected under random
  crawled-set subsets (p ~ 0.004, 10 mutual pairs).

**Alternatives:** superfans with notifications on. That explains one target,
less well 10+ targets with coordinated-cluster co-membership.

## F4. "84.1% election violence" scare stat: coordinated seeding + engineered
engagement (upgraded medium-high, re-evaluated 2026-07-18)

**Claim:** A suspiciously precise election-violence statistic ("84.1%
probability", attributed to the Kofi Annan Foundation) was seeded by multiple
accounts inside one hour, amplified by a rapid RT burst, and shows a
bought-engagement like curve.

**Evidence:**
- Origin trace (posts with "84.1%"): `Mike_Kutola` 04:00 UTC+1 2026-07-09
  ("Kofi Annan Foundation has warned...") -> `moneyacademyKE` variant 04:36
  -> 11+ RTs within 17 minutes (04:37-04:53) -> `BravinYuri` near-identical
  re-post 04:52. Multiple "original" phrasings of the same stat within the
  hour is a seeding pattern, not organic pickup.
- `05_step_jumps.csv`: the `moneyacademyKE` post gained 92.9% of its ~1,000
  likes in one snapshot interval (36 snapshots / 120h); the account holds 3
  of 11 step-jump curves in the tracked set.
- Still circulating a week later via `SokoAnalyst` RTs (in `10_nli_tail.csv`,
  violence_call 0.995).
- The attribution is checkable: whether the Kofi Annan Foundation published
  any such 84.1% figure is a concrete fact-check task. Decimal-precision
  probabilities for political violence are not how such foundations report.

**Alternatives:** a real report could exist and be quoted in parallel from a
newsletter (would explain multi-account near-simultaneity); step curve could
compress a genuine surge across a snapshot gap. Fact-check before naming.

## F5. Dogpiles and negative brigading (medium)

**Evidence** (`04_dogpiles.csv`, `04_targets.csv`):
- `NjiruAdv` parent post: 21 repliers, 52% under 100 followers, 67% negative
  sentiment, p90 lag 61 min.
- Supportive rapid-response also visible: two `WilliamsRuto` posts drew 27-37
  repliers with only ~24-26% negative - fast positive amplification.
- Highest sustained negative pressure: `Kenyans` (64% of 727 replies
  negative), `citizentvkenya` (64%), `rigathi` (57%).

## F6. Recent-account hashtag cohort (low-medium)

**Evidence** (`02_hashtag_cohorts.csv`): `#beawaretimeisover` - 61 authors,
11.5% created in the last 180d vs 4.1% corpus baseline (z = 2.9), 26% of
handles ending in 4+ digits. Weak alone; the tag's authors deserve a read.
Cluster-level creation compactness produced nothing surviving multiplicity
(best p = 0.04 across 84 clusters, `02_cluster_birth.csv`).

## F7. Awakened accounts (low, corroborative)

`01_awakened.csv`: `joshnyongesa` - 5.8-year-old account, 451 lifetime
tweets, now posting 23/day (109x its lifetime rate) AND a coordination-cluster
member. One of only two accounts matching the pattern; the lens is starved by
the >=20-posts eligibility bar (617 accounts).

## F8. Ethnic-incitement lens (Phase B, first pass; medium-high)

**Claim:** A small but real stratum of coded incitement language exists in the
corpus, including a repeated 2007-era dehumanisation formula.

**Evidence** (`10_flagged.csv`; joint rule = lexicon hit AND NLI >= 0.85):
- "Snake and its eggs" trope, three accounts in four days: `EricMarcos254`
  "tunaua nyoka na mayai yake huku mlimani" (dehum 0.98, violence 0.90);
  `Kuilean1` "nyoka na mayai yake lazima iende"; `eliudlast` (replying in the
  same thread) "Nyoka ni gachagua na wamunyoro..." - the referent is the
  Gachagua camp.
- `SodaBoflo`: "Tukimaliza kutoa wantam, next ni kutoa madoadoa Kenya. Tuwe
  safi kama pamba" ("when we finish removing wantam, next we remove the
  madoadoa - let's be clean as cotton") - purge framing with an NCIC-flagged
  term.
- 14 flagged posts total from ~3k scored (all 50 lexicon-hit posts included);
  4 of 14 flagged authors are coordination-cluster members; 0 in the >=2-lens
  convergence set.
- Community/region aggregates of flagged authors are below or near the
  MIN_LOCATION_COVERAGE bar (21-43% coverage over 14 authors) - too thin for
  any group-level statement (TRIBE_DISCLAIMER applies).

**Method notes:** validation on 50 hits + 50 controls
(`10_validation_100.csv`) showed NLI-alone over-triggers on ethnic-bloc
horse-race commentary, hence the joint rule. Historical references (e.g.
`tonykaromo` recounting the 2008 Kwekwe squad) flag as expected and need the
human read the lens is designed for. Scoring is incremental
(`kma.incitement.score_new`, R2 `incitement/` prefix).

**Final full-coverage re-evaluation (2026-07-18, 109,057 posts scored -
entire corpus):** lexicon hits 50 -> 52 (two new posts since collection
continued; neither clears the joint NLI bar), flagged set stays at 14 - the
joint-rule findings are stable and now census-complete for the capture. The
NLI-only tail grew 488 -> 1,568 rows and sharpened the most
election-dangerous pattern in the corpus: **two-directional ethnic
fear-priming around Ol Kalou**. One direction: `GlitteringAnge` "Ruto is
sending other tribes to hate and attack kikuyus" (RT'd by `edunjau1`,
`skinyanjui86`, ...), `MwangiHub` "Ruto has declared war with the people of
OlKalou", `AnuarSaddat` "ready to maim and kill Kikuyus". Other direction:
`Simonk2341` blaming "Luo-aligned ele[ments]" around the president,
`Manu8675` accusing Moses Kuria of "inciting mt. Kenya residents... to
possibly kill their legislat[or]". Each side tells its community the other is
coming for them - the classic pre-violence priming structure, running through
accusation and counter-accusation rather than open slurs (which is why the
lexicon alone cannot see it). Mainstream reporting of the same events
(citizentvkenya, TheStarKenya) flags in the tail too - expected FPs to read
past.

**Earlier partial re-evaluation at 33k scored:** the lexicon+NLI flagged set is
unchanged (lexicon hits were priority-scored first, so wider NLI coverage
cannot add joint-rule flags). What the wider coverage added is the NLI-only
extreme tail (`10_nli_tail.csv`: dehum >= 0.9 AND violence >= 0.9 AND
political-criticism <= 0.4, no lexicon hit): 488 rows, 304 distinct texts,
dominated by the Ol Kalou "goons" narrative war - RT waves of the
extortion-rings accusation against Gachagua (37 RTs), the elders-curse
response (34), Babu Owino's warning to "goons" (27), and the Kisumu
"mbogi... na mishale" armed-gang accusation (11). Most is violence-adjacent
accusation/reporting, not incitement - but `AnuarSaddat`'s "Ruto has sent
the entire government might to Ol Kalou, ready to maim and kill Kikuyus"
(6 RTs) is explicitly ethnicised violence anticipation and belongs in the
triage queue. Lexicon-expansion candidates surfaced by the tail: "mbogi"
(Sheng: gang), "mishale" (arrows), "goons" - all left OUT of the lexicon for
now because each is common in innocent/reporting use and the lexicon is
documented-sources-only; they are noted here for the human lexicon review.

## Convergence ranking

`09_convergence.csv`: 31 accounts flagged by >=2 independent lenses; only
`senator047` hits 3 (cluster + fast-replier + suspicion >= 0.5). The 2-lens
set is dominated by cluster-membership + high-suspicion pairs - mostly
near-zero-follower reply accounts (`legend_kir`, `nja6m`, `eliis5xe`, ...).

## Negative results / dead lenses (worth knowing)

- **text_sim and fast_co_share SVN channels validated 0 edges** on the full
  corpus (00 run) - copypasta exists (F1) but below SVN significance under
  the degree-corrected null; the ordering-based 03 lens is the sensitive one.
- **Platform `lang` is useless for Swahili here:** ~0% of posts labelled
  `sw`; 7.7% of "en" posts carry Swahili function words (07 calibration).
  Bilingual-framing analysis needs own language ID (Phase B).
- **Shift cohorts / circadian clustering:** no cohorts of >=3 accounts with
  near-identical distinctive profiles (01); capture window still short.
- **Ratio outliers (05)** are polluted by global viral content retweeted into
  the corpus (England, Dexerto, K-pop) - lens needs a Kenya-relevance filter.
- **Handle "families" (02)** are overwhelmingly X default handles
  (name+digits, multi-year creation spans) - not batch registration.
- One spam network caught as a control: 29 UK accountancy accounts pasting
  identical tax content inside 31 min (component 450) - the detector works.

## Caveats (verbatim from the analysis package)

- "Capture is a sample, not a census: the earliest collected post is not
  necessarily patient-zero, and spread (retweeters / repliers) is bounded by
  what the snowball census reached." (stories.SAMPLING_CAVEAT)
- "A corroboration gap is a triage flag, NOT proof of falsity. Trusted
  outlets lag breaking news and often tweet only headlines, so a real story
  can show a gap for hours. Always read the nearest trusted post before
  judging - and note that fact-checker coverage (PesaCheck / AfricaCheck) is
  weak until their timelines backfill." (stories.STORY_CAVEAT)
- "Capture is a sample, not a census: absence of a co-action is not evidence
  of absence. Recall is bounded; precision is not affected."
  (coordination.SAMPLING_CAVEAT)
- Suspicion scores are triage signals, not bot labels. Coordination and
  cohort structure are probabilistic evidence of similarity, not proof of
  malice or inauthenticity. Absence of corroboration is not evidence a claim
  is false.
