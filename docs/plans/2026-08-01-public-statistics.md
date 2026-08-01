# Public statistics and visualisation: document of intention

**Status: intention only.** Nothing here is built. This records what we want to
publish, how each number would be defined, and which of them we can already
compute. It is deliberately opinionated about what we will *not* claim.

---

## 1. What this is

A public, periodically-updated statistical picture of Kenyan political discourse
ahead of the 2027 election, with a focus no one else has: **how much of the
apparent conversation is authentic**.

The FiveThirtyEight reference is about *manner*, not method - quantitative,
plainly explained, uncertainty shown rather than hidden, a consistent visual
language, and a willingness to say "we don't know". We are not building a vote
forecast off the back of a coordination graph.

**Audience:** public site plus written analysis, updated weekly and on events.
That is the highest bar: every number must survive an adversarial reader, and the
sampling rules have to be on the page rather than assumed.

### What we will not claim
- **Not a vote forecast.** Polling is tracked and shown (§6), but we do not
  publish a win probability. We have no historical Kenyan election model, no
  voter file, and no backtest.
- **Not verdicts on individuals.** Coordination and toxicity outputs are triage.
  We do not publish "this account is a bot".
- **Not prevalence outside baseline collection.** Targeted passes oversample the
  toxic tail by design; any rate that mixes them is meaningless.

---

## 2. The four headline statistics

### A. Authenticity-adjusted share of voice — the signature number

For each candidate, coalition or theme, in a given window:

- **Share of voice**: share of baseline-scoped posts mentioning the entity
- **Coordination-attributed share**: what fraction of the *amplification* of those
  posts comes from accounts in validated coordination clusters

> "Gachagua holds 34% of election discourse this week. 11% of the amplification
> behind it traces to accounts in validated coordination clusters, against a
> 4% platform-wide baseline."

Nothing else in this space publishes the second number. It is the whole point.

| | |
|---|---|
| **Inputs** | `posts/` (baseline scope), `engagements/`, `coordination/kind=clusters` |
| **Buildable now?** | Yes, with one caveat below |
| **Denominator** | `latest_posts(scope="baseline")`, first-seen type |

**The caveat that must ship with it.** Coordination can only be attributed for
objects whose retweeters we censused. Coverage is currently partial and improving
(we recently raised the census cap ~16x). Until coverage is high, the coordinated
share is a **lower bound**, and must be labelled as one. Publishing it as a point
estimate would overstate our knowledge.

**Visual form:** stacked share-of-voice over time, with the coordinated portion
in a distinct hatch/tone, plus a coverage indicator showing what fraction of
amplification we could actually attribute.

---

### B. Incitement and toxicity trajectory

Baseline-scoped prevalence over time of:
- **explicit toxicity** — model `offensive`/`hate` or `p_hate >= 0.28`, Kenya-scoped
- **coded incitement** — documented NCIC/PeaceTech lexicon hit corroborated by NLI

Two series, deliberately separate, because the classifier is good at the first and
poor at the second - on 14 known-coded posts its mean `p_hate` *dropped*. Merging
them into one "hate" line would hide that.

| | |
|---|---|
| **Inputs** | `hatespeech/` (with persisted measure columns), `incitement/` |
| **Buildable now?** | Yes - largely exists in `notebooks/hatespeech.py` |
| **Watch** | days with n<20 dropped; off-domain excluded; 7-day rolling mean |

**Visual form:** dual-line trend to the election with a rolling mean, plus an
hour×weekday heatmap for surge timing.

---

### C. Coordination weather report

A recurring, readable state of the network:
- how much of this week's amplification is coordinated
- which clusters are active, how large, how corroborated
- what each cluster is pushing (c-TF-IDF cluster names)
- new clusters vs persistent ones

| | |
|---|---|
| **Inputs** | `coordination/` edges + clusters, `authenticity` scorecards |
| **Buildable now?** | Yes - `notebooks/coordination.py` has the machinery |

**The honesty rule.** Of 298 clusters, 14 are corroborated across both channels.
Single-channel clusters are *candidates*. The published view must separate them
visually, not blend them into one count - otherwise we publish 298 "networks"
when we can defend 14.

**Visual form:** network graph for the corroborated set, plus a weekly
"coordination share" line and a cluster table with size, corroboration and theme.

---

### D. Ethnic targeting index — highest value, highest risk

Which communities are being targeted, by what rhetoric, trending over time. This
is the metric with real violence-prevention value, and the one most capable of
causing harm if we get it wrong.

**Design constraint that changes the whole approach:** it must measure the
**target named in the text**, not the author's origin. The existing
region/community lens in `deltas.py` is an author-origin proxy with a loud
disclaimer attached - useful for exploration, *invalid* for a targeting claim.
"Posts by people from region X" is a different statement from "posts attacking
community X", and conflating them is exactly the error that would make this
inflammatory.

| | |
|---|---|
| **Inputs** | text-level target extraction (`target_group` exists in the 1,500 Opus labels), `incitement` categories |
| **Buildable now?** | **No.** Needs a target-extraction pass we have not built |
| **Prerequisite** | a validated way to extract the targeted group from post text |

**Guards we would commit to before publishing any of it:**
- report *categories of rhetoric* (dehumanisation, expulsion, veiled threat)
  against communities, never individuals
- publish counts and trends, never example posts naming a community
- an explicit uncertainty band, and suppression below a volume floor
- a written statement that this measures *speech observed on X*, which is not
  Kenyan opinion

Recommended sequencing: build it last, publish it only once A-C have been public
long enough to establish the methodology's credibility.

---

## 3. Cross-cutting methodology rules

These apply to every chart and must be visible on the site, not buried:

1. **Baseline scoping.** Every rate uses `scope="baseline"` on first-seen
   partition, with the timeline-leak correction applied
   (`db.effective_type_expr`). Publish `db.leak_corrected()` - an uncorrected
   figure carries a known 6.7% contamination from promoted-account timelines
   that landed in a baseline partition before 2026-08-01.
2. **Coverage is not census.** We sample X; absence of evidence is not evidence of
   absence. Recall is bounded, precision is not affected.
3. **Uncertainty shown.** Wilson intervals on proportions; volume floors with
   suppression below them; no trend line on fewer than N days.
4. **The 14-day horizon.** X search reaches back 14 days, so our longitudinal
   series begins when collection began (July 2026) and cannot be backfilled
   earlier. Say so on the axis.
5. **Model limits stated inline.** The hate classifier's coded-register blind spot
   is a published caveat, not a footnote.

---

## 4. Visual language

One consistent system, decided once:
- severity colour follows the class, never its rank
- coordinated vs organic uses tone/hatch, not hue, so it survives colourblindness
  and greyscale reproduction
- every chart carries n, window, and scope in the subtitle
- suppressed cells look suppressed, not empty

---

## 5. Polling tracker

Polling is tracked and displayed as its own layer - not fused into a forecast.

**Sources to ingest:** TIFA, Infotrak, Ipsos Kenya, Radio Africa/Star, plus
university and party-commissioned polls where methodology is published.

**What we would build:**
- a poll database: pollster, field dates, n, mode, sponsor, published toplines
- a simple average with recency weighting and pollster house-effect estimates
- sponsor disclosure shown prominently - party-commissioned polls flagged

**The question worth asking**, and the one that justifies pairing the two layers:
*does coordinated amplification lead, lag, or track polling movement?* We can put
share of voice, coordinated share and polling on one time axis and let the reader
see. With enough cycles it becomes testable rather than suggestive.

Honest position: with a handful of Kenyan polls per cycle and no backtest, we
report the relationship, we do not model it.

---

## 6. Build order

| phase | what | status |
|---|---|---|
| 1 | Toxicity trajectory (B) | mostly exists; needs baseline-scoped rebuild |
| 2 | Coordination weather (C) | machinery exists; needs a recurring public view |
| 3 | Authenticity-adjusted SoV (A) | needs entity tagging + census coverage to rise |
| 4 | Polling tracker | needs poll ingestion, new |
| 5 | Ethnic targeting index (D) | needs target extraction; publish last |

Phases 1-2 are largely presentation work over existing analysis. Phase 3 is the
distinctive one and depends on census coverage continuing to climb. Phase 5 needs
genuine new NLP.

---

## 7. Open questions

1. **Entity tagging for share of voice.** Candidate/coalition mention detection is
   not built. Simple alias matching, or something more robust to Sheng and
   misspelling? Affects A directly.
2. **Publication granularity for coordination.** Do we name clusters publicly
   (sizes, themes, and cluster IDs) while never naming member accounts? That is
   my recommendation, but it is a call with real consequences.
3. **Update cadence vs stability.** Weekly is readable, but coordination clusters
   move; a cluster present one week and gone the next may be a detection artefact
   rather than a real change. Needs a stability measure before publishing "new
   clusters this week".
4. **Who reviews before publication?** An ethnic targeting index in an election
   year warrants outside review - NCIC, a Kenyan civil-society partner, or an
   academic collaborator - before it goes public.
5. **Does the site show its own coverage?** I think it must: a coverage panel
   (what fraction of amplification we could attribute, how many accounts we track)
   is what separates an honest observatory from a confident-sounding one.
