# Opus v4 relabelling design

## Goal

Relabel the 2,440-post 2026 training batch with Claude Opus using a more
complete, reproducible account of Kenyan political language and the 2027
election context. The new run replaces neither the original labels nor their
provenance.

## Labelling approach

Use Claude Opus 4.6 Thinking as a single labeller and record its exact model
and prompt version. Run under a new tag so all prior Gemini, Cursor, Sonnet,
and Opus outputs remain intact.

Prompt v4 keeps the calibrated three-class taxonomy from v3:

- `hate` requires an attack on an identifiable protected group, or on a person
  because of protected-group membership.
- `offensive` covers abuse, degradation, or threats without that target.
- `neither` covers criticism, reporting, quotation, condemnation, and neutral
  electoral analysis.

## Static Kenyan context

The prompt will add versioned, neutral reference sections rather than
per-post retrieval:

1. The 2027 election cycle, relevant institutions, major political actors,
   parties, coalitions, protests, and common regional references.
2. Kenyan English, Kiswahili, Sheng, and multilingual code-switching guidance.
3. Political aliases and slogans, including `Kasongo` and `Zakayo` as common
   references to William Ruto, `Sugoi`, `Wantam`, and `Tutam`.
4. Ethnic, regional, and coded-speech terms already observed in the corpus,
   with concise explanations of their possible literal and harmful uses.
5. Difficult examples covering sarcasm, political aliases, quoted speech,
   ethnic electoral analysis, personal abuse, coded threats, and collective
   contempt.

An alias identifies an actor; it does not make that actor or their supporters
a protected group. No glossary entry is automatic evidence of hate,
offensiveness, intent, or a target. The supplied post remains the evidence,
and ambiguous context lowers confidence rather than licensing an inference.

## Data flow

1. Build prompt v4 as a separate prompt asset.
2. Add tests for prompt selection, Opus command construction, response
   validation, and output isolation.
3. Run Opus v4 against the 120-row calibrated set and produce comparison
   metrics. This is a regression check, not proof of independent accuracy.
4. Run a stratified pilot from the 2,440 posts and inspect structure, class
   distribution, flags, rationales, and representative disagreements.
5. If transport and semantic checks are sound, resume the same tagged run
   across all 2,440 posts.
6. Merge the outputs into a new labelled dataset and report class and flag
   distributions plus movement from the existing labels.

Chunks remain resumable and idempotent. Invalid JSON, missing or extra post
IDs, invalid enum values, and failed subprocesses are retried and then parked
without contaminating completed chunks.

## Validation and limitations

The existing 93-row heldout was Opus-prelabelled before human validation, so
it cannot independently measure Opus accuracy. The 120-row calibrated set and
a human pilot review can detect regressions and prompt mistakes, but accepted
single-labeller output still carries correlated model error.

Before the full run, verify:

- every expected ID appears exactly once;
- labels, flags, target groups, confidence, and rationale satisfy the contract;
- `hate` and protected-group targeting remain consistent;
- political aliases are not treated as ethnic targets;
- quoted or condemned speech is not attributed to the quoting author;
- no existing labelled artifact is overwritten.

## Deliverables

- `prompts/label_v4.md`
- tests and any minimal runner/reporting changes
- a versioned Opus v4 pilot
- a complete Opus v4 label set for 2,440 posts
- merged parquet and JSON report with provenance and label movement
