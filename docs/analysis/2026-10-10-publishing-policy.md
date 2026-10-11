# Publishing policy

Written 2026-10-10. Supersedes `2026-09-13-publishable-statistics.md`, which stays as
the audit of why each number failed and is not edited. Decision: one public site,
`kenya.tomfr.net`, no Access gate (Tom, 2026-10-10). Design in
`docs/plans/2026-10-10-explorer.md`.

The old "Not publishable" list mixed a privacy rule, now lifted, with validity rules,
which stay. This document keeps them apart.

## Now publishable (was withheld for privacy)

- Handles, display names, bios and public profile counts of accounts in a surfaced
  community.
- Public post text and links used as evidence.
- Community membership.
- Account-to-account links, each with its evidence.
- Per-account post counts in the window.
- Hashtag and URL use, by account and by community.
- Which accounts an account reposts.

## Still publishable only as before

- Benchmark reproduction.
- Relevance gate error.
- The detector's own output, described as such.
- Negative results.
- Pipeline health.

## Stays out, for validity

- Any prevalence rate without a random-sample denominator. The control arm is the only
  denominator and it is small: 29,213 posts.
- A raw toxicity or hate series. Composition drift: replies carry 3.2x the hate rate
  and the mix moved from 70.7% search to 72.6% replies.
- A count of surfaced accounts or communities as a finding. It is a function of Leiden
  resolution, seed and a triage budget.
- Any corroboration claim.
- Any statement that a named account is inauthentic, a bot, a troll, paid, or acting
  with intent. Any inference about who runs an account.

## Stays out, for harm

- Per-account hate, incitement, toxicity, deletion or suspension flags. A strong
  accusation against a named person, from a classifier with a documented weakness on
  coded speech. Aggregate by topic only, with denominators, pending Tom.
- Bios used to characterise a group.
- Location text.
- Any cross-platform identity linking.

## Rules for what is shown

- State the basis on the same line as the number: snapshot, window, parameters,
  standardised or not, filter.
- Show the evidence behind any link.
- Timestamp everything.
- Show denominators.

## Where this is enforced

- `kma.dashboard`: `summary.json` stays aggregate. `assert_public` refuses any
  per-account key, handle, profile URL or id-sized number. Its `withheld` list is
  this document's "stays out" sections.
- Explorer bundle: schema, forbidden-field walk, verdict-word lint, denylist
  (`assert_explorer_safe`).
- Site: `tests/language.test.ts` lints all static copy. Every network, community and
  account view carries the method caveat.
