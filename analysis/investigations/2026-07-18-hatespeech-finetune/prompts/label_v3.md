# Hate/offensive labelling - Kenya 2026-27 - prompt v3

> Changed from v2 after a calibrated human review. `hate` now requires an
> identifiable protected-group target. Violence, dehumanisation, or coded
> menace without that target is `offensive`, not `hate`. The legacy flag name
> `ethnic_targeting` covers targeting of any protected group.

You are annotating Kenyan political social-media posts for an
election-monitoring dataset. The labels train a classifier that routes posts
to human analysts. Missing ethnic incitement is dangerous; treating generic
political abuse as hate destroys precision.

## Classes

Choose exactly one.

**hate** - The author attacks, dehumanises, threatens, excludes, ridicules, or
stirs contempt against:

- a protected group as a group; or
- a person because of actual or perceived membership in a protected group.

Protected characteristics include ethnicity, tribe, race, religion, national
or regional origin, gender, disability, and sexual orientation. The group may
be named or clearly implied by context, but it must be identifiable. Collective
ethnic contempt is hate even without a violence call.

**offensive** - Vulgar, abusive, insulting, degrading, or threatening language
without an identifiable protected-group target. This includes personal abuse
of politicians, generic threats, and coded or dehumanising attacks whose target
is an individual, party, institution, or unspecified enemy.

**neither** - Political criticism without abuse; reporting, quoting, or
condemning hate; electoral arithmetic; factual discussion; satire or banter
that does not attack a protected group.

## Governing boundary

Ask in order:

1. Is the author attacking or threatening anyone?
2. Is the target an identifiable protected group, or a person attacked because
   of protected-group membership?
3. If yes to both, label `hate` and set `ethnic_targeting`.
4. If only the first is yes, label `offensive`.
5. If the author is reporting or condemning the attack, label `neither`.

For this task, `hate` and `ethnic_targeting` must agree:

- every `hate` row must set `ethnic_targeting`;
- a row that sets `ethnic_targeting` must be `hate`.

Mentioning ethnicity is not automatically targeting. Voting analysis,
descriptions of discrimination, and condemnation of ethnic attacks remain
`neither`.

## Coded speech

Terms such as `madoadoa`, `kwekwe`, `sangari`, `mende`, `nyoka`,
`fumigate/fukiza`, `rudi kwao`, `fukuza`, `watajua hawajui`, `41 vs 1`,
`uthamaki`, `goons`, and `mbogi` require context. A term alone is not hate.

- Coded term + identifiable protected group + attack/harm -> `hate`.
- Coded attack or threat with no identifiable protected group -> `offensive`.
- Literal, neutral, quoted, or condemned use -> `neither`.

Do not infer an ethnic target solely because the post is Kenyan political
speech. Set confidence low when context may identify a target but the supplied
text does not.

## Flags

Set every applicable flag, independently of severity:

- `dehumanisation` - people are framed as vermin, disease, filth, demons, or
  less than human.
- `violence_call` - the author calls for, threatens, celebrates, or approves
  physical violence, including a veiled threat.
- `ethnic_targeting` - the author attacks an identifiable protected group, or
  a person because of protected-group membership. Despite the legacy name,
  this includes all protected characteristics listed above.
- `coded_language` - harmful meaning depends on euphemism, metaphor, or
  context-specific code rather than a plain statement.

Consistency:

- `hate` if and only if `ethnic_targeting` is set.
- `neither` must not set `violence_call`, `dehumanisation`, or
  `ethnic_targeting`; quoted or condemned content describes someone else's
  stance.
- An `offensive` row may set `violence_call`, `dehumanisation`, or
  `coded_language`, but not `ethnic_targeting`.

`target_group` is the named or clearly implied protected group, or null.

## Calibrated examples

Input: `{"post_id":"ex1","text":"Ruto has failed this country and must go in 2027."}`
Output: `{"post_id":"ex1","label":"neither","flags":[],"target_group":null,"confidence":"high","rationale":"Political criticism with no abuse or protected-group attack."}`

Input: `{"post_id":"ex2","text":"Ruto is a useless thief and an idiot."}`
Output: `{"post_id":"ex2","label":"offensive","flags":[],"target_group":null,"confidence":"high","rationale":"Personal abuse of a politician, not a protected-group attack."}`

Input: `{"post_id":"ex3","text":"Kikuyu from central hukuwa wajinga sana."}`
Output: `{"post_id":"ex3","label":"hate","flags":["ethnic_targeting"],"target_group":"Kikuyu","confidence":"high","rationale":"'Kikuyu from central ... wajinga' attributes stupidity collectively to an ethnic group."}`

Input: `{"post_id":"ex4","text":"The Sabaot voting bloc in Mount Elgon are very clueless."}`
Output: `{"post_id":"ex4","label":"hate","flags":["ethnic_targeting"],"target_group":"Sabaot","confidence":"high","rationale":"'Sabaot ... are very clueless' ridicules an ethnic community collectively."}`

Input: `{"post_id":"ex5","text":"Tunaua nyoka na mayai yake. Hakuna kuachilia hata mmoja."}`
Output: `{"post_id":"ex5","label":"offensive","flags":["dehumanisation","violence_call","coded_language"],"target_group":null,"confidence":"low","rationale":"Eliminationist coded threat, but the supplied text does not identify a protected-group target."}`

Input: `{"post_id":"ex6","text":"Nyoka wa Mt Kenya lazima waondoke; tunaua nyoka na mayai yake."}`
Output: `{"post_id":"ex6","label":"hate","flags":["dehumanisation","violence_call","ethnic_targeting","coded_language"],"target_group":"Mt Kenya community","confidence":"high","rationale":"Snake metaphor and elimination threat explicitly target a regional ethnic community."}`

Input: `{"post_id":"ex7","text":"Are you blind or just fucking mental? Fuck yourself."}`
Output: `{"post_id":"ex7","label":"offensive","flags":[],"target_group":null,"confidence":"high","rationale":"Vulgar personal abuse with no protected-group target."}`

Input: `{"post_id":"ex8","text":"The Luo vote is consolidating behind Raila while Western stays split."}`
Output: `{"post_id":"ex8","label":"neither","flags":[],"target_group":null,"confidence":"high","rationale":"Ethnic-bloc electoral analysis without contempt or attack."}`

Input: `{"post_id":"ex9","text":"Manyora says pushing Sifuna out makes ODM look owned by the Luo community."}`
Output: `{"post_id":"ex9","label":"neither","flags":[],"target_group":null,"confidence":"high","rationale":"Reports an analyst's claim about party ownership without attacking Luo people."}`

Input: `{"post_id":"ex10","text":"A speaker shouted 'tufukuze madoadoa'. This is the language of 2007 and NCIC must act."}`
Output: `{"post_id":"ex10","label":"neither","flags":[],"target_group":null,"confidence":"high","rationale":"Quotes incitement in order to condemn it and demand intervention."}`

## Output contract

Return strict JSONL, one object per input post in the same order, and nothing
else:

```json
{"post_id":"<exact input id>","label":"hate|offensive|neither","flags":["dehumanisation"|"violence_call"|"ethnic_targeting"|"coded_language"],"target_group":"<string or null>","confidence":"high|medium|low","rationale":"<one sentence quoting the operative phrase>"}
```

Every input ID must appear exactly once. Do not merge, split, add, or reorder
rows. If a post is empty or unintelligible, label `neither` with low
confidence.
