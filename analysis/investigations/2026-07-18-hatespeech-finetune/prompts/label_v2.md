# Hate/offensive labelling - Kenya 2026-27 - prompt v2

> Changed from v1: the NCIC Act standard is now the governing rule for the
> hate/neither boundary. Collective ethnic contempt is hate even with no
> violence or expulsion call - see "The NCIC standard" below. v1 left this
> case unstated and the two labellers split on it.

You are annotating tweets by Kenyan users for an election-monitoring research
dataset. Your labels train a classifier that flags posts for human review by
analysts tracking ethnic incitement ahead of the 2027 Kenyan general election.
Accuracy matters in both directions: missing real incitement costs lives,
and flagging ordinary political criticism as hate destroys the tool's
credibility.

## Classes

Choose exactly one label per post.

**hate** - Language that attacks, dehumanises, threatens, or calls for the
exclusion or harm of a person or group **on the basis of a protected
characteristic**: ethnicity, tribe, race, religion, national origin, gender,
disability, sexual orientation. This includes the Kenyan NCIC Act sense of
hate speech: expression that stirs up ethnic contempt, ridicule, or hatred,
or incites violence or expulsion against an ethnic community.

**offensive** - Vulgar, abusive, insulting, or degrading language that is NOT
grounded in a protected characteristic: profanity, personal insults, abusive
attacks on a politician as an individual, sexual crudeness, generic threats.
Offensive to read, but not group-based hate.

**neither** - Everything else, including robust political criticism, tribal
arithmetic and coalition talk, reporting or quoting hate in order to condemn
it, satire that does not itself attack a group, and ordinary banter.

## The NCIC standard - the governing rule

Kenya's National Cohesion and Integration Act (s.13) makes it hate speech to
use threatening, abusive or insulting words that are **intended or likely to
stir up ethnic hatred** - hatred against a group defined by colour, race,
nationality, or ethnic or national origin. This standard governs every
hate/neither judgement in this task. Apply it directly.

The consequence to internalise: **a violence call is not required.**
Expression that stirs up contempt, ridicule or hatred against an ethnic
community is `hate` on its own terms.

Label `hate` when a post does any of these to an ethnic, racial, religious or
national-origin group **as a group**:

- attributes a negative characteristic collectively - the community is
  corrupt, thieving, stupid, treacherous, parasitic
- declares the community unfit to lead, to hold office, or to belong -
  "a Kalenjin presidency is always a curse", "these people can never be
  trusted with power"
- holds the whole community responsible for the acts of its politicians
- ridicules or degrades the community, its language, or its customs
- calls for its exclusion, expulsion, or harm (the plainest case)

Label `neither` when the target is not the group as a group:

- criticism of a named politician, government, party, or policy, however
  harsh - "Ruto has failed this country and must go"
- ethnic-bloc electoral analysis and voting arithmetic - "Mt Kenya will vote
  as one bloc", "the Luo vote is consolidating"
- describing or condemning ethnic hatred rather than expressing it
- factual or historical statement about a community without contempt

The line, stated once: **criticising a leader who happens to belong to a
community is not hate; transferring that criticism onto the community itself
is.** "Ruto is a thief" is `neither`. "Kalenjins are thieves" is `hate`.

Further rules of thumb:
- Quoting or condemning hate is not hate. Judge the author's stance.
- If a post attacks a politician with a slur tied to their ethnicity, it is
  `hate`, not `offensive` - the ethnic marker is what decides.
- Where the NCIC standard applies, apply it even if the post is mild in tone
  or reads as ordinary opinion. Contempt does not have to be loud.
- When genuinely torn between two labels, pick the lower-severity one and set
  `confidence: low`. This does **not** override the NCIC standard - a clear
  case of collective ethnic contempt is `hate` with `confidence: high`.

## Kenya 2026-27 context

Election cycle towards the August 2027 general election. Discussion is
dominated by ethnic-bloc politics, the Ruto/Gachagua fallout, Gen-Z protest
movements, and 2007-08 post-election-violence memory. Posts mix English,
Swahili and Sheng inside a single sentence. **Judge meaning, not language** -
a post is not benign because the operative phrase is in Sheng.

### Coded-term glossary

These terms carry historical weight in Kenyan incitement. The `fp_risk`
column is how often the term is used innocently.

| term | meaning | fp_risk |
|---|---|---|
| madoadoa | "spots/stains" - marking a community for removal; NCIC-flagged, 2007-08 era | low |
| kwekwe | "weeds" (to be uprooted); NCIC-flagged | medium |
| sangari | "couch grass to uproot"; 2007-08 era | low |
| madimoni | "demons" | medium |
| mende | "cockroaches" - also an everyday word for insects | high |
| nyoka | "snakes" - also an everyday word, and a common insult for an individual | high |
| fumigate / fukiza | pest-control framing applied to people; 2017 era | medium |
| rudi kwao / warudi kwao | "(they should) go back to their homeland" | medium |
| fukuza / tufukuze | "chase away, expel" - generic verb, context decides the target | high |
| waondoke | "they must leave" | medium |
| watajua hawajui | "they will know they don't know" - NCIC-flagged veiled threat | low |
| kimeumana | "it has bitten / things have escalated" - 2017-era coded | medium |
| watu wa kule | "those people from there" | medium |
| wabara | coastal term for upcountry people; MRC-era othering | medium |
| 41 vs 1 | 2007-era anti-Kikuyu coalition framing | low |
| uthamaki | Kikuyu kingship/supremacy discourse marker | medium |
| goons / mbogi | hired political thugs; also ordinary slang for a crew of friends | high |

**A coded term alone is not hate. Coded term + an identifiable target group +
harmful intent is hate.** "Kuna nyoka kwa shamba" (there is a snake in the
farm) is `neither`. "Nyoka wa Mt Kenya lazima waondoke" (the snakes of Mt
Kenya must leave) is `hate` - dehumanising metaphor, ethnic target, expulsion.

## Flags

Set every flag that applies (empty list if none):

- `dehumanisation` - the target is described as vermin, disease, filth,
  demons, or otherwise as less than human.
- `violence_call` - calls for, threatens, celebrates or approves physical
  violence, including veiled threats.
- `ethnic_targeting` - an ethnic, tribal, religious or regional community is
  the target.
- `coded_language` - meaning is carried by euphemism, metaphor, or a glossary
  term rather than plain statement.

Consistency: a post labelled `neither` must not carry `violence_call`. Any
post with `ethnic_targeting` plus `dehumanisation` or `violence_call` should
almost always be labelled `hate`.

`target_group`: the named or clearly implied group ("Kikuyu", "Luo",
"Somalis", "Mt Kenya", "Muslims"), or null if there is no group target.

## Examples

Input: `{"post_id": "ex1", "text": "Hii serikali ya Ruto imeuza nchi. 2027 tunawaondoa kwa kura."}`
Output: `{"post_id": "ex1", "label": "neither", "flags": [], "target_group": null, "confidence": "high", "rationale": "Harsh political criticism with an explicit ballot-box remedy: 'tunawaondoa kwa kura'."}`

Input: `{"post_id": "ex2", "text": "Mt Kenya bloc will deliver 3 million votes, Western is still undecided."}`
Output: `{"post_id": "ex2", "label": "neither", "flags": [], "target_group": null, "confidence": "high", "rationale": "Ordinary ethnic-bloc electoral arithmetic, no attack."}`

Input: `{"post_id": "ex3", "text": "Hawa madoadoa wote lazima warudi kwao before December. Hatutaki wageni hapa."}`
Output: `{"post_id": "ex3", "label": "hate", "flags": ["dehumanisation", "ethnic_targeting", "coded_language"], "target_group": "non-local ethnic community", "confidence": "high", "rationale": "'madoadoa' plus 'lazima warudi kwao' - 2007-08 removal framing with a deadline."}`

Input: `{"post_id": "ex4", "text": "Tunaua nyoka na mayai yake. Hakuna kuachilia hata mmoja."}`
Output: `{"post_id": "ex4", "label": "hate", "flags": ["dehumanisation", "violence_call", "coded_language"], "target_group": "implied ethnic community", "confidence": "medium", "rationale": "'tunaua nyoka na mayai yake' - killing the snake and its eggs extends to offspring; eliminationist even though the target is unnamed."}`

Input: `{"post_id": "ex5", "text": "This useless idiot MP is a thief and a fool. Amekula pesa yetu yote."}`
Output: `{"post_id": "ex5", "label": "offensive", "flags": [], "target_group": null, "confidence": "high", "rationale": "'useless idiot', 'thief' - personal abuse of an individual politician, no protected characteristic."}`

Input: `{"post_id": "ex6", "text": "Nyoka imeingia kwa store yangu usiku, nimeshtuka sana."}`
Output: `{"post_id": "ex6", "label": "neither", "flags": [], "target_group": null, "confidence": "high", "rationale": "Literal snake in a store; glossary term used in its everyday sense."}`

Input: `{"post_id": "ex7", "text": "Wewe mbwa wewe, hio mbogi yako ni washenzi tu. Fala."}`
Output: `{"post_id": "ex7", "label": "offensive", "flags": [], "target_group": null, "confidence": "medium", "rationale": "'mbwa', 'washenzi', 'fala' - crude Sheng abuse aimed at a person and their crew, not an ethnic group."}`

Input: `{"post_id": "ex8", "text": "Watajua hawajui. Wakithubutu kuiba kura tena, hii nchi itawaka moto."}`
Output: `{"post_id": "ex8", "label": "hate", "flags": ["violence_call", "coded_language"], "target_group": null, "confidence": "medium", "rationale": "NCIC-flagged veiled threat 'watajua hawajui' plus 'itawaka moto' - threat of violent consequence."}`

Input: `{"post_id": "ex9", "text": "A politician told the rally 'tufukuze madoadoa' yesterday. This is exactly the language of 2007 and NCIC must act now."}`
Output: `{"post_id": "ex9", "label": "neither", "flags": ["coded_language"], "target_group": null, "confidence": "high", "rationale": "Quotes incitement in order to condemn it: 'NCIC must act now'."}`

Input: `{"post_id": "ex10", "text": "Hao watu wa kule ni mende tu, wanajaza mji wetu. Tunahitaji fumigation."}`
Output: `{"post_id": "ex10", "label": "hate", "flags": ["dehumanisation", "ethnic_targeting", "coded_language"], "target_group": "watu wa kule (othered community)", "confidence": "high", "rationale": "'mende' plus 'fumigation' - pest framing of a community described as infesting the city."}`

Input: `{"post_id": "ex11", "text": "A Kalenjin presidency is always a curse to this nation. NEVER AGAIN."}`
Output: `{"post_id": "ex11", "label": "hate", "flags": ["ethnic_targeting"], "target_group": "Kalenjin", "confidence": "high", "rationale": "'A Kalenjin presidency is always a curse' declares an entire community categorically unfit to lead - NCIC ethnic contempt, no violence call needed."}`

Input: `{"post_id": "ex12", "text": "Hawa watu ni wezi tu, wamekuwa wakiiba tangu uhuru. Hawawezi kubadilika."}`
Output: `{"post_id": "ex12", "label": "hate", "flags": ["ethnic_targeting"], "target_group": "implied ethnic community", "confidence": "medium", "rationale": "'hawa watu ni wezi tu ... hawawezi kubadilika' - collective and immutable attribution of thieving to a community."}`

Input: `{"post_id": "ex13", "text": "Gachagua is a thief and a tribalist who looted this county dry."}`
Output: `{"post_id": "ex13", "label": "offensive", "flags": [], "target_group": null, "confidence": "high", "rationale": "'thief', 'tribalist' aimed at one named politician, not transferred onto his community."}`

Input: `{"post_id": "ex14", "text": "The Luo vote is consolidating behind Raila again while Western stays split."}`
Output: `{"post_id": "ex14", "label": "neither", "flags": [], "target_group": null, "confidence": "high", "rationale": "Ethnic-bloc voting analysis with no contempt: 'the Luo vote is consolidating'."}`

## Output contract

You will receive a JSONL block, one post per line. Return **strict JSONL,
one object per input post, in the same order**. Output nothing else: no
preamble, no explanation, no markdown code fences, no trailing commentary.

Schema per line:

```
{"post_id": "<echo the input post_id exactly>", "label": "hate|offensive|neither", "flags": ["dehumanisation"|"violence_call"|"ethnic_targeting"|"coded_language"], "target_group": "<string or null>", "confidence": "high|medium|low", "rationale": "<one sentence quoting the operative phrase>"}
```

Every input post_id must appear exactly once in your output. Do not add,
merge, split, or reorder posts. If a post is empty or unintelligible, label
it `neither` with `confidence: low`.
