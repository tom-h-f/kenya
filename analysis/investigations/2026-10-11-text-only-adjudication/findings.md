# Text-only adjudication of explorer communities (2026-10-11)

Question: do the communities held together only by text-similarity edges justify keeping the text trace, and how much of community 0 is text?

Data: public explorer bundle, run 20261011T033003Z (text_similarity percentile 96, threshold 0.85). Communities 41, 42, 45, 77, 90, 101 (text-only, 4-9 accounts), 0 (4,550 accounts, 83.6% text weight), 8 (144 accounts, 98.8% co-retweet, control), 35 (11 accounts, 100% co-retweet, control). Posts and edge evidence come from `evidence/c<cid>.json` and `graph/c<cid>.json`. Reading is blind, by headless Claude Code (model claude-sonnet-5) in the isolated setup of `2026-09-11-textsim-sensitivity/05_a2_headless.py`: empty temp cwd, no tools, no MCP, hooks off, no session persistence, user settings not loaded. `02_read.py --probe` confirmed only the harness context (email, environment, date, model name) loads. The reader never sees community ids, trace names, trace shares, similarity scores, handles or account ids.

## Method

- Small communities: every post in the evidence map (4-18 posts).
- Communities 0 and 8: 40 posts. Community 0: 5 text-edge pairs from each of 4 edge-similarity quartiles (20 pairs, 40 posts), seeded draw. Community 8 has only 19 text pairs, so the sample is its 40 posts drawn at random from the evidence map (mostly originals its accounts reposted). Community 35's evidence map holds 33 posts from one source account, all of them read.
- Per post the reader sees text, timestamp, and how many distinct accounts posted near-identical text. "Near-identical" here is a word-overlap rule I added (word-set Jaccard >= 0.6 after stripping urls and @mentions, connected components); a post with no such partner shows 1 account.
- Two runs per community with different post shuffles. One extra pair of calls classifies community 0's 20 biggest near-identical families.
- 21 headless calls in total (1 probe, 18 community, 2 family), about $0.40 reported cost.

## Verdicts

Run 1 / run 2. cat = category, pat = text pattern. org = organic_same_topic, coord = coordinated_messaging, camp = campaign_hashtag_drive. Pairs: text edges in evidence; copy pairs: those with word Jaccard >= 0.6.

| cid | posts read | text pairs (copy) | cat | pat | political | Kenya | conf | agree (cat/pat) |
|---|---|---|---|---|---|---|---|---|
| 41 | 18 | 29 (29) | spam_commercial both | template_with_variation both | no | yes | high/high | yes/yes |
| 42 | 11 | 11 (0) | org both | same_topic both | yes | yes | med/med | yes/yes |
| 45 | 9 | 11 (2) | org both | paraphrase both | yes | yes | med/med | yes/yes |
| 77 | 6 | 4 (4) | coord both | template_with_variation both | yes | yes | high/high | yes/yes |
| 90 | 4 | 3 (0) | org both | same_topic both | yes | yes | med/med | yes/yes |
| 101 | 6 | 3 (1) | other / org | same_topic both | no | yes | low/med | no/yes |
| 0 | 40 | 22,200 (see below) | coord both | template_with_variation both | yes | yes | med/med | yes/yes |
| 8 (control) | 40 | 19 (10) | org / coord | same_topic both | yes | no both | med/med | no/yes |
| 35 (control) | 33 | 0 | coord / org | same_topic both | yes | yes | high/med | no/yes |

Self-agreement over the 9 communities: category 6/9, text pattern 9/9, political 9/9, Kenya-relevant 9/9. The three category disagreements are the ones with a mixed or single-source batch: 101 (other vs org, same one-line reading: unrelated chatter plus a recycled quote), 8 (org vs coord, both reasons name Tanzanian trial-update copy threads inside a mostly topical batch), 35 (coord vs org, both reasons describe a single branded news-ticker feed). Category is the unstable field, text pattern is stable.

One-line readings (consistent across both runs):

- 41: one coupon-code promo template with randomised filler words, emoji and links, 9 accounts. Commercial spam.
- 42: separately worded replies about "bloggers" and two politicians. No shared copy; same subject only.
- 45: separate users reword the same question about one claim. Paraphrase.
- 77: four near-identical keyword-list posts with a shared hashtag, same day. Template with a political aim, but 4 accounts and 4 pairs.
- 90: four differently worded replies on ethnic politics. Same subject only.
- 101: unrelated Sheng posts on money and relationships plus one recycled generic quote. Not political.
- 8: Tanzanian politics (Lissu trial updates, protest posts). Reader calls it not Kenya-relevant in both runs although the explorer's Kenya model scores it 0.95.
- 35: one branded Kenyan news-ticker feed.

Observation: literal overlap predicts the verdict. Text-only communities with copy pairs share (41: 29/29, 77: 4/4) were read as spam and template messaging. Those with almost none (42: 0/11, 90: 0/3, 45: 2/11, 101: 1/3) were read as organic or unrelated. Cosine alone does not separate them: pair similarity runs 0.85-0.92 in the organic communities, 0.85-0.97 in 41 and 0.97-0.99 in 77, but 45 (organic paraphrase) reaches 0.97.

## Community 0 text edge mass

- Graph edges 21,964; 20,415 carry the text trace (93%, bit 16); trace_share gives text 83.6% of edge weight. Of the 20,415, 17,363 have evidence in the bundle (the rest are capped or cross to thin-graph keys), and the figures below use those.
- Word overlap of the two posts on each evidenced text edge, in fifths: 0-0.2: 8,961 (52%); 0.2-0.4: 5,744 (33%); 0.4-0.6: 1,535 (9%); 0.6-0.8: 523 (3%); 0.8-1.0: 600 (3%). Only 1,123 edges (6.5%) link posts that are literal near-copies. The rest are embedding-similar posts about the same subject (the Nairobi governor race, Sifuna, Ruto, poll results) in different words.
- Joining all text pairs by transitive closure makes one component of 4,183 posts and 2,279 accounts spanning 3,174 hours, with 60% of edges. A sample of it is replies and quote-reactions on one political topic, not a script. This is a chaining artefact of the 0.85 threshold, so family counts over the raw text edges overstate copying.
- Reader on the 40 sampled posts: coordinated messaging with template-with-variation in both runs (confidence medium), but the mixtures are about 40-45% coordinated, 25-40% organic, 10-20% hashtag drive. Its stated evidence is a set of wire-style news lines (a minister's travel, a refinery groundbreaking, a poll figure) repeated by many accounts, which reads as news-copy amplification rather than a political script. Treat the coordinated label as medium.
- Reader on the 20 biggest literal-copy families (295 of the 1,123 copy edges): category agreement 12/20, pattern agreement 14/20. Edge-mass by category, run 1 / run 2: organic 183 / 177, coordinated 34 / 80, hashtag drive 67 / 10, engagement farming 11 / 20, other 0 / 8. About 60% organic, 20% coordinated, 15% hashtag drive. The hashtag-drive and coordinated shares trade off between runs, so only "mostly organic, some template campaign text" is stable. The largest family (85 edges, 19 posts, 17 accounts over 3 hours) was read as organic paraphrase both times.
- Net: the text trace puts community 0 together, but about 93% of its edges are same-topic similarity and about 6.5% are copies. The copy edges are mostly organic echoing, with a minority of template campaign text.

## Do the text-only communities justify the text trace?

Partly. Of 6 text-only communities, 1 is commercial spam (41), 1 is template messaging with a political aim (77, 4 accounts, one day), and 4 are organic same-topic replies (42, 45, 90, 101). The trace finds things worth seeing (a coupon-spam template, one keyword-salad burst) but as a coordination signal on its own its precision here is 1 in 6 communities. It is the right trace for spam and copy-paste, the wrong one for "same topic". The measured driver is word overlap, not cosine.

## Recommendations

1. Add a literal-overlap floor on text edges, word Jaccard >= 0.6 after stripping urls and @mentions, alongside the 0.85 cosine. On this run it would remove all 11 edges of 42 and 3 of 90, keep 41 and 77 whole, and cut community 0 from 17,363 evidenced text edges to 1,123. Needs a rerun of community detection to see what survives; this study does not measure that.
2. Add a spam filter for template-with-variation text. Community 41 survives because the filler words differ per post while the coupon frame is constant. A frame check (long shared token run with a few swapped tokens, plus a promo-code or coupon vocabulary and links) would label it commercial and keep it out of the coordination list. Measure it on the 29 pairs of 41 before shipping.
3. Do not read text-trace edges alone as coordination in the UI. Label text-only communities "similar wording" and show the word-overlap alongside the cosine.
4. Keep the cosine floor at 0.85 for retrieval but do not raise it as the fix: organic paraphrase in 45 reaches 0.97 while 41 starts at 0.85.
5. For community 0, stop using the transitive component as a family. Cap families on copy pairs only (as here) or the 4,183-post chain will keep appearing as one "campaign".
6. Note for the explorer: the Kenya model scores community 8 at 0.95 but a text reader finds Tanzanian content. Worth a check of the relevance model on Swahili Tanzanian politics.

## Limits

- One reader model, two shuffles, 21 calls. Category self-agreement is 6/9, so single category labels are soft; text pattern, politics and Kenya relevance are stable.
- The reader sees the near-identical-account counts, which come from my word-overlap rule, so verdicts on 41 and 77 partly rest on that count. It was shown as a number only.
- Community 0 is a 40-post sample (20 pairs); its figures are a mass-weighted descriptive, with no confidence interval. The Jaccard 0.6 cut is a descriptive threshold, not tuned.
- Controls 8 and 35 are sampled from evidence posts, which are mostly reposted originals, so they test the reader's pattern calls, not the co-retweet method.
- One family reply (run 1) came back with unquoted enum values; the enum words were quoted by `03_report.py` before parsing and nothing else was changed.
- No account handles or ids are in this file; they are only in untracked `out/`.

## Reproduce

From `analysis/`: `00_fetch.py`, `01_packets.py`, `02_read.py --probe`, `02_read.py`, `03_report.py` (all `uv run python investigations/2026-10-11-text-only-adjudication/<script>`). Outputs, prompts, replies and shuffle keys are in untracked `out/`.
