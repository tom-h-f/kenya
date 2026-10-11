# A learned relevance gate

`kma.measure.domain_bucket` is two regexes. Measured against Tom's blind labels
(B2, 2026-09-11) it is precise and leaky: precision 0.971, recall 0.655
corpus-weighted. Its misses are plain Kenyan politics - Orengo, Karua, Sifuna,
#RutoMustGo - and 20 of the 21 misses in that sample sat in the `ambiguous`
bucket, which is 74.7% of the corpus. Kenya share feeds v2's scores, the
dossiers and cluster promotion, so the recall gap is load-bearing.

## Labels

3,000 posts from the pinned snapshot (`01_sample.py`), 70% drawn from
`ambiguous`, with all 300 B2 posts excluded so the evaluation sets never leak
into training. Labelled by headless claude-opus-5 (`02_label.py`) under
`measure_eval`'s own definitions - the labeller whose labels agreed with Tom on
90 of 100 posts, kappa 0.83. 3,000 labelled in 60 chunks, 0 failed, about 2 s
per post-chunk of 50.

| gate bucket | kenya | offdomain | unclear |
|---|---|---|---|
| ambiguous | 276 | 1,000 | 824 |
| kenya | 425 | 17 | 8 |
| offdomain | 0 | 444 | 6 |

`unclear` is dropped for training, as `measure_eval.score` drops it: 2,162 rows.

## Training

`03_train.py` on mike-pc: `Davlan/afro-xlmr-large` (public, so no token on that
box), fp32 weights with bf16 autocast, 8-bit AdamW, batch 32 with gradient
checkpointing, 3 epochs. 90 s, peak 5.29 GiB - inside the ~7 GiB point where
that card starts spilling to shared memory. Held-out 10%: precision 0.947,
recall 0.934.

## Against the B2 labels, corpus-weighted

| set | scorer | precision | recall |
|---|---|---|---|
| Tom's 100 (n=91) | gate | 0.971 | 0.655 |
| | model | 0.918 | 1.000 |
| model-labelled 300 (n=269) | gate | 0.980 | 0.508 |
| | model | 0.974 | 0.975 |

Tom's 100 is the honest test: those posts are not in the training data and the
labels are his, not a model's. The 300 are labelled by the same model family
that produced the training labels, so treat that row as an upper bound.

The trade is recall 0.655 -> 1.000 for precision 0.971 -> 0.918, on 91 posts.
Caveats: one snapshot; training labels are model labels, which is exactly the
mistake that cost the hate-speech work a round (it trained into the labeller's
blind spot on coded incitement) - topical relevance is easier than coded
incitement, and the human check here is the reason to believe it, but a second
human pass over a fresh sample is what would settle it.

Not wired into production. `kma.measure.domain_bucket` still decides
`kenya_share`; adopting the model means scoring the corpus and persisting a
`relevance/` prefix, which is a separate decision.

## Scored over the corpus

1,281,210 posts (every post with text, latest copy) scored on mike-pc in 2,204 s
(581/s, 3.60 GiB peak) and persisted as probabilities to
`relevance/platform=x/model=kenya-relevance-afroxlmr-2026-09-12/`. The cut is
not tuned, so the probability is what is stored; readers choose a threshold.

At p >= 0.5 the model calls **37.7%** of the corpus Kenyan against the gate's
**21.5%**. That is the size the gate's measured recall predicts: 0.215 / 0.655
= 0.33. The distribution is bimodal - 58.5% below 0.1, 32.5% above 0.9 - and it
agrees with the gate where the gate is strong (95.9% of its `kenya` bucket,
1.5% of its `offdomain` bucket). The work happens in `ambiguous`, 76% of the
corpus, of which it calls 22.5% Kenyan.

Failure modes visible in `07_check_scores.py`'s samples, on 1.28M posts rather
than the 91 of the B2 check:

- **Near-empty posts ride on the handle.** "@StomaCop @WMPolice [emoji]" scores
  0.80 - West Midlands Police, no Kenyan content. Replies to Kenyan politicians
  are legitimately Kenyan, which is where the model learned it.
- **Short slogans are missed:** "UDA Wantam" at 0.28.
- **It fixes real gate errors:** Indian and French "DCP" posts drop out, Martha
  Karua and Laikipia posts come in. Only 522 posts move the other way (gate
  off-domain, model Kenyan).

Anything adopting this should think about the first one - a floor on content
words, or a higher threshold where precision matters.

## The independent human check (the one that counts)

100 fresh posts, none of them trained or labelled on before, stratified 50/50 by
what the model says, shown blind with English translations
(`10_human_check.py`, `12_label_page.py`). Tom labelled them 2026-09-13: 50
kenya, 38 offdomain, 12 unclear. The 88 placeable ones:

| call | TP | FP | FN | precision | recall | precision (corpus-weighted) | recall (corpus-weighted) |
|---|---|---|---|---|---|---|---|
| keyword gate | 36 | 1 | 14 | 0.973 | 0.720 | 0.973 | 0.516 |
| classifier | 45 | 3 | 5 | 0.938 | 0.900 | 0.889 | 0.827 |
| gate OR model | 45 | 4 | 5 | 0.918 | 0.900 | 0.875 | 0.827 |

**This supersedes the 0.997 / 0.980 recorded above.** That figure came from the
B2 sets, whose labels a model of the same family produced; on genuinely
held-out posts judged by a human the classifier is weaker. What survives is the
reason it was adopted: it roughly doubles corpus-weighted recall, 0.516 to
0.827, for precision 0.973 to 0.889.

The union with the gate is not worth taking: it adds a false positive and
recovers nothing, because on this sample the gate's catches are a subset of the
model's.

**All five misses are posts whose Kenyan-ness lives in a handle**, which the
scorer strips: `@Ruto_tutam2027 <link>` at 0.09, `RT @LindaMwananchi_: Ni
Mbaya` at 0.07, `@Naomikibandi Imagine akothee as migoris women rep` at 0.17.
Stripping mentions removed the handle-riding false positives AND the handle-
carried true positives; for a Kenya monitor a reply to @Ruto_tutam2027 is about
Kenya. The fix is not to put raw handles back - that is what scored
"@SomeForcePolice [emoji]" at 0.80 - but to give the model a FEATURE for the
mentioned account, such as whether it is a known Kenyan one. Not done.

The three false positives are ordinary: Sheng banter with no politics in it,
and one post about Zimbabwe.

## 2026-10-11: regional false positives, retrained

Superseded by `kenya-relevance-afroxlmr-2026-10-11`. Everything above describes
the 2026-09-12 and 2026-09-13 models. Code, labels' provenance and the 17 candidate
runs: `analysis/investigations/2026-10-11-relevance-regional/`.

The 09-13 model called 114 of 133 Tanzania-only posts Kenyan (85.7%; the persisted
scores gave 115, 86.5%) and 46 of 50 Tanzanian posts in a fresh holdout. Retrained with
regional hard negatives and a Kenyan-handle input, and deployed.

### Labels

| set | rows | source |
|---|---|---|
| original training labels | 3,000 (2,917 after the author split) | claude-opus-5 headless, 2026-09-12. `out/` was untracked: recovered from the main checkout's `out/labelled.parquet`, not regenerated. Now also in R2 `training/relevance/2026-10-11/` |
| regional train pool | 880 (321 kenya, 472 offdomain, 87 unclear) | claude-opus-5 headless |
| regional holdout | 623 (198 kenya, 336 offdomain, 89 unclear) | same reader; author-disjoint from the train pool and from every original-label row kept |
| Tom's human check | 88 placeable of 100 | blind human labels 2026-09-13, recovered from the old `out/` |
| Tom's B2 | 100 (91 placeable), model-labelled 300 (269) | `analysis/out/measure_*.csv`, main checkout |
| tz200 | 200 (133 tanzania_only, 26 kenya) | claude-sonnet-5 headless, 2026-10-11 |

1,503 new labels, inside the ~1,500 budget. The reader is the repo's isolated headless
method (`claude -p`, empty cwd, `--tools ""`, empty MCP config, hooks off, no settings
sources, no session persistence), probe-verified: asked to list its tools, MCP servers,
skills, hooks, CLAUDE.md and files it reported none and the init message had an empty
tool list. Definitions are `02_label.py`'s word for word plus a `country` field; the
only validation of the reader is the earlier kappa 0.83 against Tom. Chunks of 50,
validated, cached; one chunk that failed three times was split in halves.

Sampling (`07_pools.py`): 1,331,743 non-retweet posts with distinct text, each in one
stratum, a labelled post standing for `N_stratum / n_stratum` posts so holdout figures
are corpus-weighted. Strata: seven region groups (Tanzania, Uganda, Great Lakes, Horn,
West Africa, South Africa, other Africa), each split by the previous model's call so its
false positives are over-sampled; community 8 member posts; Kenyan posts that name a
neighbour; diaspora; Swahili that is Kenyan and Swahili that is not (so the model does
not learn "Swahili = not Kenya"); short posts mentioning a Kenyan handle and short posts
riding on a foreign one; mined false positives (p >= 0.9, no Kenyan anchor word); and a
random rest stratum in the holdout. `kenya.tomfr.net/data/<run>/evidence/c8.json` is not
served (the public bundle has `communities.json` and per-community detail only), so
community 8 evidence is the member-post file from the text-edge-floor study. Community
ids are per run: the Tanzanian community of the published 2026-10-11 run is cid 9
(`tanzaniamassacre`, `mo29`, `samiamuuaji`).

### Training

`05_train.py` on mike-pc: afro-xlmr-large, fp32 weights, bf16 autocast, 8-bit AdamW,
batch 32, gradient checkpointing, best epoch by a fixed 10% validation split. Peak
5.29 to 5.31 GiB; 2 to 3.5 minutes a run. Text goes through `kma.relevance_prep`, the
function serving uses; the variant and the handle list ship in the model directory.

Handle input: the five old misses on Tom's check all lived in a stripped handle.
`06_handles.py` takes a handle as Kenyan when at least 20 non-retweet posts mention it,
at least 30% of them carry a Kenyan anchor word and at most 5% name a neighbouring
country; neither input is a learned score. 919 handles, including `@Ruto_tutam2027`,
`@LindaMwananchi_` and `@Naomikibandi`, excluding `@SomeForcePolice`. Variants: `strip`
(every mention removed, the old rule), `known` (keep listed handles), `known_lex` (also
keep a handle that contains a Kenyan term). 17 runs, `candidates.csv`. Learning rate 1e-5
for 3 epochs underfits the harder data; 2e-5 to 3e-5 for 4 to 5 epochs is stable across
seeds. Chosen: `known_lex`, lr 3e-5, 4 epochs, seed 1 - best Tom-label F1 among runs with
weighted regional false-positive rate under 0.10. It was chosen on the human check and
the holdout, so those figures carry selection optimism; the mean of the seven runs at
lr >= 2e-5 is shown beside it.

### Old against new, same posts, threshold 0.5

Weighted as in 2026-09-13 (gate-bucket shares for the human sets, which reproduces
0.889 / 0.827; stratum shares for the holdout).

| set | old (09-13) | new (chosen) | new, mean of 7 |
|---|---|---|---|
| Tom's 88, weighted P / R / F1 | 0.889 / 0.827 / 0.857 | 0.900 / 0.931 / 0.915 | 0.896 / 0.891 / 0.893 |
| Tom's 88, unweighted P / R | 0.938 / 0.900 | 0.941 / 0.960 | |
| Tom's B2 100 (n=91), weighted P / R / F1 | 0.997 / 0.980 / 0.989 | 0.918 / 1.000 / 0.957 | 0.957 F1 |
| Tom's B2 100, unweighted F1 | 0.973 | 0.974 | |
| model-labelled 300 (n=269), weighted F1 | 0.924 | 0.983 | |
| regional holdout, weighted P / R / F1 | 0.963 / 0.765 / 0.852 | 0.993 / 0.953 / 0.972 | 0.959 F1 |
| regional holdout, unweighted P / R / F1 | 0.535 / 0.894 / 0.669 | 0.858 / 0.914 / 0.885 | 0.893 F1 |
| regional false-positive rate, weighted | 0.322 | 0.076 | 0.071 |
| regional false-positive rate, unweighted | 0.536 (147 of 274) | 0.099 (27 of 274) | 0.094 |
| Tanzanian posts (reader: offdomain) called Kenyan | 46 of 50 | 5 of 50 | |
| tz200 Tanzania-only called Kenyan | 114 of 133 (85.7%) | 18 of 133 (13.5%) | 11.6% |
| tz200 Kenya recall | 24 of 26 | 20 of 26 | 74.7% |
| hard-positive recall (Kenyan posts naming a neighbour, diaspora, Swahili, handle-driven) | 0.914 | 0.975 | 0.965 |

The false-positive rate is over holdout posts the reader called `offdomain` that name
a neighbouring or African country. By group, new / old (weighted): Tanzania 0.055 /
0.942, Uganda 0.230 / 0.779, Great Lakes 0.049 / 0.247, Horn 0.034 / 0.055, West Africa
0.071 / 0.185, South Africa 0.056 / 0.212, other Africa 0.066 / 0.220. Uganda is the
weak spot because its stories overlap Kenya's (Besigye's detention and the Kenyan
protests over it, Kenyan police arrests in Kampala).

Three-way confusion on the holdout (reader label by the model's call), old: kenya 177
called Kenyan / 21 not, offdomain 154 / 182, unclear 32 / 57. New: kenya 181 / 17,
offdomain 30 / 306, unclear 45 / 44.

Threshold sweep, new model (holdout weighted F1 / false-positive rate): 0.5 0.972 /
0.076, 0.8 0.970 / 0.047, 0.9 0.971 / 0.031, 0.95 0.943 / 0.024; Tom's 88 recall is
0.931 at 0.5 and 0.862 at 0.9. The old model at 0.9 cuts its false-positive rate only to
0.121 and halves Tom's recall (0.528). The new model is insensitive to the cut: 0.5 stays,
0.8 where precision matters more than a point of recall.

Where it still errs. False positives are Swahili procedural threads
(`#UPDATES_KESI_YA_UHAINI`, the treason-trial updates) and replies to Kenyan accounts
about foreign events (`@ntvkenya Sudan is being freed`): the Kenyan handle lifts them,
which is the price of the handle input. Misses are Kenyan stories that name a neighbour
(Kenyans abroad, Kenyan clubs in CECAFA) and non-political Swahili chatter. Three of the
six tz200 "kenya" posts the new model misses are Tanzanian trial updates the sonnet
reader mislabelled, so 74.7% understates recall.

Caveats:
- The regional labels are one model's, the failure mode the hate-speech work met. The
  human sets moved the same way, but a second fresh human pass is still owed.
- Tom's 88 were drawn stratified by the previous model's calls and are one of the sets
  the new model was chosen on.
- B2 weighted F1 falls 3.2 points (two false positives in 91, amplified by the weights);
  unweighted it is flat.
- The handle list was computed over the whole corpus including holdout posts (anchor
  and region words only, no labels).
- tz200's 114 of 133 differs from the recorded 115 by one post: bf16 batch effects in a
  fresh scoring against the persisted scores.

### Deployed and rescored

Weights: `models/relevance/kenya-relevance-afroxlmr-2026-10-11/` (config, tokenizer,
safetensors, `prep.json`, `kenyan_handles.txt`). `08_publish_model.py --verify` fetched
them back into an empty cache and re-scored the B2 sets on a CPU: 0.918 / 1.000 and
0.976 / 0.990, identical to the GPU run. The 09-13 weights and scores are untouched.
Labelled sets and the handle list: `training/relevance/2026-10-11/` in R2.

Active slug: `relevance.ACTIVE_MODEL`, with `PREVIOUS_MODEL` beside it
and `KMA_RELEVANCE_MODEL` to point one process at another slug. Every reader
(`dashboard`, `coord2_communities`, `dossier`, `explorer_build`, `modal_relevance`)
takes `relevance.MODEL`; `test_relevance.py` pins that. The incitement backlog is the
one deliberate exception, see below. `relevance.score_texts` reads the preparation
(`prep.json`, handle list) from the model directory.

Rescore: all 2,516,710 posts with text, on mike-pc ($0 Modal), 11 resumable parts at
about 620 posts/s, 3.60 GiB peak; no corpus post missing, ids unique. Written to
`relevance/platform=x/model=kenya-relevance-afroxlmr-2026-10-11/` in the usual layout.
`pending_posts` under the new slug then showed 857 (collected since the export). The
Modal app `kma-relevance` was redeployed with its daily schedule and a manual run
loaded the new weights from R2 and scored those 857; `kma.modal_spend` projection is
$5.81 for the month before and after (budget $30).

### What the corpus does (`12_compare_corpus.py`, `corpus_comparison.txt`)

Share called Kenyan (p >= 0.5): old 32.7%, new 36.3%. Of 2,480,344 posts both
scored: 765,230 Kenyan under both, 136,156 newly Kenyan, 45,587 no longer Kenyan.
It rises overall because the handle input recovers Kenyan replies and Swahili
chatter the stripped model missed; among posts with a Kenyan anchor word the share is
93.1% to 95.8%, without one 15.6% to 19.6%. Region-keyword posts with no Kenyan anchor
word, share called Kenyan old / new: Tanzania 0.957 / 0.184, Uganda 0.895 / 0.447,
Rwanda 0.503 / 0.245, DRC 0.356 / 0.141, Nigeria 0.181 / 0.043, Ghana 0.376 / 0.102,
South Africa 0.288 / 0.104. Burundi stays at 0.95 on purpose: those posts are Kenyans
arguing about Burundians in Kenya. Ethiopia and Somalia move by a few points up.

### Drift baseline (`11_drift.py`)

Disagreement with the keyword gate inside the `ambiguous` bucket, from the week of
2026-08-03: old model 0.125 to 0.175, new model 0.148 to 0.225; last 4 weeks 0.205
against the 4 before 0.191 (stable). The new series sits higher because the model
calls more of that bucket Kenyan. Compare the new model against its own history only.

### Communities (cloud, `kma-v2-daily`, 2026-10-11)

Same code, same window (`daily-2026-10-10-21d`), same graph (12,129 accounts, 75
communities, identical membership: only the slug moved). Run 20261011T113946Z (old
model) against 20261011T133655Z (new). `verify.sh postrun`: 8 of 8 passed; peak RSS
2.1 GB. Detail in `community_effect.txt`.

- The Tanzanian community (cid 9, 137 accounts, `tanzaniamassacre`, `mo29`,
  `samiamuuaji`): kenya_model 0.966 to 0.561. Still over the 0.5 floor and still
  listed. Its accounts retweet a lot of Kenyan content (Sifuna, Ruto, KTN); 91 of 132
  scored accounts are still at or over 0.5. The Swahili `MAANDAMANO` posts sit at
  0.5 to 0.8. The model has cut the Tanzanian share, not removed the community.
- cid 29 (9 accounts) 0.951 to 0.199 and cid 40 (6 accounts) 0.689 to 0.311 leave the
  kept set: pro-government Tanzanian accounts replying to @ntvkenya. cid 44 (6
  accounts) 0.194 to 0.903 enters: replies to @WilliamsRuto and @NationAfrica about a
  Nairobi event.
- Kept communities 23 to 22; listed accounts 326 to 317 (17 dropped, 8 added: 9 from
  cid 29, 6 from cid 40, 2 from cid 6 out; 6 from cid 44 and 2 others in). Mean
  kenya_model of listed accounts 0.918 to 0.922. Big Kenyan communities rise 0.03 to
  0.07 (cid 0 0.894 to 0.963) as relevance coverage reaches 1.0.

### Still open

- A second fresh human pass over a new sample: the figures above rest on Tom's 88 posts
  (chosen on) and on one model reader's labels.
- Uganda is the weakest region (0.23 false-positive rate on the holdout); the Tanzanian
  community is still over the floor. More labelled Tanzanian and Ugandan negatives, or a
  floor above 0.5 for communities, would be the next lever. Neither was done.
- The handle list is a fixed snapshot of 2026-10-11. A refresh changes what the model
  sees for a mention, so it needs a rescore, not just a file swap.
- The incitement cron on pi0 signs the old slug (see the pi0 change in the 2026-10-11
  hand-off). Until it is switched, `incitement_backlog.RELEVANCE_MODEL` stays on
  `-2026-09-13`, whose scores stop growing, so newer posts fall back to the keyword gate
  for scoring ORDER only.
