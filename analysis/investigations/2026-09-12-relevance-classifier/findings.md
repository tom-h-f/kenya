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
