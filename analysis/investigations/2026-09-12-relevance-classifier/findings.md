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
