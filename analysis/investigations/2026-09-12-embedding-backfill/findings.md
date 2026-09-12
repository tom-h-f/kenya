# Embedding backfill on mike-pc

Only 62% of text-eligible posts in the 2026-09-05 snapshot had an embedding, so
the text trace could not see 38% of what people wrote. The enrich worker embeds
with a per-cycle cap and has fallen behind. This drains the backlog on mike-pc's
RTX 3070, with R2 access staying on tac2: `01_export_pending.py` exports what
`db.pending_posts` says is not yet embedded, `02_encode.py` encodes it,
`03_write.py` writes it back in `semantic.embed_new`'s exact row format.

## Check before writing

`02_encode.py` re-encodes posts that are already embedded and refuses to go on
unless the vectors match. Smoke run, 2026-09-12: 500 embedded posts, cosine
against the persisted vectors min 1.000000, p01 1.000000 - fp32 on CUDA
reproduces production's CPU vectors. 5,000 pending posts in 6 s (803 posts/s),
2.17 GiB peak.
