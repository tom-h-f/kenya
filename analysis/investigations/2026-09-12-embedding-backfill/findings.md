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

## The drain

567,030 posts exported (`db.pending_posts`, the enrich worker's own definition
of not-yet-embedded), encoded on mike-pc in 683 s (830 posts/s, 2.17 GiB peak),
written back in 12 runs of 50,000 under the production key pattern. Vectors
read back as `list<double>` of length 768 with the model and dim columns the
enrich worker writes.

Text-eligible posts on snapshot `2026-09-05-promotion-off` with an embedding:
**485,285 of 485,285, 100%**, against 62% before. The text trace can now see
every eligible post rather than the 62% that happened to be embedded.

Two passes died mid-upload before this one: a dropped PUT, and the Mac sleeping
on battery mid-request (R2 rejects the resumed request as
RequestTimeTooSkewed). Each chunk is now recorded in `out/written.json` and
retried three times, so a restart costs one chunk rather than the nine minutes
of re-reading every embedded id.

## What this does not do

The persisted text trace still comes from the 2026-09-05 export, which was
built when 62% of eligible posts had vectors. Rebuilding it on the fuller set
is a GPU pass plus a re-run of the community report, and it would move the v2
output that A2 was just read against.
