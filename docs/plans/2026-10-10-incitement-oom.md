# Incitement pass OOM on tf1 (2026-10-10)

## Symptom
`analysis-enrich-1` (mem_limit 2g on a 3.8 GiB host) kills the incitement pass
with rc=-9 every cycle. Incitement partitions in R2 stop at 2026-09-28; the
collector's hate-seeking gates read those flags, so they are stale.

## Measured
- 2 of 2 incitement passes killed since the container started 14:17Z; embed,
  classify and hate passes: 0 of 7 killed.
- Every pass ramps anon memory to ~2.0 GiB while `pending_posts` scans the
  corpus (DuckDB limit is 1.5 GiB, derived from the cgroup), then survives
  by swapping. Incitement is the only pass that then runs a second 2 GiB peak.
- That peak is inference: 4 NLI pairs per post at ENRICH_BATCH_SIZE=64. CPU,
  2 threads, 128 real posts: batch 64 adds 1391 MB of activations (peak RSS
  2246 MB, 0.7 posts/s); batch 16 adds 56 MB (840 MB, 1.1 posts/s).
- cgroup swap sat at its 2 GiB cap when the kill landed.

## Fix
1. Cap the NLI inference batch at 16 on CPU (`INCITEMENT_CPU_BATCH_SIZE`),
   accelerators untouched. Tests in tests/test_incitement.py.
2. Deploy: replace tf1 `analysis/src/kma/incitement.py` only (identical to
   the repo base apart from this change), rebuild the enrich image (only the
   `COPY . .` layers change), recreate `analysis-enrich-1`.
3. No mem_limit change: the host has 3.8 GiB and ~0.5 GiB of other containers;
   a 3g cap risks the host-level OOM seen 2026-09-29.

## Verify
Next incitement pass completes without rc=-9 and a new `incitement/` run
appears in R2.

## Not fixed here
Throughput. R2 on 2026-10-10: 2,486,239 distinct posts with text, 1,011,691
incitement ids, 1,474,548 unscored. 36k-67k posts/day arrive (>=93% of each
day since 09-28 unscored); the pass scores 500 per ~2 h cycle (~6k/day) at
~1 post/s on 2 CPUs. Even a never-failing pass cannot keep up. Needs a
decision (Modal backfill, a smaller scored subset, or a cheaper model), not a
knob.
