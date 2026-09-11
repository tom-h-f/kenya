# Iran NodePruning divergence (finishing-the-revamp B3)

Run 2026-09-11 on the IOHunter release (zenodo.org/records/13357621), pickles
copied from the `iohunter-bench` Modal volume.

## The question

The reproduction gate scores Iran at **71.43** Macro-F1 against IOHunter's
published **60.83 +/- 1.09** - 10.6 points above, outside two standard
deviations - while the other five countries reproduce. Earlier work ruled out
the rewiring seed, the eigenvector convergence settings and the isolated-node
definition.

## 1. The protocol matches the reference, line by line

Against mminici/InfoOpsGFM - IOHunter's own repository - at the commits that
produced the AAAI results (`src/run_NodePruning.py` a583560 and
`src/my_utils.py` 16e6a32, both 2024-08-19/21; `src/model_eval.py`;
`src/data_loader.py`):

- same data: `data_loader` opens `0.7_datasets.pkl` and uses its `graph` and
  `splits` unchanged, as `kma.iohunter_gate` does;
- same rewiring: neighbour-based isolation, five random non-isolated targets,
  weight 1.0, self-loop removed;
- same sweep: percentiles in 0.5 steps, `<=` cut;
- same selection: train and val merged, `np.argmax` (first best on ties);
- same metric: sklearn macro-F1.

The only differences are the rewiring RNG stream (global `random` seeded
12121995 against our `random.Random(seed)`) and `eigenvector_centrality`'s
defaults (max_iter 100, tol 1e-6 against our 1000 / 1e-8). The repository pins
no dependency versions.

## 2. The eigenvector computation does not move it

`eigenvector_variants.py`. Eigenvector centrality is not unique on a
disconnected graph, and Iran's fused network has 2,372 components with only
72.8% of nodes in the largest, so the iteration rather than the maths decides
what small-component nodes receive. That made it the last plausible mechanism.

| country | target | ours (A+I, 1000, 1e-8) | networkx default (A+I, 100, 1e-6) | plain A (100, 1e-6) | exact eigenvector |
|---|---|---|---|---|---|
| russia | 87.65 | 87.83 | 87.83 | 87.83 | 87.83 |
| venezuela | 95.05 | 95.05 | 95.05 | 95.05 | 95.05 |
| china | 63.66 | 63.67 | 63.67 | 63.67 | 63.67 |
| **iran** | **60.83** | **71.43** | **71.46** | **71.46** | **71.43** |
| UAE | 84.66 | 84.64 | 84.66 | 84.66 | 84.64 |
| cuba | 57.92 | 57.92 | 57.92 | 57.92 | 57.92 |

The matrix-form iteration was checked against networkx's own output before any
row was read. networkx has iterated with (A+I) since at least 2.8.8, so no
environment the authors could have used in 2024 differs on this point anyway.

## 3. The reference code on the reference data does not give the published number

`reference_nodepruning.py` runs IOHunter's NodePruning verbatim, with its own
seed, on the released pickles - `kma.coord2` is not involved at all.

| country | reference code on the release | published | ours (gate) |
|---|---|---|---|
| russia | 87.65 +/- 1.95 | 87.65 +/- 1.95 | 87.83 |
| venezuela | 95.05 +/- 1.47 | 95.05 +/- 1.47 | 95.05 |
| china | 63.66 +/- 0.70 | 63.66 +/- 0.70 | 63.67 |
| **iran** | **71.31 +/- 0.76** | **60.83 +/- 1.09** | **71.43** |

For three countries the reference code reproduces its own published mean AND
standard deviation to two decimals, so this harness is exact. On Iran it gives
71.31.

## Finding

**The published Iran figure cannot be produced by the published code on the
published data.** Our implementation (71.43) agrees with the reference code
(71.31) to within the rewiring noise. The Iran gap is an inconsistency between
the IOHunter paper and its own data release, not a fault in the copy.

The likeliest explanation is that the paper's Iran row came from an earlier
build of the Iran network than the one uploaded to Zenodo. That is inference:
the release carries no version history that could confirm it.

The released Iran network is also unlike the others in the one respect that
touches this baseline: 1,509 of its 14,486 nodes are isolated (self-loop only),
1,361 of them drivers - **29.2% of all Iran positives**, against 0.0-13.1% in
the other five countries. Its `tweetSim` trace holds 4,659 nodes and 4,659
edges, one self-loop per driver and nothing else.

## Consequences

- **Judge the copy on Iran against the reference code on the release,
  71.31 +/- 0.76**, where ours is within noise, and keep the published 60.83
  recorded beside it with this explanation. `kma.iohunter_gate` is NOT changed
  here: moving an acceptance target is a decision, not a fix, and the
  project's rule against re-specifying a criterion without evidence applies.
  The evidence is now on the table for that decision.
- The published Iran figure should not be cited as a target for anything
  built on this release.
- Optional: tell the authors (Minici, Luceri), who are already being contacted
  about the control set.

## B1, found on the way

B1 asks for traces built from the raw IO archive for one campaign and compared
against the shipped networks. The release cannot support that comparison.
`data.zip` holds 56 entries, all under `data/processed/<country>/`. Nodes are
bare integers `0..n-1` with no attributes, and nothing in the release maps them
to account ids. Combined with the archive's still-open campaign attribution
(next-steps A2, check 4), shipped nodes cannot be matched to archive accounts.

Two routes remain: ask the authors for the node-id mapping, or compare
aggregate structure (per-trace edge counts, degree distributions) for a
campaign once attribution is solved. Neither is quick, and neither is started.
