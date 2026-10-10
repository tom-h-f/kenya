# Text trace value (2026-10-10)

Findings: `docs/analysis/2026-10-10-text-trace-value.md`. Run in order from
`analysis/`; outputs go to untracked `out/`.

1. `01_export.py`, `01b_collected.py`: window rows, behavioural networks, collection days
2. `02_pairs.py --embeddings manifest|current --tag ...`: post pairs behind the text trace
3. `03_arms.py`: with/without text and short-window arms; `rank.py`: shared ranking helpers
4. `04_null.py`: noise floor; `05_precision.py`; `06_read.py` (blind read, 80 pair budget, spent)
5. `07_unique.py`, `08_window.py`; `10_benchmark.py` (IOHunter, Q2)

Not run: 21 d window at full coverage; `cloud` host timing.
