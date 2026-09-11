"""What the 0.85 text trace actually links: the post pairs behind v2's top 500.

    cd analysis && uv run python investigations/2026-09-11-textsim-sensitivity/06_pair_audit.py \\
        investigations/2026-09-11-textsim-sensitivity/out/2026-09-05-promotion-off [--judge 100]

v2-findings section 4 says 0.85 was chosen "because that is where near-duplicates
sit in our encoder's space". This checks that against the pairs themselves:
every cross-author post pair at or above the cut among the top 500's embedded
posts (from `00_export`'s rows and matrix), a sample's word overlap, and a
random-pair baseline for the embedding space. `--judge N` has a blind headless
reader label N of the sampled pairs same_message / same_topic / unrelated,
with the same isolation as `05_a2_headless.py`.

Found 2026-09-11 while rebuilding A2's dossiers: v2's four empty groups are all
text-similarity edges, and the pairs behind them are unrelated Sheng replies.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from kma import coord2
from kma import coordination as co
from kma.db import connect

CUT = 0.85
SAMPLE = 3000

JUDGE_SYSTEM = """You compare pairs of social media posts, mostly Kenyan English, Swahili and Sheng. \
For each pair decide:
  same_message  the two posts carry the same message or claim - copied, templated or paraphrased
  same_topic    same subject or target, but a different message
  unrelated     different subjects
Ignore @mentions at the start. Judge only the content."""


def unit(matrix: np.ndarray, index: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix[index], dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pairs_at_cut(x: np.ndarray, authors: np.ndarray, cut: float) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = [], []
    for start in range(0, len(x), 2048):
        i, j = np.nonzero(x[start : start + 2048] @ x.T >= cut)
        i = i + start
        keep = (j > i) & (authors[i] != authors[j])
        rows.append(i[keep])
        cols.append(j[keep])
    return np.concatenate(rows), np.concatenate(cols)


def texts(con, ids) -> pd.Series:
    con.register("_ids", pd.DataFrame({"platform_post_id": np.unique(ids)}))
    return con.sql(
        f"SELECT platform_post_id, text FROM {co.posts_source('x')} SEMI JOIN _ids USING (platform_post_id) "
        "QUALIFY row_number() OVER (PARTITION BY platform_post_id ORDER BY collected_at DESC) = 1"
    ).df().set_index("platform_post_id")["text"]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def words(text) -> set:
    return set(coord2.clean_text(str(text)).lower().split())


def grams(text, n: int = 4) -> set:
    s = coord2.clean_text(str(text)).lower()
    return {s[k : k + n] for k in range(max(len(s) - n + 1, 1))}


def baseline(matrix, pools: dict[str, np.ndarray], rng) -> None:
    for name, pool in pools.items():
        a, b = rng.choice(pool, 100_000), rng.choice(pool, 100_000)
        cos = (unit(matrix, np.sort(a))[rng.permutation(len(a))] * unit(matrix, np.sort(b))).sum(1)
        print(f"  {name:<22} n={len(pool):>7}  mean {cos.mean():.3f}  p99 {np.quantile(cos, .99):.3f}"
              f"  share >= {CUT} {np.mean(cos >= CUT):.4f}")


def judge(sample: pd.DataFrame) -> pd.Series:
    body = "\n\n".join(
        f"PAIR {k}\nA: {r.text_a.strip()}\nB: {r.text_b.strip()}" for k, r in sample.iterrows()
    )
    prompt = (f"{body}\n\nReturn ONLY a JSON array, one object per pair, in order: "
              '{"pair": <n>, "label": "same_message" | "same_topic" | "unrelated"}')
    cmd = ["claude", "-p", "--model", "claude-sonnet-5", "--system-prompt", JUDGE_SYSTEM, "--tools", "",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
           "--settings", '{"disableAllHooks": true}', "--setting-sources", "local",
           "--no-session-persistence", "--output-format", "json"]
    with tempfile.TemporaryDirectory() as empty:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=empty, timeout=900)
    if proc.returncode:
        raise SystemExit(proc.stderr[-500:])
    out = json.loads(proc.stdout)
    result = next(m for m in (out if isinstance(out, list) else [out]) if m.get("type") == "result")["result"]
    labels = pd.DataFrame(json.loads(result[result.find("[") : result.rfind("]") + 1]))
    return labels.set_index("pair")["label"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshot_dir", type=Path)
    ap.add_argument("--judge", type=int, default=0, help="pairs for the blind reader; 0 skips it")
    args = ap.parse_args()

    rows = pd.read_parquet(args.snapshot_dir / "rows.parquet")
    matrix = np.load(args.snapshot_dir / "matrix.npy", mmap_mode="r")
    top = set(pd.read_parquet(args.snapshot_dir / "sweep_min0.75" / "top500_t0.85.parquet").user_id.astype(str))
    in_top = rows.user_id.astype(str).isin(top).to_numpy()
    index = np.flatnonzero(in_top)
    authors = rows.user_id.astype(str).to_numpy()[index]

    rng = np.random.default_rng(0)
    print("random-pair cosine baseline:")
    baseline(matrix, {"all eligible posts": np.arange(len(rows)), "top-500 authors' posts": index,
                      "other authors' posts": np.flatnonzero(~in_top)}, rng)

    i, j = pairs_at_cut(unit(matrix, index), authors, CUT)
    print(f"\ntop-500 embedded posts {len(index):,}; cross-author pairs >= {CUT}: {len(i):,}")
    pick = np.random.default_rng(0).choice(len(i), size=min(SAMPLE, len(i)), replace=False)
    post_ids = rows.post_id.to_numpy()
    sample = pd.DataFrame({"a": post_ids[index[i[pick]]].astype(str), "b": post_ids[index[j[pick]]].astype(str)})
    text = texts(connect(), np.r_[sample.a, sample.b])
    sample["text_a"], sample["text_b"] = sample.a.map(text), sample.b.map(text)
    sample["tok_j"] = [jaccard(words(x), words(y)) for x, y in zip(sample.text_a, sample.text_b)]
    sample["gram_j"] = [jaccard(grams(x), grams(y)) for x, y in zip(sample.text_a, sample.text_b)]

    q = [.1, .25, .5, .75, .9]
    print(f"sample {len(sample):,}: token Jaccard quantiles {np.round(np.quantile(sample.tok_j, q), 3)}")
    print(f"             char-4gram Jaccard quantiles {np.round(np.quantile(sample.gram_j, q), 3)}")
    print(f"  under 0.1 of words shared: {np.mean(sample.tok_j < 0.1):.3f}; "
          f"near-copies (>= 0.5): {np.mean(sample.tok_j >= 0.5):.3f}")
    sample.to_parquet(args.snapshot_dir / "pair_audit_sample.parquet")

    if args.judge:
        judged = sample.sample(args.judge, random_state=1).reset_index(drop=True)
        judged["label"] = judged.index.map(judge(judged))
        judged.to_parquet(args.snapshot_dir / "pair_audit_judged.parquet")
        print(f"\nblind reader on {len(judged)} pairs:")
        print(judged.label.value_counts().to_string())


if __name__ == "__main__":
    main()
