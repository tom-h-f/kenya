"""RTX 3070 throughput and fp16 agreement for the incitement NLI pass.

    PYTHONPATH=analysis/src uv run python bench.py sample.parquet 5000

`sample.parquet`: 6,000 random scored posts (platform_post_id, text and the four stored
`*_score` columns from `incitement/`), drawn with `using sample 6000 rows (reservoir, 7)`.
Sweeps batch size for fp16 and fp32 (sorted by length, then unsorted), then scores all
N posts at the best batch and compares against the stored scores and against each
other. QUICK=1 skips the sweep and uses batch 32. Results: docs/analysis/incitement-backlog.md
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch
from transformers import pipeline
from kma.incitement import nli_scores, MODEL
from kma.classify import MAX_LENGTH
df = pd.read_parquet(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else len(df)
df = df.head(N).reset_index(drop=True)
COLS = ["dehumanisation","violence_call","othering","political_criticism"]
stored = df[[c + "_score" for c in COLS]].to_numpy()
def mk(dtype):
    return pipeline("zero-shot-classification", model=MODEL, truncation=True, max_length=MAX_LENGTH, device=0, dtype=dtype)
def run(pipe, texts, bs, sort):
    order = np.argsort([len(t) for t in texts]) if sort else np.arange(len(texts))
    st = [texts[i] for i in order]
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.time()
    res = nli_scores(pipe, st, bs)
    torch.cuda.synchronize(); dt=time.time()-t
    out = np.zeros((len(texts),4))
    for j,i in enumerate(order): out[i]=[res[j][c] for c in COLS]
    return out, len(texts)/dt, torch.cuda.max_memory_allocated()/2**30
def agree(a, b, name):
    d = np.abs(a-b)
    m1, m2 = a[:, :3].max(1), b[:, :3].max(1)
    r = dict(name=name, mean_abs=float(d.mean()), p99_abs=float(np.quantile(d,.99)), max_abs=float(d.max()),
             menace_flip_0_5=float(((m1>.5)!=(m2>.5)).mean()), menace_flip_0_9=float(((m1>.9)!=(m2>.9)).mean()),
             argmax_agree=float((a.argmax(1)==b.argmax(1)).mean()),
             corr=[float(np.corrcoef(a[:,k],b[:,k])[0,1]) for k in range(4)])
    print(json.dumps(r), flush=True)
texts = df.text.tolist()
res = {}
BEST = dict(float16=32, float32=32)
for dtype in (torch.float16, torch.float32):
    p = mk(dtype); tag = str(dtype).split(".")[-1]
    nli_scores(p, texts[:64], 32)
    for sort in (() if os.environ.get('QUICK') else (True, False)):
        for bs in (16, 32, 64, 128, 256, 512):
            if sort is False and bs > 32: break
            try:
                o, rate, vram = run(p, texts[:1000], bs, sort)
                print(f"{tag} sort={sort} bs={bs}: {rate:.1f} posts/s peak {vram:.2f} GiB", flush=True)
                if sort and rate > BEST.get(tag + "_rate", 0): BEST[tag + "_rate"] = rate; BEST[tag] = bs
                if vram > 6.5: break
            except torch.cuda.OutOfMemoryError:
                print(f"{tag} sort={sort} bs={bs}: OOM", flush=True); torch.cuda.empty_cache(); break
    o, rate, vram = run(p, texts, BEST[tag], True)
    print(f"{tag} full {len(texts)}: {rate:.1f} posts/s peak {vram:.2f} GiB", flush=True)
    res[tag] = o; agree(o, stored, f"{tag}_vs_stored")
    del p; torch.cuda.empty_cache()
agree(res["float16"], res["float32"], "fp16_vs_fp32")
sub = slice(0, 2000)
agree(res["float16"][sub], stored[sub], "fp16_vs_stored_first2000")
agree(res["float32"][sub], stored[sub], "fp32_vs_stored_first2000")
