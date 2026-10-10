"""Incremental topic assignment: nearest fitted posts decide a new post's topic.

The topic fit (`semantic.persist_topics`) is a UMAP + HDBSCAN batch job and
keeps no model, so a post embedded after the fit has no topic until the next
refit. This module gives each such post the topic of its nearest neighbours
among the posts the newest fit assigned to a topic, in the original 768-d
cosine space, and writes the result to the same `topics/` prefix.

    python -m kma.topics_assign calibrate --manifest m.json --cache DIR
    python -m kma.topics_assign run --manifest m.json --cache DIR --run-id ID

It reads and writes through a `Store`. In production that is a `ManifestStore`:
a JSON file of short-lived signed urls made by the scheduler, so the GPU box
holds no R2 credentials. `S3Store` takes the R2 keys from the environment.

The fit's own assignments are the only reference. A post that is in the fit is
never reassigned, and a knn row made against an older fit counts as unassigned,
because topic ids are only meaningful within one fit. This is not a substitute
for a refit: it cannot create topics or move the boundaries of existing ones.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DIM = 768
PLATFORM = "x"
METHOD = "knn"

EXIT_MORE_WORK = 3
EXIT_OOM = 4

# `-1` is the fit's own label for "in no topic".
NOISE = -1

# Topic sizes in the fit, for reporting how small topics fare against large.
SIZE_BINS = ((0, 100), (100, 500), (500, 10**9))

DEFAULT_CHUNK_ROWS = 50_000
DEFAULT_QUERY_BLOCK = 1024
REF_TILE = 262_144
MAX_K = 50

_TOPICS_SCHEMA = pa.schema(
    [
        ("platform_post_id", pa.string()),
        ("topic", pa.int64()),
        ("model", pa.string()),
        ("params", pa.string()),
        ("assigned_at", pa.timestamp("us", tz="UTC")),
    ]
)


def slug(model: str) -> str:
    return model.split("/")[-1]


def embeddings_prefix(platform: str = PLATFORM, model: str = MODEL) -> str:
    return f"embeddings/platform={platform}/model={slug(model)}/"


def topics_prefix(platform: str = PLATFORM, model: str = MODEL) -> str:
    return f"topics/platform={platform}/model={slug(model)}/"


def marker_prefix(platform: str = PLATFORM, model: str = MODEL) -> str:
    # Outside `topics/platform=*/`, so no reader's glob can match a marker.
    return f"topics/_assign/platform={platform}/model={slug(model)}/"


def run_day(run_id: str) -> str:
    return f"{run_id[0:4]}-{run_id[4:6]}-{run_id[6:8]}"


def chunk_key(run_id: str, index: int, platform: str = PLATFORM, model: str = MODEL) -> str:
    return (
        f"{topics_prefix(platform, model)}dt={run_day(run_id)}"
        f"/run={run_id}-c{index:04d}.parquet"
    )


def marker_key(run_id: str, platform: str = PLATFORM, model: str = MODEL) -> str:
    return f"{marker_prefix(platform, model)}run={run_id}.json"


def run_id_of(key: str) -> str:
    stem = Path(key).stem
    return stem[4:] if stem.startswith("run=") else stem


# --- storage ---------------------------------------------------------------


@dataclass(frozen=True)
class Obj:
    key: str
    size: int


class ManifestStore:
    """Signed urls from a JSON manifest, so the job needs no credentials.

        {"version": 1, "run_id": ...,
         "objects": {key: {"size": int, "url": GET url}},
         "puts": {key: {"put": PUT url, "get": GET url}}}
    """

    def __init__(self, manifest: dict, attempts: int = 4):
        if manifest.get("version") != 1:
            raise ValueError(f"unsupported manifest version {manifest.get('version')!r}")
        self.objects = manifest["objects"]
        self.puts = manifest.get("puts", {})
        self.run_id = manifest.get("run_id")
        self.attempts = attempts

    @classmethod
    def load(cls, path: Path) -> ManifestStore:
        return cls(json.loads(Path(path).read_text()))

    def list(self, prefix: str) -> list[Obj]:
        return sorted(
            (Obj(k, int(v["size"])) for k, v in self.objects.items() if k.startswith(prefix)),
            key=lambda o: o.key,
        )

    def put_keys(self) -> list[str]:
        return sorted(self.puts)

    def _retry(self, what: str, fn):
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                return fn()
            except (OSError, ValueError) as exc:
                last = exc
                time.sleep(min(2**attempt, 20))
        raise RuntimeError(f"{what} failed after {self.attempts} attempts: {last}") from last

    def get(self, key: str, dest: Path) -> None:
        if key in self.objects:
            url, size = self.objects[key]["url"], int(self.objects[key]["size"])
        elif key in self.puts:
            url, size = self.puts[key]["get"], None
        else:
            raise KeyError(f"{key} is not in the manifest")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")

        def fetch() -> None:
            with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as fh:
                while block := resp.read(1 << 20):
                    fh.write(block)
            got = tmp.stat().st_size
            if size is not None and got != size:
                raise ValueError(f"{key}: got {got} bytes, listing says {size}")

        self._retry(f"get {key}", fetch)
        tmp.rename(dest)

    def put(self, key: str, src: Path) -> None:
        if key not in self.puts:
            raise KeyError(f"{key} has no signed PUT url in the manifest")
        data = Path(src).read_bytes()
        req = urllib.request.Request(self.puts[key]["put"], data=data, method="PUT")

        def send() -> None:
            with urllib.request.urlopen(req, timeout=300) as resp:
                if resp.status not in (200, 201, 204):
                    raise OSError(f"PUT {key}: HTTP {resp.status}")

        self._retry(f"put {key}", send)


class S3Store:
    """R2 through boto3, with the keys in the environment (R2_* names)."""

    def __init__(self):
        import boto3

        account = os.environ["R2_ACCOUNT_ID"]
        self.bucket = os.environ.get("R2_BUCKET", "kenya-monitor-2027")
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    def list(self, prefix: str) -> list[Obj]:
        out: list[Obj] = []
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix
        ):
            out += [Obj(o["Key"], o["Size"]) for o in page.get("Contents", [])]
        return sorted(out, key=lambda o: o.key)

    def get(self, key: str, dest: Path) -> None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        self.client.download_file(self.bucket, key, str(tmp))
        tmp.rename(dest)

    def put(self, key: str, src: Path) -> None:
        self.client.upload_file(str(src), self.bucket, key)

    def put_keys(self) -> list[str]:
        return []


def mirror(store, prefix: str, cache: Path, workers: int = 8) -> list[Path]:
    """Make `cache/<prefix>` equal to the store's listing, fetching only what is
    missing or a different size. Returns the local paths in key order."""
    objs = store.list(prefix)
    want = {o.key: o.size for o in objs}
    root = Path(cache) / prefix
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file() and path.relative_to(cache).as_posix() not in want:
                path.unlink()
    todo = [
        o
        for o in objs
        if not (Path(cache) / o.key).is_file() or (Path(cache) / o.key).stat().st_size != o.size
    ]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda o: store.get(o.key, Path(cache) / o.key), todo))
    return [Path(cache) / o.key for o in objs]


# --- the fit and what is already assigned ---------------------------------


def parse_params(params: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in params.split(",") if "=" in p)


def file_provenance(path: Path) -> tuple[str, datetime]:
    """`params` and `assigned_at` of a topics file, from its first row."""
    pf = pq.ParquetFile(path)
    head = next(pf.iter_batches(batch_size=1, columns=["params", "assigned_at"]))
    return head.column(0)[0].as_py(), head.column(1)[0].as_py()


@dataclass
class FitState:
    fit_run: str
    fit_files: list[Path]
    knn_files: list[Path]  # knn rows made against this fit
    stale_knn_files: list[Path]  # knn rows made against an older fit


def fit_state(topic_files: list[Path]) -> FitState:
    fits: list[tuple[datetime, str, Path]] = []
    knn: list[tuple[str, Path]] = []
    for path in topic_files:
        params, assigned_at = file_provenance(path)
        if parse_params(params).get("method") == METHOD:
            knn.append((parse_params(params).get("fit_run", ""), path))
        else:
            fits.append((assigned_at, path.name, path))
    if not fits:
        raise RuntimeError("no fit run under topics/: run the full fit first (modal_backfill.py --topic)")
    fits.sort()
    newest_at = fits[-1][0]
    current = [p for at, _, p in fits if at == newest_at]
    fit_run = run_id_of(current[0].name)
    return FitState(
        fit_run=fit_run,
        fit_files=current,
        knn_files=[p for run, p in knn if run == fit_run],
        stale_knn_files=[p for run, p in knn if run != fit_run],
    )


def _read(path: Path, columns: list[str]) -> pa.Table:
    # ParquetFile, not read_table: the mirror's `model=`/`dt=` directories would
    # otherwise be inferred as hive partitions and clash with the `model` column.
    return pq.ParquetFile(path).read(columns=columns)


def read_columns(paths: list[Path], columns: list[str]) -> pa.Table:
    tables = [_read(p, columns) for p in paths]
    return pa.concat_tables(tables) if tables else pa.table({c: [] for c in columns})


# --- embeddings ------------------------------------------------------------


def _vectors(batch_column: pa.Array) -> np.ndarray:
    lengths = pc.list_value_length(batch_column)
    if lengths.null_count or pc.min(lengths).as_py() != DIM or pc.max(lengths).as_py() != DIM:
        raise ValueError(f"embedding column is not {DIM}-d everywhere")
    flat = pc.list_flatten(batch_column).to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(flat.reshape(-1, DIM), dtype=np.float32)


def plan_rows(
    embedding_files: list[Path], wanted: pa.Array | None, exclude: pa.Array | None = None
) -> list[tuple[Path, np.ndarray]]:
    """Which rows of which files to read, one row per post.

    A post embedded twice is taken from the newest file (descending key order),
    matching `latest_embeddings`. `wanted` keeps only those ids, `exclude`
    drops them; ids are read without touching the 768-d column."""
    plan: list[tuple[Path, np.ndarray]] = []
    claimed: set[str] = set()
    for path in sorted(embedding_files, key=lambda p: p.as_posix(), reverse=True):
        ids = _read(path, ["platform_post_id"]).column(0).combine_chunks()
        keep = pa.array(np.ones(len(ids), dtype=bool))
        if wanted is not None:
            keep = pc.and_(keep, pc.is_in(ids, value_set=wanted))
        if exclude is not None:
            keep = pc.and_(keep, pc.invert(pc.is_in(ids, value_set=exclude)))
        rows = np.flatnonzero(keep.to_numpy(zero_copy_only=False))
        if not len(rows):
            continue
        row_ids = ids.take(pa.array(rows)).to_pylist()
        fresh = np.array([i not in claimed for i in row_ids], dtype=bool)
        rows = rows[fresh]
        if not len(rows):
            continue
        claimed.update(i for i, f in zip(row_ids, fresh) if f)
        plan.append((path, rows))
    return plan


def read_planned(plan: list[tuple[Path, np.ndarray]], batch_rows: int = 32_768):
    """Yield (ids, float32 vectors) for the planned rows, batch by batch."""
    for path, rows in plan:
        pf = pq.ParquetFile(path)
        offset = 0
        for batch in pf.iter_batches(
            batch_size=batch_rows, columns=["platform_post_id", "embedding"]
        ):
            n = batch.num_rows
            lo, hi = np.searchsorted(rows, [offset, offset + n])
            local = rows[lo:hi] - offset
            offset += n
            if not len(local):
                continue
            taken = batch.take(pa.array(local))
            yield taken.column(0).to_pylist(), _vectors(taken.column(1))


# --- reference -------------------------------------------------------------


@dataclass
class Reference:
    vectors: np.ndarray  # float16, L2-normalised, n x DIM
    topics: np.ndarray  # int32, n

    def __len__(self) -> int:
        return len(self.topics)


def build_reference(
    embedding_files: list[Path], fit_table: pa.Table, include_noise: bool
) -> Reference:
    """The fit's posts with their vectors, normalised and held as float16."""
    if not include_noise:
        fit_table = fit_table.filter(pc.greater_equal(fit_table.column("topic"), 0))
    ids = fit_table.column("platform_post_id").combine_chunks()
    topic_of = dict(zip(ids.to_pylist(), fit_table.column("topic").to_pylist()))
    plan = plan_rows(embedding_files, wanted=ids)
    vec_parts: list[np.ndarray] = []
    topic_parts: list[np.ndarray] = []
    for got_ids, vecs in read_planned(plan):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vec_parts.append((vecs / np.maximum(norms, 1e-12)).astype(np.float16))
        topic_parts.append(np.array([topic_of[i] for i in got_ids], dtype=np.int32))
    if not vec_parts:
        raise RuntimeError("no reference vectors found: the fit's posts have no embeddings")
    return Reference(np.concatenate(vec_parts), np.concatenate(topic_parts))


def load_reference(
    cache: Path,
    fit_run: str,
    embedding_files: list[Path],
    fit_table: pa.Table,
    include_noise: bool,
) -> Reference:
    tag = f"ref-{fit_run}-{'noise' if include_noise else 'topics'}"
    vec_path, top_path = Path(cache) / f"{tag}.vec.npy", Path(cache) / f"{tag}.topic.npy"
    if vec_path.is_file() and top_path.is_file():
        return Reference(np.load(vec_path), np.load(top_path))
    ref = build_reference(embedding_files, fit_table, include_noise)
    Path(cache).mkdir(parents=True, exist_ok=True)
    for path, arr in ((vec_path, ref.vectors), (top_path, ref.topics)):
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, arr)
        tmp.rename(path)
    return ref


# --- knn and vote ----------------------------------------------------------


def knn_search(queries: np.ndarray, ref_dev, k: int, block: int, device: str, tile: int = REF_TILE):
    """Top-k cosine neighbours of `queries` (float32, n x DIM) in `ref_dev`, a
    normalised tensor on `device`. Returns (similarities n x k, indices n x k)."""
    import torch

    n = len(queries)
    k = min(k, ref_dev.shape[0])
    sims = np.empty((n, k), dtype=np.float32)
    idx = np.empty((n, k), dtype=np.int64)
    for lo in range(0, n, block):
        q = torch.from_numpy(queries[lo : lo + block]).to(device=device, dtype=torch.float32)
        q = (q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(ref_dev.dtype)
        best_s = best_i = None
        for r0 in range(0, ref_dev.shape[0], tile):
            s = q @ ref_dev[r0 : r0 + tile].T
            ts, ti = s.topk(min(k, s.shape[1]), dim=1)
            ti = ti + r0
            del s
            if best_s is None:
                best_s, best_i = ts, ti
            else:
                best_s, pos = torch.cat([best_s, ts], 1).topk(k, dim=1)
                best_i = torch.cat([best_i, ti], 1).gather(1, pos)
        sims[lo : lo + block] = best_s.float().cpu().numpy()
        idx[lo : lo + block] = best_i.cpu().numpy()
    return sims, idx


FEATURES = ("top1", "support", "share")


def vote(sims: np.ndarray, topics: np.ndarray, k: int, power: float) -> dict[str, np.ndarray]:
    """Similarity-weighted vote among the first k neighbours.

    Weight is max(sim, 0) ** power (0 is a plain majority). Returns the winning
    topic and three confidence features: `top1` the best neighbour's
    similarity, `support` the mean similarity of the winner's neighbours,
    `share` the winner's fraction of the total weight."""
    s = sims[:, :k].astype(np.float64)
    t = topics[:, :k]
    w = np.ones_like(s) if power == 0 else np.maximum(s, 0.0) ** power
    out_label = np.empty(len(s), dtype=np.int64)
    out_support = np.empty(len(s), dtype=np.float64)
    out_share = np.empty(len(s), dtype=np.float64)
    step = 8192
    for lo in range(0, len(s), step):
        sl = slice(lo, lo + step)
        eq = t[sl, :, None] == t[sl, None, :]
        score = (eq * w[sl, None, :]).sum(-1)
        win = score.argmax(1)
        rows = np.arange(len(win))
        label = t[sl][rows, win]
        mask = t[sl] == label[:, None]
        out_label[sl] = label
        out_support[sl] = (s[sl] * mask).sum(1) / mask.sum(1)
        total = w[sl].sum(1)
        out_share[sl] = np.where(total > 0, score[rows, win] / np.where(total > 0, total, 1), 0.0)
    return {"label": out_label, "top1": s[:, 0], "support": out_support, "share": out_share}


def decide(votes: dict[str, np.ndarray], feature: str, threshold: float) -> np.ndarray:
    return np.where(votes[feature] >= threshold, votes["label"], NOISE)


# --- calibration -----------------------------------------------------------


def _topic_sizes(topic: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(topic[topic >= 0], return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def _metrics(pred: np.ndarray, truth: np.ndarray, size_of_truth: np.ndarray) -> dict:
    in_topic = truth >= 0
    noise = ~in_topic
    out = {
        "topic_recall": float((pred[in_topic] == truth[in_topic]).mean()),
        "noise_recall": float((pred[noise] == NOISE).mean()) if noise.any() else None,
        "coverage": float((pred >= 0).mean()),
        "precision": float((pred[pred >= 0] == truth[pred >= 0]).mean()) if (pred >= 0).any() else None,
    }
    out["balanced"] = (
        0.5 * (out["topic_recall"] + out["noise_recall"])
        if out["noise_recall"] is not None
        else out["topic_recall"]
    )
    for lo, hi in SIZE_BINS:
        sel = in_topic & (size_of_truth >= lo) & (size_of_truth < hi)
        out[f"recall_size_{lo}_{hi}"] = float((pred[sel] == truth[sel]).mean()) if sel.any() else None
        out[f"n_size_{lo}_{hi}"] = int(sel.sum())
    return out


def grid_search(
    sims: np.ndarray,
    nbr_topics: np.ndarray,
    truth: np.ndarray,
    size_of_truth: np.ndarray,
    ks=(1, 3, 5, 10, 20, 30, 50),
    powers=(0, 1, 4, 16),
    thresholds=None,
) -> list[dict]:
    thresholds = np.round(np.arange(0.30, 0.951, 0.01), 2) if thresholds is None else thresholds
    rows = []
    for k in ks:
        if k > sims.shape[1]:
            continue
        for power in powers:
            if k == 1 and power != 0:
                continue
            v = vote(sims, nbr_topics, k, power)
            for feature in FEATURES:
                for thr in thresholds:
                    m = _metrics(decide(v, feature, float(thr)), truth, size_of_truth)
                    rows.append({"k": k, "power": power, "feature": feature, "threshold": float(thr), **m})
    return rows


def calibrate(
    ref: Reference,
    heldout_vectors: np.ndarray,
    heldout_topics: np.ndarray,
    sizes: dict[int, int],
    *,
    device: str,
    block: int = DEFAULT_QUERY_BLOCK,
    ref_noise_included: bool = False,
) -> dict:
    """Predict held-out fit posts from `ref` (which must exclude them) and pick
    the knn settings by balanced accuracy over topic and noise posts."""
    import torch

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    ref_dev = torch.from_numpy(ref.vectors).to(device=device, dtype=dtype)
    sims, idx = knn_search(heldout_vectors, ref_dev, MAX_K, block, device)
    nbr_topics = ref.topics[idx]
    size_of_truth = np.array([sizes.get(int(t), 0) for t in heldout_topics])
    grid = grid_search(sims, nbr_topics, heldout_topics, size_of_truth)
    best = max(grid, key=lambda r: (round(r["balanced"], 6), -r["k"], -r["power"], -r["threshold"]))
    return {"best": best, "grid": grid, "n_heldout": int(len(heldout_topics)), "n_reference": len(ref)}


def fp16_agreement(ref: Reference, queries: np.ndarray, k: int, device: str, block: int) -> float:
    """Fraction of queries whose k-th-neighbour index sets match between an fp16
    and an fp32 reference: the check that fp16 storage does not change results."""
    import torch

    r16 = torch.from_numpy(ref.vectors).to(device=device, dtype=torch.float16)
    _, i16 = knn_search(queries, r16, k, block, device)
    del r16
    r32 = torch.from_numpy(ref.vectors).to(device=device, dtype=torch.float32)
    _, i32 = knn_search(queries, r32, k, block, device)
    del r32
    t16, t32 = ref.topics[i16], ref.topics[i32]
    return float((vote(np.ones_like(i16, dtype=np.float32), t16, k, 0)["label"]
                  == vote(np.ones_like(i32, dtype=np.float32), t32, k, 0)["label"]).mean())


@dataclass(frozen=True)
class Settings:
    k: int
    power: float
    feature: str
    threshold: float
    include_noise: bool
    heldout_agreement: float
    noise_rejected: float | None
    balanced: float

    def to_json(self) -> dict:
        return self.__dict__.copy()


def settings_from(best: dict, include_noise: bool) -> Settings:
    return Settings(
        k=best["k"],
        power=best["power"],
        feature=best["feature"],
        threshold=best["threshold"],
        include_noise=include_noise,
        heldout_agreement=round(best["topic_recall"], 4),
        noise_rejected=None if best["noise_recall"] is None else round(best["noise_recall"], 4),
        balanced=round(best["balanced"], 4),
    )


def ensure_calibration(
    cache: Path,
    fit_run: str,
    embedding_files: list[Path],
    fit_table: pa.Table,
    device: str,
    block: int,
    status=lambda stage, **extra: print(stage, extra, flush=True),
    force: bool = False,
    reference: str | None = None,
    fraction: float = 0.05,
    seed: int = 42,
) -> Settings:
    """The settings chosen for this fit, calibrating once per fit run.

    Both reference definitions are measured and the better balanced accuracy
    wins (`reference` forces one); the full grid is kept in the JSON."""
    path = Path(cache) / f"calibration-{fit_run}.json"
    if path.is_file() and not force:
        return Settings(**json.loads(path.read_text())["chosen"])
    status("calibrate", fit_run=fit_run)
    res = run_calibration(embedding_files, fit_table, device=device, fraction=fraction, seed=seed, block=block)
    pick = reference or max(
        ("topics", "topics+noise"), key=lambda r: res[r]["best"]["balanced"] + (r == "topics") * 1e-9
    )
    chosen = res[pick]["settings"]
    _write_json(
        path,
        {"fit_run": fit_run, "reference": pick, "chosen": chosen, "results": res,
         "calibrated_at": datetime.now(timezone.utc).isoformat()},
    )
    return Settings(**chosen)



def params_string(s: Settings, fit_run: str, n_reference: int) -> str:
    return (
        f"method={METHOD},k={s.k},threshold={s.threshold:.2f},fit_run={fit_run},"
        f"heldout_agreement={s.heldout_agreement:.4f},vote_power={s.power:g},"
        f"feature={s.feature},reference={'topics+noise' if s.include_noise else 'topics'},"
        f"n_reference={n_reference},backend=torch_fp16"
    )


def holdout_split(fit_table: pa.Table, fraction: float, seed: int) -> np.ndarray:
    """Boolean mask of fit rows to hold out. Stratified by topic so small topics
    keep members in the reference; noise rows are held out at the same rate."""
    rng = np.random.default_rng(seed)
    topic = fit_table.column("topic").to_numpy()
    held = np.zeros(len(topic), dtype=bool)
    for t in np.unique(topic):
        rows = np.flatnonzero(topic == t)
        n = int(round(len(rows) * fraction))
        if t >= 0:
            n = min(n, len(rows) - 1)
        if n > 0:
            held[rng.choice(rows, size=n, replace=False)] = True
    return held


def run_calibration(
    embedding_files: list[Path],
    fit_table: pa.Table,
    *,
    device: str,
    fraction: float = 0.05,
    seed: int = 42,
    block: int = DEFAULT_QUERY_BLOCK,
) -> dict:
    """Hold out `fraction` of the fit, predict it from the rest, return the
    grid for both reference definitions and the chosen settings."""
    held = holdout_split(fit_table, fraction, seed)
    ids = fit_table.column("platform_post_id").to_pylist()
    topic = fit_table.column("topic").to_numpy()
    sizes = _topic_sizes(topic)
    all_ids = pa.array(ids)
    plan = plan_rows(embedding_files, wanted=all_ids)
    got: dict[str, np.ndarray] = {}
    for batch_ids, vecs in read_planned(plan):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        v16 = (vecs / np.maximum(norms, 1e-12)).astype(np.float16)
        for i, vec in zip(batch_ids, v16):
            got[i] = vec
    present = np.array([i in got for i in ids])
    held_rows = np.flatnonzero(held & present)
    train_rows = np.flatnonzero(~held & present)
    vec_of = lambda rows: np.stack([got[ids[r]] for r in rows])  # noqa: E731
    held_vectors = vec_of(held_rows).astype(np.float32)
    held_topics = topic[held_rows]

    results = {}
    for include_noise in (False, True):
        keep = train_rows if include_noise else train_rows[topic[train_rows] >= 0]
        ref = Reference(vec_of(keep), topic[keep].astype(np.int32))
        res = calibrate(ref, held_vectors, held_topics, sizes, device=device, block=block)
        res["settings"] = settings_from(res["best"], include_noise).to_json()
        res["include_noise"] = include_noise
        results[("topics+noise" if include_noise else "topics")] = res
    sample = held_vectors[: min(2000, len(held_vectors))]
    ref_topics = Reference(vec_of(train_rows[topic[train_rows] >= 0]), topic[train_rows[topic[train_rows] >= 0]].astype(np.int32))
    results["fp16_vs_fp32_label_match"] = fp16_agreement(
        ref_topics, sample, results["topics"]["best"]["k"], device, block
    )
    results["fraction"] = fraction
    results["seed"] = seed
    results["n_topics"] = len(sizes)
    results["n_fit_posts"] = len(ids)
    results["n_fit_with_embedding"] = int(present.sum())
    return results


# --- the run ---------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.rename(path)


def chunk_table(ids: list[str], topics: np.ndarray, params: str, model: str, at: datetime) -> pa.Table:
    n = len(ids)
    return pa.table(
        {
            "platform_post_id": pa.array(ids, pa.string()),
            "topic": pa.array(topics.astype(np.int64), pa.int64()),
            "model": pa.array([slug(model)] * n, pa.string()),
            "params": pa.array([params] * n, pa.string()),
            "assigned_at": pa.array([at] * n, pa.timestamp("us", tz="UTC")),
        },
        schema=_TOPICS_SCHEMA,
    )


def assign(
    store,
    cache: Path,
    *,
    run_id: str,
    platform: str = PLATFORM,
    model: str = MODEL,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    block: int = DEFAULT_QUERY_BLOCK,
    limit: int | None = None,
    dry_run: bool = False,
    status_file: Path | None = None,
    settings: Settings | None = None,
) -> dict:
    """One assignment pass. Returns the marker dict that was written."""
    import torch

    t0 = time.time()
    cache = Path(cache)

    def status(stage: str, **extra) -> None:
        print(f"[{time.time() - t0:7.1f}s] {stage} {extra or ''}", flush=True)
        if status_file:
            _write_json(status_file, {"stage": stage, "run_id": run_id, "elapsed_s": time.time() - t0,
                                      "updated_at": datetime.now(timezone.utc).isoformat(), **extra})

    status("mirror")
    emb_files = mirror(store, embeddings_prefix(platform, model), cache / "mirror")
    topic_files = mirror(store, topics_prefix(platform, model), cache / "mirror")
    state = fit_state(topic_files)
    fit_table = read_columns(state.fit_files, ["platform_post_id", "topic"])
    status("fit", fit_run=state.fit_run, fit_rows=fit_table.num_rows, embeddings_files=len(emb_files))

    if settings is None:
        settings = ensure_calibration(cache, state.fit_run, emb_files, fit_table, device, block, status)
    ref = load_reference(cache, state.fit_run, emb_files, fit_table, settings.include_noise)
    status("reference", n_reference=len(ref))

    assigned = read_columns(state.fit_files + state.knn_files, ["platform_post_id"]).column(0).combine_chunks()
    plan = plan_rows(emb_files, wanted=None, exclude=assigned)
    pending_total = sum(len(r) for _, r in plan)
    to_assign = pending_total
    if limit is not None and pending_total > limit:
        plan, to_assign = _truncate(plan, limit)
    status("pending", pending=pending_total, to_assign=to_assign, already_assigned=len(assigned))

    params = params_string(settings, state.fit_run, len(ref))
    ref_dev = torch.from_numpy(ref.vectors).to(device=device, dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    puts = set(store.put_keys()) if hasattr(store, "put_keys") else set()
    chunks: list[dict] = []
    out_dir = cache / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"topic": 0, "noise": 0}
    cur_block = block
    buf_ids: list[str] = []
    buf_vecs: list[np.ndarray] = []
    exhausted = False

    def flush(rows: int | None = None) -> bool:
        """Assign and write the first `rows` buffered posts (all, by default)."""
        nonlocal cur_block, buf_ids, buf_vecs, exhausted
        if not buf_ids:
            return True
        vecs = np.concatenate(buf_vecs)
        take = len(buf_ids) if rows is None else rows
        ids, queries = buf_ids[:take], vecs[:take]
        buf_ids, buf_vecs = buf_ids[take:], [vecs[take:]]
        idx = len(chunks)
        key = chunk_key(run_id, idx, platform, model)
        if not dry_run and puts and key not in puts:
            exhausted = True
            return False
        while True:
            try:
                sims, nbr = knn_search(queries, ref_dev, settings.k, cur_block, device)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_block <= 64:
                    raise
                cur_block //= 2
                status("oom_retry", query_block=cur_block)
        v = vote(sims, ref.topics[nbr], settings.k, settings.power)
        labels = decide(v, settings.feature, settings.threshold)
        table = chunk_table(ids, labels, params, model, datetime.now(timezone.utc))
        local = out_dir / f"c{idx:04d}.parquet"
        pq.write_table(table, local, compression="zstd")
        if not dry_run:
            store.put(key, local)
        chunks.append({"key": key, "rows": table.num_rows, "bytes": local.stat().st_size})
        stats["topic"] += int((labels >= 0).sum())
        stats["noise"] += int((labels < 0).sum())
        status("chunk", chunk=idx, rows=table.num_rows, assigned=sum(c["rows"] for c in chunks),
               to_assign=to_assign)
        return True

    ok = True
    for ids, vecs in read_planned(plan):
        buf_ids += ids
        buf_vecs.append(vecs)
        while ok and len(buf_ids) >= chunk_rows:
            ok = flush(chunk_rows)
        if not ok:
            break
    while ok and buf_ids:
        ok = flush(min(chunk_rows, len(buf_ids)))

    assigned_rows = sum(c["rows"] for c in chunks)
    readback = _readback(store, chunks, cache, dry_run)
    complete = ok and not exhausted and assigned_rows == pending_total
    marker = {
        "status": "complete" if complete else "partial",
        "run_id": run_id,
        "platform": platform,
        "model": slug(model),
        "fit_run": state.fit_run,
        "params": params,
        "settings": settings.to_json(),
        "pending_at_start": pending_total,
        "assigned": assigned_rows,
        "topic_ge0": stats["topic"],
        "noise": stats["noise"],
        "stale_knn_files_ignored": len(state.stale_knn_files),
        "chunks": chunks,
        "readback_rows": readback["rows"],
        "readback_distinct": readback["distinct"],
        "dry_run": dry_run,
        "limit": limit,
        "elapsed_s": round(time.time() - t0, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.startswith("cuda") else None,
        "query_block": cur_block,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    mk = marker_key(run_id, platform, model)
    mpath = out_dir / "marker.json"
    _write_json(mpath, marker)
    if not dry_run and (not puts or mk in puts):
        store.put(mk, mpath)
    status("done", **{k: marker[k] for k in ("status", "assigned", "pending_at_start", "peak_vram_gib")})
    return marker


def _truncate(plan, limit: int):
    out, left = [], limit
    for path, rows in plan:
        if left <= 0:
            break
        out.append((path, rows[:left]))
        left -= min(len(rows), left)
    return out, limit - left


def _readback(store, chunks: list[dict], cache: Path, dry_run: bool) -> dict:
    """Re-read the chunks that were written and count what is actually there."""
    if dry_run or not chunks:
        return {"rows": 0, "distinct": 0}
    tmp = Path(tempfile.mkdtemp(dir=cache))
    ids: list[pa.Array] = []
    rows = 0
    for i, c in enumerate(chunks):
        dest = tmp / f"{i}.parquet"
        store.get(c["key"], dest)
        t = _read(dest, ["platform_post_id"])
        if t.num_rows != c["rows"]:
            raise RuntimeError(f"{c['key']}: read back {t.num_rows} rows, wrote {c['rows']}")
        rows += t.num_rows
        ids.append(t.column(0).combine_chunks())
        dest.unlink()
    tmp.rmdir()
    distinct = len(pc.unique(pa.concat_arrays(ids)))
    return {"rows": rows, "distinct": distinct}


# --- cli -------------------------------------------------------------------


def _device(arg: str) -> str:
    if arg != "auto":
        return arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _store(args):
    if args.manifest:
        return ManifestStore.load(Path(args.manifest))
    return S3Store()


def cmd_calibrate(args) -> int:
    store, cache, device = _store(args), Path(args.cache), _device(args.device)
    emb_files = mirror(store, embeddings_prefix(args.platform, args.model), cache / "mirror")
    topic_files = mirror(store, topics_prefix(args.platform, args.model), cache / "mirror")
    state = fit_state(topic_files)
    fit_table = read_columns(state.fit_files, ["platform_post_id", "topic"])
    chosen = ensure_calibration(
        cache, state.fit_run, emb_files, fit_table, device, args.query_block,
        force=True, reference=args.reference, fraction=args.holdout, seed=args.seed,
    )
    print(json.dumps({"fit_run": state.fit_run, "chosen": chosen.to_json()}, indent=1))
    return 0


def cmd_run(args) -> int:
    import torch

    store, cache, device = _store(args), Path(args.cache), _device(args.device)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        marker = assign(
            store, cache, run_id=run_id, platform=args.platform, model=args.model, device=device,
            chunk_rows=args.chunk_rows, block=args.query_block, limit=args.limit, dry_run=args.dry_run,
            status_file=Path(args.status_file) if args.status_file else None,
        )
    except torch.cuda.OutOfMemoryError:
        print("GPU out of memory at the smallest query block", file=sys.stderr)
        return EXIT_OOM
    return 0 if marker["status"] == "complete" else EXIT_MORE_WORK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", help="signed-url manifest; omit to use R2_* env keys")
        p.add_argument("--cache", required=True, help="local mirror and reference cache")
        p.add_argument("--platform", default=PLATFORM)
        p.add_argument("--model", default=MODEL)
        p.add_argument("--device", default="auto")
        p.add_argument("--query-block", type=int, default=DEFAULT_QUERY_BLOCK)
    sub.choices["calibrate"].add_argument("--holdout", type=float, default=0.05)
    sub.choices["calibrate"].add_argument("--seed", type=int, default=42)
    sub.choices["calibrate"].add_argument("--reference", choices=("topics", "topics+noise"),
                                          help="force a reference definition; default is the better one")
    r = sub.choices["run"]
    r.add_argument("--run-id")
    r.add_argument("--limit", type=int)
    r.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    r.add_argument("--dry-run", action="store_true", help="write local files only, no PUT")
    r.add_argument("--status-file")
    args = ap.parse_args(argv)
    return {"calibrate": cmd_calibrate, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
