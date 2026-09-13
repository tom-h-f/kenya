"""Is this post about Kenya: the classifier's call, with the keyword gate behind it.

`kma.measure.domain_bucket` is two regexes. Measured against Tom's blind labels
it is precise and leaky - precision 0.971, recall 0.655 corpus-weighted - and
its misses are plain Kenyan politics that happens to avoid the anchor
vocabulary. The classifier trained on 2026-09-12 scores 0.918 / 1.000 on the
same labels, so this module is what readers should ask.

    from kma import relevance
    relevance.buckets(con, posts)        # kenya / offdomain / ambiguous per row

Two properties worth knowing before using it:

- **The gate is the fallback, not a rival.** Scores are persisted per post, so
  anything collected since the last scoring pass has none, and those rows fall
  back to `domain_bucket`. A mixed answer is the normal case.
- **The threshold is a reader's choice.** Probabilities are persisted, the cut
  is not tuned, and 0.5 is only where the evaluation was run. Near-empty posts
  ride on their @mentions ("@SomeForcePolice [emoji]" scores 0.80), so a pass
  that cares about precision should raise it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb
import pandas as pd

from kma.bench import _r2_client
from kma.db import BUCKET, relevance_source

log = logging.getLogger("kma")

# 2026-09-13 retrains 09-12 on mention-stripped text, which is what `score_texts`
# serves: trained with mentions and served without them, the 09-12 model scored
# precision 0.918 / recall 1.000 against Tom's labels, and this one 0.997 /
# 0.980. Scores carry their model, so the two never mix.
MODEL = os.getenv("KMA_RELEVANCE_MODEL", "kenya-relevance-afroxlmr-2026-09-13")
THRESHOLD = float(os.getenv("KMA_RELEVANCE_THRESHOLD", "0.5"))

# The weights live in R2 beside the data, not on the HF Hub: there is no HF
# token on tac2 or in .env, and every host that scores already holds R2
# credentials. The slug is on every score row, so a file and a score can always
# be matched up.
MODEL_PREFIX = "models/relevance"
CACHE = Path(os.getenv("KMA_MODEL_CACHE", Path.home() / ".cache" / "kma-models"))


def model_key(name: str = MODEL) -> str:
    return f"{MODEL_PREFIX}/{name}"


def _transfer_client():
    """An R2 client that survives a flaky link.

    A 2.2 GB checkpoint is ~60 multipart parts, and one dropped part fails the
    whole upload: this link dropped three transfers in a row on 2026-09-13.
    Adaptive retries and a single-threaded transfer trade speed for finishing.
    """
    from botocore.config import Config

    return _r2_client(Config(retries={"max_attempts": 10, "mode": "adaptive"},
                             connect_timeout=30, read_timeout=120, max_pool_connections=4))


def publish_model(source: Path, name: str = MODEL, bucket: str = BUCKET) -> list[str]:
    """Upload a trained model directory to R2. Returns the keys written."""
    from boto3.s3.transfer import TransferConfig

    client = _transfer_client()
    transfer = TransferConfig(multipart_chunksize=16 * 2**20, max_concurrency=2,
                              num_download_attempts=10, use_threads=True)
    written = []
    for path in sorted(p for p in Path(source).iterdir() if p.is_file()):
        key = f"{model_key(name)}/{path.name}"
        client.upload_file(str(path), bucket, key, Config=transfer)
        written.append(key)
        log.info("uploaded %s (%.1f MiB)", key, path.stat().st_size / 2**20)
    return written


def fetch_model(name: str = MODEL, cache: Path | None = None, bucket: str = BUCKET) -> Path:
    """The model directory, downloaded from R2 once and cached by size.

    Any host with R2 credentials can score; nothing needs the training box.
    """
    target = Path(cache or CACHE) / name
    target.mkdir(parents=True, exist_ok=True)
    client = _transfer_client()
    prefix = model_key(name)
    listing = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/").get("Contents", [])
    if not listing:
        raise FileNotFoundError(f"no model at r2://{bucket}/{prefix}/; publish_model first")
    for obj in listing:
        local = target / obj["Key"].split("/")[-1]
        if local.exists() and local.stat().st_size == obj["Size"]:
            continue
        log.info("downloading %s (%.1f MiB)", obj["Key"], obj["Size"] / 2**20)
        client.download_file(bucket, obj["Key"], str(local))
    return target


def score_texts(model_dir: Path, texts: list[str], batch: int = 256, max_len: int = 128) -> "np.ndarray":
    """`p_kenya` per text. CUDA when there is one, CPU otherwise.

    Batches are sorted by length so padding does not dominate, and scores come
    back in input order. Mentions are stripped first: measured on 1.28M posts,
    a post with no words of its own rides on the handles it replies to
    ("@SomeForcePolice [emoji]" scored 0.80), which matters once a score
    decides what the collector chases.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from kma.coord2 import _MENTION

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()

    cleaned = [_MENTION.sub(" ", t or "") for t in texts]
    order = np.argsort([len(t) for t in cleaned])[::-1]
    out = np.empty(len(cleaned), dtype=np.float32)
    for start in range(0, len(order), batch):
        index = order[start : start + batch]
        encoded = tokenizer([cleaned[k] for k in index], truncation=True, max_length=max_len,
                            padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits.float()
        out[index] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return out


def write_scores(
    con: duckdb.DuckDBPyConnection,
    scored: pd.DataFrame,
    model: str = MODEL,
    platform: str = "x",
    bucket: str = BUCKET,
) -> str:
    """One run of scores to the R2 `relevance/` prefix. Returns the key.

    `scored` needs `platform_post_id` and `p_kenya`. The same row shape every
    reader expects, written the way `semantic.embed_new` writes embeddings.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    buf = scored[["platform_post_id", "p_kenya"]].assign(model=model, scored_at=now)
    key = (f"relevance/platform={platform}/model={model}"
           f"/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet")
    con.register("_rel_out", buf)
    try:
        con.execute(f"COPY _rel_out TO 'r2://{bucket}/{key}' (FORMAT parquet, COMPRESSION zstd)")
    finally:
        con.unregister("_rel_out")
    log.info("wrote %s (%d posts)", key, len(buf))
    return key


def latest_scores_cte(platform: str = "x", model: str = MODEL) -> str:
    """One score per post, the most recent, as a SELECT for a caller's CTE."""
    return f"""
        SELECT platform_post_id, p_kenya FROM {relevance_source(platform, model)}
        QUALIFY row_number() OVER (
            PARTITION BY platform_post_id ORDER BY scored_at DESC) = 1
    """


def scores(
    con: duckdb.DuckDBPyConnection,
    post_ids,
    platform: str = "x",
    model: str = MODEL,
) -> pd.Series:
    """`p_kenya` for the posts that have one, indexed by post id.

    An empty result rather than an error when nothing is persisted yet: every
    caller has the gate to fall back to.
    """
    ids = pd.DataFrame({"platform_post_id": pd.Series(list(post_ids), dtype="object").astype(str)})
    if ids.empty:
        return pd.Series(dtype="float64", name="p_kenya")
    con.register("_rel_wanted", ids.drop_duplicates())
    try:
        got = con.sql(
            f"""
            WITH s AS ({latest_scores_cte(platform, model)})
            SELECT CAST(s.platform_post_id AS VARCHAR) AS platform_post_id, s.p_kenya
            FROM s SEMI JOIN _rel_wanted w ON w.platform_post_id = s.platform_post_id
            """
        ).df()
    except duckdb.Error:
        log.warning("relevance: no scores for model %s; falling back to the keyword gate", model)
        return pd.Series(dtype="float64", name="p_kenya")
    finally:
        con.unregister("_rel_wanted")
    return got.set_index("platform_post_id")["p_kenya"]


def buckets(
    con: duckdb.DuckDBPyConnection,
    posts: pd.DataFrame,
    *,
    post_id: str = "post_id",
    text: str = "text",
    platform: str = "x",
    model: str = MODEL,
    threshold: float = THRESHOLD,
) -> pd.Series:
    """`kenya` / `offdomain` / `ambiguous` per row: the model where it scored
    the post, `measure.domain_bucket` where it did not.

    The model answers a binary question, so it never returns `ambiguous` - that
    value only survives on posts it has not seen, which is what a reader should
    take it to mean.
    """
    from kma import measure

    gate = posts[text].map(measure.domain_bucket)
    if posts.empty:
        return gate

    p = posts[post_id].astype(str).map(scores(con, posts[post_id], platform, model))
    learned = p.map(lambda v: None if pd.isna(v) else ("kenya" if v >= threshold else "offdomain"))
    out = learned.fillna(gate)
    log.info("relevance: %d of %d posts scored by %s", int(p.notna().sum()), len(posts), model)
    return out.rename("domain")
