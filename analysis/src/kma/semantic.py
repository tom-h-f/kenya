"""Semantic layer: embed posts, persist to R2, and search by meaning.

Embeddings are expensive so they are persisted to the R2 `embeddings/` prefix
and built incrementally (only posts not already embedded for a given model).
Topic assignments are persisted too, to `topics/`: the UMAP fit is a batch job
(see `persist_topics`), and readers take the result via `load_topics`. Search
computes live.

    from kma.db import connect
    from kma.semantic import embed_new, search
    con = connect()
    embed_new(con, limit=500)                 # first slice / validation
    search(con, "IEBC cannot be trusted", k=10)
"""

from __future__ import annotations

import gc
import re
from datetime import datetime, timezone

import duckdb
import pyarrow as pa

from kma.db import (
    BUCKET,
    embeddings_source,
    pending_posts,
    posts_source,
    topics_source,
)

_CLEAN = re.compile(r"https?://\S+|@\w+|#\w+|[^\w\s]", re.UNICODE)

# Swahili / Sheng function words + Twitter noise, on top of sklearn's English list.
# c-TF-IDF labels are dominated by these otherwise (na, ya, wa, ni ...).
_SWAHILI_STOP = {
    "na", "ya", "wa", "ni", "kwa", "la", "za", "ku", "cha", "vya", "wa", "si",
    "kama", "lakini", "ama", "au", "hata", "bado", "sasa", "tena", "pia", "tu",
    "sana", "katika", "kwenye", "hivyo", "ndio", "ndipo", "huyu", "hii", "hizi",
    "hao", "wale", "wote", "kila", "moja", "watu", "mtu", "yako", "yangu", "yake",
    "wako", "kwani", "ili", "juu", "chini", "mbele", "nyuma", "huko", "hapa",
    "est", "quoi", "les", "des", "une", "pour", "que", "amp", "https", "http",
}


def _stopwords() -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return list(ENGLISH_STOP_WORDS | _SWAHILI_STOP)


STOPWORDS = _stopwords()

MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DIM = 768

_MODELS: dict[str, object] = {}


def _slug(model: str) -> str:
    return model.split("/")[-1]


def _model(model: str):
    if model not in _MODELS:
        from sentence_transformers import SentenceTransformer

        _MODELS[model] = SentenceTransformer(model)
    return _MODELS[model]


def _embedded_ids(con: duckdb.DuckDBPyConnection, platform: str, model: str) -> set[str]:
    try:
        rel = con.sql(
            f"SELECT DISTINCT platform_post_id FROM {embeddings_source(platform, _slug(model))}"
        )
    except duckdb.Error:
        return set()  # no embeddings written yet
    return set(rel.df()["platform_post_id"].tolist())


def _pending(con: duckdb.DuckDBPyConnection, platform: str, model: str, limit: int | None):
    return pending_posts(
        con, embeddings_source(platform, _slug(model)), platform, limit
    )


def embed_new(
    con: duckdb.DuckDBPyConnection,
    model: str = MODEL,
    platform: str = "x",
    limit: int | None = None,
    batch_size: int = 64,
) -> int:
    """Embed posts not yet embedded for `model`, write one Parquet run to R2.
    Returns the number embedded."""
    df = _pending(con, platform, model, limit)
    if df.empty:
        return 0
    vecs = _model(model).encode(
        df["text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    now = datetime.now(timezone.utc)
    table = pa.table(
        {
            "platform_post_id": df["platform_post_id"].tolist(),
            "model": [_slug(model)] * len(df),
            "dim": [DIM] * len(df),
            "embedding": [v.tolist() for v in vecs],
            "embedded_at": [now] * len(df),
        }
    )
    key = (
        f"embeddings/platform={platform}/model={_slug(model)}"
        f"/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    )
    con.register("_emb_buf", table)
    try:
        con.execute(
            f"COPY _emb_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_emb_buf")
    return len(df)


def _load_vectors(con: duckdb.DuckDBPyConnection, platform: str, model: str) -> None:
    con.execute("INSTALL vss; LOAD vss;")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _emb AS
        SELECT platform_post_id, CAST(embedding AS FLOAT[{DIM}]) AS vec FROM (
            SELECT * FROM {embeddings_source(platform, _slug(model))}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY embedded_at DESC
            ) = 1
        )
        """
    )


def search(
    con: duckdb.DuckDBPyConnection,
    query: str,
    k: int = 10,
    platform: str = "x",
    model: str = MODEL,
):
    """k nearest posts to `query` by cosine similarity over persisted embeddings."""
    _load_vectors(con, platform, model)
    q = _model(model).encode([query], normalize_embeddings=True)[0].tolist()
    return con.execute(
        f"""
        WITH lp AS (
            SELECT * FROM {posts_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        SELECT e.platform_post_id, p.author_handle, p.text,
               array_cosine_similarity(e.vec, CAST(? AS FLOAT[{DIM}])) AS sim
        FROM _emb e JOIN lp p USING (platform_post_id)
        ORDER BY sim DESC LIMIT {k}
        """,
        [q],
    ).df()


def _embeddings_with_text(con: duckdb.DuckDBPyConnection, platform: str, model: str):
    return con.sql(
        f"""
        WITH e AS (
            SELECT * FROM {embeddings_source(platform, _slug(model))}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY embedded_at DESC
            ) = 1
        ), lp AS (
            SELECT * FROM {posts_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        SELECT e.platform_post_id, e.embedding, p.author_handle, p.text,
               p.created_at, p.lang
        FROM e JOIN lp p USING (platform_post_id)
        ORDER BY e.platform_post_id
        """
    ).df()


def assign_topics(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    model: str = MODEL,
    min_cluster_size: int = 25,
    n_neighbors: int = 15,
    n_components: int = 5,
    random_state: int | None = 42,
    backend: str = "auto",
):
    """Cluster embeddings into narrative topics via UMAP -> HDBSCAN (the BERTopic
    recipe: UMAP preserves local structure far better than PCA, so most posts land
    in a cluster instead of noise). Returns the per-post frame with a `topic`
    column; topic -1 is unclustered.

    `backend` picks the implementation: ``"cpu"`` is umap-learn + sklearn,
    ``"gpu"`` is RAPIDS cuML, ``"auto"`` takes the GPU when cuML imports. These
    are DIFFERENT implementations, not one algorithm at two speeds - assignments
    from the two backends are not comparable, so do not mix them in one series.

    A set `random_state` makes the CPU fit reproducible.
    """
    import numpy as np

    df = _embeddings_with_text(con, platform, model)
    if len(df) < max(min_cluster_size, n_neighbors + 1):
        df["topic"] = -1
        return df
    x = np.asarray(df["embedding"].tolist(), dtype="float32")
    # The vector column is dead weight from here on and no caller reads it back,
    # but as pandas objects it is the largest thing in the process, so holding it
    # through the fit costs tens of GB for nothing.
    df = df.drop(columns=["embedding"])
    gc.collect()
    df["topic"] = _cluster_vectors(
        x, min_cluster_size, n_neighbors, n_components, random_state, backend
    )
    return df


def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError(f"unknown backend {backend!r} (expected auto|cpu|gpu)")
    if backend != "auto":
        return backend
    try:
        import cuml  # noqa: F401
    except Exception:
        return "cpu"
    return "gpu"


def _cluster_vectors(
    x,
    min_cluster_size: int,
    n_neighbors: int,
    n_components: int,
    random_state: int | None,
    backend: str,
):
    """UMAP -> HDBSCAN on a float32 matrix, returning integer topic labels."""
    import numpy as np

    resolved = _resolve_backend(backend)
    if resolved == "gpu":
        from cuml.cluster import HDBSCAN
        from cuml.manifold import UMAP

        reduced = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
            random_state=random_state,
            verbose=True,
        ).fit_transform(x)
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=1
        ).fit_predict(reduced)
        return np.asarray(labels).astype("int64")

    from sklearn.cluster import HDBSCAN
    from umap import UMAP

    reduced = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    ).fit_transform(x)
    return HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=1, metric="euclidean"
    ).fit_predict(reduced)


def persist_topics(
    con: duckdb.DuckDBPyConnection,
    platform: str = "x",
    model: str = MODEL,
    min_cluster_size: int = 25,
    n_neighbors: int = 15,
    n_components: int = 5,
    random_state: int | None = 42,
    backend: str = "auto",
) -> int:
    """Run `assign_topics` and write the assignments to the R2 `topics/` prefix.

    UMAP over the whole embedding corpus is hours of work and tens of GB of
    peak RSS, far past what a notebook should do live. This is the batch half:
    run it on Modal, then `load_topics` serves every reader from the persisted
    result.
    """
    df = assign_topics(
        con,
        platform=platform,
        model=model,
        min_cluster_size=min_cluster_size,
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=random_state,
        backend=backend,
    )
    if not len(df):
        return 0
    # Only the id and its cluster get written. Holding `embedding` past the fit
    # is what killed the first Modal run: 658k rows of 768 floats live as pandas
    # objects, which dwarfs everything else in the frame.
    ids = df["platform_post_id"].tolist()
    topics = [int(t) for t in df["topic"]]
    del df
    gc.collect()
    now = datetime.now(timezone.utc)
    # `backend` belongs in the provenance: cuML and umap-learn are different
    # implementations, so runs from the two are not comparable.
    params = (
        f"min_cluster_size={min_cluster_size},n_neighbors={n_neighbors},"
        f"n_components={n_components},random_state={random_state},"
        f"backend={_resolve_backend(backend)}"
    )
    n = len(ids)
    table = pa.table(
        {
            "platform_post_id": ids,
            "topic": topics,
            "model": [_slug(model)] * n,
            "params": [params] * n,
            "assigned_at": [now] * n,
        }
    )
    key = (
        f"topics/platform={platform}/model={_slug(model)}"
        f"/dt={now:%Y-%m-%d}/run={now:%Y%m%dT%H%M%SZ}.parquet"
    )
    con.register("_topic_buf", table)
    try:
        con.execute(
            f"COPY _topic_buf TO 'r2://{BUCKET}/{key}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.unregister("_topic_buf")
    return n


def load_topics(
    con: duckdb.DuckDBPyConnection, platform: str = "x", model: str = MODEL
):
    """Persisted topic assignments joined back to post text.

    Shaped exactly like `assign_topics` output minus the embedding vector, so
    `topic_summary` consumes it unchanged.
    """
    try:
        con.sql(f"SELECT 1 FROM {topics_source(platform, _slug(model))} LIMIT 1").fetchall()
    except duckdb.IOException as exc:
        raise RuntimeError(
            f"No persisted topics for platform={platform} model={_slug(model)}. "
            "Build them first: "
            "`modal run --detach modal_backfill.py --topic --spawn` (from analysis/)."
        ) from exc
    return con.sql(
        f"""
        WITH t AS (
            SELECT * FROM {topics_source(platform, _slug(model))}
            QUALIFY row_number() OVER (
                PARTITION BY platform_post_id ORDER BY assigned_at DESC
            ) = 1
        ), lp AS (
            SELECT * FROM {posts_source(platform)}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        SELECT t.platform_post_id, t.topic, p.author_handle, p.text,
               p.created_at, p.lang
        FROM t JOIN lp p USING (platform_post_id)
        ORDER BY t.platform_post_id
        """
    ).df()


def _short_name_from_terms(terms: str, max_words: int = 3) -> str:
    words = [t.strip() for t in terms.split(", ") if t.strip()]
    return " ".join(words[:max_words])


def _dedupe_topic_names(names: list[str], term_rows: list[str], max_words: int = 3) -> list[str]:
    used: set[str] = set()
    out: list[str] = []
    for name, terms in zip(names, term_rows):
        candidate = name
        extra = [t.strip() for t in terms.split(", ") if t.strip()]
        n = max_words + 1
        while candidate in used and n <= len(extra):
            candidate = " ".join(extra[:n])
            n += 1
        if candidate in used:
            candidate = f"{candidate} alt"
        used.add(candidate)
        out.append(candidate)
    return out


def topic_summary(df, top_terms: int = 8, max_words: int = 3):
    """Per-topic size, c-TF-IDF top terms (distinctive per cluster), short
    display name (1-3 words), label ``name (n=size)``, and a sample post.
    Input is `assign_topics` output."""
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer

    clusters = sorted(t for t in df["topic"].unique() if t != -1)
    cols = ["topic", "size", "terms", "name", "label", "sample"]
    if not clusters:
        return pd.DataFrame(columns=cols)
    docs = [
        " ".join(_CLEAN.sub(" ", s.lower()) for s in df.loc[df["topic"] == c, "text"])
        for c in clusters
    ]
    cv = CountVectorizer(stop_words=STOPWORDS, min_df=2, token_pattern=r"[a-z]{3,}")
    counts = cv.fit_transform(docs).toarray()
    # c-TF-IDF (BERTopic): term freq per class, discounted by prevalence across classes.
    words_per_class = np.maximum(counts.sum(axis=1, keepdims=True), 1)
    tf = counts / words_per_class
    idf = np.log(1 + words_per_class.mean() / np.maximum(counts.sum(axis=0), 1))
    ctfidf = tf * idf
    vocab = cv.get_feature_names_out()
    rows = [
        {
            "topic": int(c),
            "size": int((df["topic"] == c).sum()),
            "terms": ", ".join(vocab[j] for j in np.argsort(ctfidf[i])[::-1][:top_terms]),
            "sample": df.loc[df["topic"] == c, "text"].iloc[0][:100],
        }
        for i, c in enumerate(clusters)
    ]
    raw_names = [_short_name_from_terms(r["terms"], max_words) or "topic" for r in rows]
    names = _dedupe_topic_names(raw_names, [r["terms"] for r in rows], max_words)
    for row, name in zip(rows, names):
        row["name"] = name
        row["label"] = f"{name} (n={row['size']})"
    return pd.DataFrame(rows, columns=cols).sort_values(
        "size", ascending=False, ignore_index=True
    )


def with_topic_names(
    df: pd.DataFrame,
    names: pd.DataFrame,
    *,
    id_col: str = "topic",
    drop_id: bool = False,
) -> pd.DataFrame:
    """Attach topic name + label columns keyed on `id_col` (default ``topic``)."""
    import pandas as pd

    out = df.merge(
        names[["topic", "name", "label"]].rename(columns={"topic": id_col}),
        on=id_col,
        how="left",
    )
    if drop_id:
        out = out.drop(columns=[id_col])
    return out
