"""The text trace's vector load: latest vector per post, narrow dtype, coverage."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from kma import coord2, textsim_run

TEXT = "electoral commission rigged tallying centres nationwide again"


def _vec(*head: float) -> list[float]:
    v = np.zeros(768, dtype=np.float32)
    v[: len(head)] = head
    return (v / np.linalg.norm(v)).astype(np.float64).tolist()


@pytest.fixture
def manifest(tmp_path):
    con = duckdb.connect()
    posts = pd.DataFrame({
        "author_id": ["u1", "u2", "u3", "u4"],
        "platform_post_id": ["p1", "p2", "p3", "p4"],
        "created_at": pd.to_datetime(["2026-09-20 10:00", "2026-09-20 10:05",
                                       "2026-09-20 10:10", "2026-09-20 10:15"], utc=True),
        "is_repost": [False] * 4,
        "repost_of_id": [None] * 4,
        "text": [f"{TEXT} one", f"{TEXT} one", f"{TEXT} one", f"{TEXT} one"],
        "hashtags": [None] * 4,
        "urls": [None] * 4,
        "collected_at": pd.to_datetime(["2026-09-20 11:00"] * 4, utc=True),
    })
    posts.to_parquet(tmp_path / "posts.parquet")
    embeddings = pd.DataFrame({
        "platform_post_id": ["p1", "p2", "p2", "p3"],
        "embedded_at": pd.to_datetime(["2026-09-21", "2026-09-21", "2026-09-22", "2026-09-21"], utc=True),
        # p2 was embedded twice; the OLDER row points the other way, so the
        # trace only links p1 and p2 if the latest row is the one it keeps.
        "embedding": [_vec(1.0), _vec(0.0, 1.0), _vec(1.0), _vec(0.0, 0.0, 1.0)],
    })
    embeddings.to_parquet(tmp_path / "embeddings.parquet")
    return con, pd.DataFrame([
        {"prefix": "posts", "path": str(tmp_path / "posts.parquet")},
        {"prefix": "embeddings", "path": str(tmp_path / "embeddings.parquet")},
    ])


def test_the_latest_vector_per_post_is_the_one_compared(manifest):
    con, frame = manifest
    edges, info = textsim_run.build(
        con, "snap", manifest=frame, persist=False, threshold=0.9, overlap=0.0,
        similarity_for=coord2.gpu_cosine_pairs,
    )
    assert edges[["source", "target"]].values.tolist() == [["u1", "u2"]]
    assert edges["weight"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert info["rows"] == 3 and info["eligible"] == 4


def test_coverage_is_reported_as_rows_over_eligible(manifest):
    con, frame = manifest
    _, info = textsim_run.build(
        con, "snap", manifest=frame, persist=False, threshold=0.9, overlap=0.0,
        similarity_for=coord2.gpu_cosine_pairs,
    )
    assert info["embedding_coverage"] == pytest.approx(0.75)
