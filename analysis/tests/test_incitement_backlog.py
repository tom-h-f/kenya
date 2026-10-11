"""kma.incitement_backlog: ordering, chunk layout, schema and crash resume.

No R2, no GPU, no model: a directory-backed store stands in for the bucket and a
fake scorer returns scores derived from the text.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kma import incitement as inc
from kma import incitement_backlog as ib
from kma import relevance
from kma.topics_assign import Obj

T0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
RUN = "20261011T010000Z"


class DirStore:
    def __init__(self, root: Path, allowed_puts: set[str] | None = None):
        self.root = root
        self._puts = allowed_puts
        self.put_log: list[str] = []

    def list(self, prefix):
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(
            (Obj(p.relative_to(self.root).as_posix(), p.stat().st_size) for p in base.rglob("*") if p.is_file()),
            key=lambda o: o.key,
        )

    def get(self, key, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes((self.root / key).read_bytes())

    def put(self, key, src):
        if self._puts is not None and key not in self._puts:
            raise KeyError(key)
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(src).read_bytes())
        self.put_log.append(key)

    def put_keys(self):
        return sorted(self._puts) if self._puts is not None else []


class FakeScorer:
    batch = 128

    def __init__(self, fail_on_call: int | None = None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def peak_gib(self):
        return 0.5

    def score(self, texts):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("boom")
        return [
            {"dehumanisation": 0.1, "violence_call": min(len(t) / 100, 1.0), "othering": 0.2,
             "political_criticism": 0.3}
            for t in texts
        ]


def write_posts(root: Path, rows, name="a", day="2026-10-01"):
    path = root / ib.posts_prefix() / "type=search" / f"dt={day}" / f"run={name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "platform_post_id": [r[0] for r in rows],
                "text": [r[1] for r in rows],
                "collected_at": pa.array([T0 + timedelta(minutes=r[2]) for r in rows], pa.timestamp("us", tz="UTC")),
            }
        ),
        path,
    )


def write_relevance(root: Path, scores: dict[str, float]):
    path = root / ib.relevance_prefix() / "dt=2026-10-01" / "run=r.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "platform_post_id": list(scores),
                "p_kenya": list(scores.values()),
                "scored_at": pa.array([T0] * len(scores), pa.timestamp("us", tz="UTC")),
            }
        ),
        path,
    )


def write_scored(root: Path, ids):
    df = {"platform_post_id": ids, "lexicon_hits": [[] for _ in ids], "lexicon_categories": [[] for _ in ids]}
    table = inc.score_table(
        __import__("pandas").DataFrame(df),
        [{"dehumanisation": 0.0, "violence_call": 0.0, "othering": 0.0, "political_criticism": 0.0}] * len(ids),
        T0,
    )
    path = root / ib.incitement_prefix() / "dt=2026-09-28" / "run=old.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "bucket"
    write_posts(
        root,
        [
            ("old_kenya", "Ruto and the Kenya government", 0),
            ("new_other", "a football score line", 50),
            ("new_kenya", "Nairobi county news", 40),
            ("old_other", "recipe for pilau", 5),
            ("done", "already scored text", 60),
            ("blank", "   ", 70),
            ("dup", "first copy", 1),
            ("dup", "second copy", 2),
        ],
    )
    write_relevance(root, {"old_kenya": 0.9, "new_kenya": 0.6, "new_other": 0.1, "old_other": 0.2, "done": 0.9})
    write_scored(root, ["done"])
    return root


def run_job(bucket, tmp_path, scorer=None, **kw):
    kw.setdefault("run_id", RUN)
    kw.setdefault("chunk_rows", 2)
    return ib.run(DirStore(bucket), tmp_path / "cache", scorer=scorer or FakeScorer(), **kw)


def read_outputs(bucket):
    files = sorted((bucket / ib.incitement_prefix()).rglob("run=*.parquet"))
    return [f for f in files if "old" not in f.name]


def test_relevance_constants_match_what_pi0_signs():
    assert ib.RELEVANCE_MODEL == relevance.ACTIVE_MODEL
    assert ib.KENYA_THRESHOLD == relevance.THRESHOLD
    cron = (Path(__file__).parents[2] / "deploy" / "pi0" / "kenya-incitement-cron.sh").read_text()
    assert f'RELEVANCE = "{ib.RELEVANCE_MODEL}"' in cron


def test_plan_orders_kenya_first_then_newest(bucket, tmp_path):
    marker = run_job(bucket, tmp_path)
    plan = pq.read_table(tmp_path / "cache" / "runs" / RUN / "plan.parquet").to_pandas()
    assert plan["platform_post_id"].tolist() == ["new_kenya", "old_kenya", "new_other", "old_other", "dup"]
    assert plan["kenya"].tolist() == [True, True, False, False, False]
    assert plan.loc[plan.platform_post_id == "dup", "text"].item() == "second copy"
    assert marker["pending_at_start"] == 5 and marker["kenya_first"] == 2


def test_unscored_relevance_falls_back_to_keyword_gate(tmp_path):
    root = tmp_path / "bucket"
    write_posts(root, [("a", "Kenya police statement", 1), ("b", "plain words", 2), ("c", "Kenya again", 0)])
    write_relevance(root, {"zzz": 0.0})
    job = run_job(root, tmp_path, chunk_rows=10)
    plan = pq.read_table(tmp_path / "cache" / "runs" / RUN / "plan.parquet").to_pandas()
    assert plan["platform_post_id"].tolist() == ["a", "c", "b"]
    assert job["status"] == "complete"


def test_output_layout_and_schema_match_score_new(bucket, tmp_path):
    marker = run_job(bucket, tmp_path)
    assert marker["status"] == "complete" and marker["scored"] == 5
    keys = sorted(p.relative_to(bucket).as_posix() for p in read_outputs(bucket))
    assert keys == [f"incitement/platform=x/dt=2026-10-11/run={RUN}-c{i:04d}.parquet" for i in range(3)]
    reference = inc.score_table(
        __import__("pandas").DataFrame(
            {"platform_post_id": ["x"], "lexicon_hits": [[]], "lexicon_categories": [[]]}
        ),
        [{"dehumanisation": 0.0, "violence_call": 0.0, "othering": 0.0, "political_criticism": 0.0}],
        T0,
    )
    for path in read_outputs(bucket):
        assert pq.read_schema(path).equals(reference.schema)


def test_marker_is_outside_the_reader_glob(bucket, tmp_path):
    run_job(bucket, tmp_path)
    marker = bucket / ib.marker_key(RUN)
    assert marker.is_file() and json.loads(marker.read_text())["status"] == "complete"
    con = duckdb.connect()
    glob = f"{bucket}/incitement/platform=x/dt=*/run=*.parquet"
    ids = con.sql(f"select platform_post_id from read_parquet('{glob}', union_by_name=true)").df()
    assert not ids.empty and ids.platform_post_id.is_unique


def test_second_run_has_nothing_pending(bucket, tmp_path):
    run_job(bucket, tmp_path)
    scorer = FakeScorer()
    again = ib.run(DirStore(bucket), tmp_path / "cache2", run_id="20261012T010000Z", scorer=scorer)
    assert again["status"] == "complete" and again["pending_at_start"] == 0
    assert scorer.calls == 0


def test_crash_resume_does_not_duplicate_or_rescore(bucket, tmp_path):
    with pytest.raises(RuntimeError):
        run_job(bucket, tmp_path, scorer=FakeScorer(fail_on_call=2))
    assert len(read_outputs(bucket)) == 1
    scorer = FakeScorer()
    marker = run_job(bucket, tmp_path, scorer=scorer)
    assert scorer.calls == 2
    assert marker["status"] == "complete" and marker["scored"] == 5
    con = duckdb.connect()
    glob = f"{bucket}/incitement/platform=x/dt=*/run={RUN}-c*.parquet"
    got = con.sql(f"select platform_post_id from read_parquet('{glob}')").df()
    assert len(got) == 5 and got.platform_post_id.is_unique
    assert marker["readback_rows"] == marker["readback_distinct"] == 5


def test_resume_reuses_the_plan_even_if_new_scores_landed(bucket, tmp_path):
    with pytest.raises(RuntimeError):
        run_job(bucket, tmp_path, scorer=FakeScorer(fail_on_call=2))
    write_scored(bucket, ["old_other"])
    marker = run_job(bucket, tmp_path)
    assert marker["pending_at_start"] == 5 and marker["scored"] == 5


def test_chunk_already_uploaded_is_not_put_twice(bucket, tmp_path):
    store = DirStore(bucket)
    with pytest.raises(RuntimeError):
        ib.run(store, tmp_path / "cache", run_id=RUN, chunk_rows=2, scorer=FakeScorer(fail_on_call=3))
    first = list(store.put_log)
    ib.run(store, tmp_path / "cache", run_id=RUN, chunk_rows=2, scorer=FakeScorer())
    again = store.put_log[len(first):]
    assert first and not set(first) & set(again)


def test_limit_plans_only_the_head(bucket, tmp_path):
    marker = run_job(bucket, tmp_path, limit=2)
    assert marker["pending_at_start"] == 2 and marker["scored"] == 2


def test_missing_put_urls_report_partial(bucket, tmp_path):
    allowed = {ib.chunk_key(RUN, 0), ib.marker_key(RUN)}
    marker = ib.run(DirStore(bucket, allowed), tmp_path / "cache", run_id=RUN, chunk_rows=2, scorer=FakeScorer())
    assert marker["status"] == "partial" and marker["scored"] == 2


def test_dry_run_writes_nothing_remote(bucket, tmp_path):
    marker = run_job(bucket, tmp_path, dry_run=True)
    assert read_outputs(bucket) == [] and not (bucket / ib.marker_key(RUN)).exists()
    assert marker["scored"] == 5


def test_lexicon_hits_are_populated(tmp_path):
    root = tmp_path / "bucket"
    write_posts(root, [("a", "hawa ni madoadoa", 1), ("b", "habari za leo", 0)])
    run_job(root, tmp_path, chunk_rows=10)
    out = pq.read_table(read_outputs(root)[0]).to_pandas().set_index("platform_post_id")
    assert out.loc["a", "lexicon_hits"].tolist() == ["madoadoa"]
    assert out.loc["b", "lexicon_hits"].tolist() == []
