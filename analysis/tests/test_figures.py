"""The figures exporter publishes four audited numbers and refuses a fifth.

What is tested is the discipline, not the arithmetic: that a transcribed number
cannot travel without its source, that the live half is recomputed rather than
read from a file, and that the absence of any prevalence rate is explicit
rather than an omission a reader could fill in.
"""

from __future__ import annotations

import json

import pytest
import yaml

from kma import figures


def _recorded(tmp_path, block: dict):
    path = tmp_path / "recorded.yaml"
    path.write_text(yaml.safe_dump(block))
    return path


def test_a_transcribed_figure_needs_its_source_and_date(tmp_path):
    path = _recorded(tmp_path, {"thing": {"title": "t", "measured": "2026-09-13"}})

    with pytest.raises(ValueError, match="source"):
        figures.load_recorded(path)


def test_a_transcribed_figure_needs_a_measurement_date(tmp_path):
    path = _recorded(tmp_path, {"thing": {"title": "t", "source": "docs/x.md"}})

    with pytest.raises(ValueError, match="measured"):
        figures.load_recorded(path)


def test_the_shipped_recorded_figures_all_carry_provenance():
    loaded = figures.load_recorded()

    assert set(loaded) == {"benchmark_reproduction", "relevance_error", "adjudication"}
    for name, block in loaded.items():
        assert block["source"] and block["measured"], name


def test_the_shipped_figures_claim_no_influence_operation():
    """The adjudication is publishable as a description of the method's output.
    Nothing on either side was called an influence operation, and the figure
    has to keep saying so - it is the finding."""
    assert figures.load_recorded()["adjudication"]["influence_operations_found"] == 0


class _Sql:
    def __init__(self, frame): self._frame = frame
    def df(self): return self._frame


class _Con:
    """Answers only the snapshot-name lookup; everything else is monkeypatched."""

    def __init__(self, names): self._names = list(names)
    def sql(self, _query): return self
    def fetchone(self): return (self._names.pop(0),)


def _stub_ranking(monkeypatch, *, deepened, t0, t1):
    import pandas as pd

    from kma import bench, db

    before = pd.DataFrame({"user_id": ["a", "b", "c", "d"], "rank": [1, 2, 3, 4]})
    after = pd.DataFrame({"user_id": ["c", "d"], "rank": [1, 2]})
    calls = {"n": 0}

    def fake_scores(_con, run, platform="x"):
        calls["n"] += 1
        return _Sql(before if calls["n"] == 1 else after)

    monkeypatch.setattr(db, "coord2_scores", fake_scores)
    monkeypatch.setattr(db, "deepened_accounts", lambda _c, **k: _Sql(deepened))
    monkeypatch.setattr(
        db, "held_out_accounts",
        lambda _c, **k: _Sql(pd.DataFrame({"user_id": ["b", "d"]})),
    )
    stamps = {"snap_before": t0, "snap_after": t1}
    monkeypatch.setattr(
        bench, "load",
        lambda name, con=None: pd.DataFrame({"created_at": [stamps[name]]}),
    )
    return _Con(["snap_before", "snap_after"])


def test_ranking_stability_reports_the_holdout_beside_the_treated_arm(monkeypatch):
    """The treated arm alone is not a result. A recomputed ranking churns, so
    without the control there is nothing to read the 0% against."""
    import pandas as pd

    t0 = pd.Timestamp("2026-09-14T09:00:00Z")
    t1 = pd.Timestamp("2026-09-14T15:00:00Z")
    deepened = pd.DataFrame({
        "user_id": ["a", "c"],
        "first_deepened_at": [t0 + pd.Timedelta(hours=1)] * 2,
    })
    con = _stub_ranking(monkeypatch, deepened=deepened, t0=t0, t1=t1)

    out = figures.ranking_stability(con, "before", "after")

    assert out["treated"] == {"in_top_before": 2, "in_top_after": 1, "retained": 0.5}
    assert out["holdout"] == {"in_top_before": 2, "in_top_after": 1, "retained": 0.5}
    assert "holdout" in out["note"]


def test_accounts_deepened_before_the_earlier_snapshot_are_in_neither_arm(monkeypatch):
    """They are treated in BOTH runs, so counting them inflates the treated arm
    and can put a survivor in it. This is the discrepancy that had the page
    reporting 391 treated with 1 survivor where the check script said 390/0."""
    import pandas as pd

    t0 = pd.Timestamp("2026-09-14T09:00:00Z")
    t1 = pd.Timestamp("2026-09-14T15:00:00Z")
    deepened = pd.DataFrame({
        "user_id": ["a", "c"],
        # `c` survives into the later ranking and was deepened BEFORE t0.
        "first_deepened_at": [t0 + pd.Timedelta(hours=1), t0 - pd.Timedelta(days=1)],
    })
    con = _stub_ranking(monkeypatch, deepened=deepened, t0=t0, t1=t1)

    out = figures.ranking_stability(con, "before", "after")

    assert out["treated"] == {"in_top_before": 1, "in_top_after": 0, "retained": 0.0}


def test_the_export_says_prevalence_is_absent_rather_than_omitting_it(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "corpus_shape", lambda con, snapshot: {"snapshot": snapshot})
    monkeypatch.setattr(figures, "ranking_stability", lambda con, b, a: {"before_run": b})

    out = tmp_path / "figures.json"
    got = figures.export(
        out, snapshot="s", before_run="b", after_run="a", con=object(),
    )

    assert "prevalence" in got["absent"]
    assert "denominator" in got["absent"]["prevalence"]
    assert json.loads(out.read_text())["absent"] == got["absent"]
