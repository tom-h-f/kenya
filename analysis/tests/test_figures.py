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


def test_ranking_stability_reports_the_holdout_beside_the_treated_arm(monkeypatch):
    """The treated arm alone is not a result. A recomputed ranking churns, so
    without the control there is nothing to read the 0% against."""
    import pandas as pd

    before = pd.DataFrame({"user_id": ["a", "b", "c", "d"], "rank": [1, 2, 3, 4]})
    after = pd.DataFrame({"user_id": ["c", "d"], "rank": [1, 2]})

    class _Sql:
        def __init__(self, frame): self._frame = frame
        def df(self): return self._frame

    calls = {"n": 0}

    def fake_scores(_con, run, platform="x"):
        calls["n"] += 1
        return _Sql(before if calls["n"] == 1 else after)

    from kma import db

    monkeypatch.setattr(db, "coord2_scores", fake_scores)
    monkeypatch.setattr(db, "deepened_accounts", lambda _c, **k: _Sql(pd.DataFrame({"user_id": ["a", "c"]})))
    monkeypatch.setattr(db, "held_out_accounts", lambda _c, **k: _Sql(pd.DataFrame({"user_id": ["b", "d"]})))

    out = figures.ranking_stability(None, "before", "after")

    assert out["treated"] == {"in_top_before": 2, "in_top_after": 1, "retained": 0.5}
    assert out["holdout"] == {"in_top_before": 2, "in_top_after": 1, "retained": 0.5}
    assert "holdout" in out["note"]


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
