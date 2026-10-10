"""Modal spend report: the arithmetic and the exit codes, with no network and no modal."""

from __future__ import annotations

import sys
from datetime import date, timedelta

import pytest

from kma import modal_spend
from kma.modal_spend import DayCost, date_chunks, month_bounds, summarise

OCT = (date(2026, 10, 1), date(2026, 11, 1))


def _days(app: str, first: date, costs: list[float]) -> list[DayCost]:
    return [DayCost(app, first + timedelta(days=i), c) for i, c in enumerate(costs)]


def test_month_bounds_default_and_explicit():
    assert month_bounds(None, date(2026, 10, 10)) == OCT
    assert month_bounds("2026-02", date(2026, 10, 10)) == (date(2026, 2, 1), date(2026, 3, 1))
    assert month_bounds("2026-12", date(2026, 12, 3)) == (date(2026, 12, 1), date(2027, 1, 1))


@pytest.mark.parametrize("bad", ["2026", "2026-13", "oct", "2026-1-x"])
def test_month_bounds_rejects_malformed(bad):
    with pytest.raises(ValueError, match="YYYY-MM"):
        month_bounds(bad, date(2026, 10, 10))


def test_month_bounds_rejects_future_month():
    with pytest.raises(ValueError, match="has not started"):
        month_bounds("2026-11", date(2026, 10, 10))


def test_date_chunks_respect_the_span_limit():
    chunks = date_chunks(date(2026, 8, 1), date(2026, 10, 11))
    assert chunks[0][0] == date(2026, 8, 1) and chunks[-1][1] == date(2026, 10, 11)
    assert all((b - a).days <= 31 for a, b in chunks)
    assert all(prev[1] == nxt[0] for prev, nxt in zip(chunks, chunks[1:]))
    assert sum((b - a).days for a, b in chunks) == 71


def test_date_chunks_exact_limit_is_one_call_and_empty_is_none():
    assert date_chunks(date(2026, 8, 1), date(2026, 9, 1)) == [(date(2026, 8, 1), date(2026, 9, 1))]
    assert date_chunks(date(2026, 10, 1), date(2026, 10, 1)) == []


def test_summarise_projects_trailing_rate_over_remaining_days():
    as_of = date(2026, 10, 10)
    rows = _days("a", OCT[0], [1.0] * 9) + _days("b", OCT[0], [0.0] * 5 + [7.0, 0.0, 0.0, 0.0])
    s = summarise(rows, *OCT, as_of=as_of, budget=30)

    assert s.month_to_date == pytest.approx(16.0)
    assert s.trailing == pytest.approx(7 * 1.0 + 7.0)
    assert s.daily_rate == pytest.approx(2.0)
    assert s.days_remaining == 22
    assert s.projected == pytest.approx(16.0 + 2.0 * 22)
    assert s.over_budget
    by_app = {a.app: a for a in s.apps}
    assert by_app["a"].projected == pytest.approx(9.0 + 22.0)
    assert by_app["b"].projected == pytest.approx(7.0 + 22.0)
    assert sum(a.projected for a in s.apps) == pytest.approx(s.projected)
    assert [a.app for a in s.apps] == ["a", "b"]


def test_summarise_window_reaches_into_the_previous_month():
    as_of = date(2026, 10, 3)
    rows = _days("a", date(2026, 9, 26), [1.0] * 7)
    s = summarise(rows, *OCT, as_of=as_of, budget=30)
    assert s.month_to_date == pytest.approx(2.0)
    assert s.trailing == pytest.approx(7.0)
    assert s.projected == pytest.approx(2.0 + 1.0 * 29)


def test_summarise_ignores_the_unreported_day_and_beyond():
    rows = _days("a", date(2026, 10, 8), [1.0, 1.0, 50.0])
    s = summarise(rows, *OCT, as_of=date(2026, 10, 10), budget=30)
    assert s.month_to_date == pytest.approx(2.0)


def test_summarise_finished_month_projects_nothing_more():
    rows = _days("a", date(2026, 9, 1), [2.0] * 30)
    s = summarise(rows, date(2026, 9, 1), date(2026, 10, 1), date(2026, 10, 1), budget=30)
    assert s.days_remaining == 0
    assert s.projected == pytest.approx(60.0)
    assert s.over_budget


def test_summarise_sums_multiple_objects_of_one_app_per_day():
    rows = [DayCost("kma-textsim", date(2026, 10, 2), 0.1), DayCost("kma-textsim", date(2026, 10, 2), 0.2)]
    s = summarise(rows, *OCT, as_of=date(2026, 10, 5), budget=30)
    assert [(a.app, a.month_to_date) for a in s.apps] == [("kma-textsim", pytest.approx(0.3))]


def test_render_names_every_app_and_the_verdict():
    rows = _days("kma-relevance", OCT[0], [1.0] * 9)
    text = modal_spend.render(summarise(rows, *OCT, as_of=date(2026, 10, 10), budget=30))
    assert "kma-relevance" in text and "total" in text
    assert "$9.00" in text and "$31.00" in text
    assert "OVER BUDGET" in text


def _fake_fetch(rows):
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return rows

    return fetch, calls


def test_main_within_budget_exits_zero(capsys):
    fetch, calls = _fake_fetch(_days("a", OCT[0], [0.5] * 9))
    code = modal_spend.main(["--budget", "30"], today=date(2026, 10, 10), fetch=fetch)
    assert code == 0
    assert calls == [(date(2026, 10, 1), date(2026, 10, 10))]
    assert "within budget" in capsys.readouterr().out


def test_main_over_budget_exits_nonzero(capsys):
    fetch, _ = _fake_fetch(_days("a", OCT[0], [2.0] * 9))
    code = modal_spend.main(["--budget", "30"], today=date(2026, 10, 10), fetch=fetch)
    assert code == 1
    assert "OVER BUDGET" in capsys.readouterr().out


def test_main_budget_comes_from_env(monkeypatch, capsys):
    monkeypatch.setenv(modal_spend.BUDGET_ENV, "100")
    fetch, _ = _fake_fetch(_days("a", OCT[0], [2.0] * 9))
    assert modal_spend.main([], today=date(2026, 10, 10), fetch=fetch) == 0
    assert "budget $100.00" in capsys.readouterr().out


def test_main_flag_beats_env(monkeypatch):
    monkeypatch.setenv(modal_spend.BUDGET_ENV, "100")
    fetch, _ = _fake_fetch(_days("a", OCT[0], [2.0] * 9))
    assert modal_spend.main(["--budget", "30"], today=date(2026, 10, 10), fetch=fetch) == 1


def test_main_empty_report_is_said_not_assumed_zero(capsys):
    fetch, _ = _fake_fetch([])
    code = modal_spend.main([], today=date(2026, 10, 10), fetch=fetch)
    captured = capsys.readouterr()
    assert code == modal_spend.EXIT_NO_DATA
    assert "returned no rows" in captured.err
    assert "projected" not in captured.out


def test_main_early_month_fetches_the_previous_month_for_the_rate():
    fetch, calls = _fake_fetch(_days("a", date(2026, 9, 26), [1.0] * 7))
    code = modal_spend.main(["--budget", "40"], today=date(2026, 10, 3), fetch=fetch)
    assert calls == [(date(2026, 9, 26), date(2026, 10, 3))]
    assert code == 0


def test_main_finished_month_does_not_fetch_beyond_it():
    fetch, calls = _fake_fetch(_days("a", date(2026, 9, 1), [1.0] * 30))
    modal_spend.main(["--month", "2026-09"], today=date(2026, 10, 10), fetch=fetch)
    assert calls == [(date(2026, 9, 1), date(2026, 10, 1))]


def test_main_rejects_future_month(capsys):
    fetch, calls = _fake_fetch([])
    assert modal_spend.main(["--month", "2026-12"], today=date(2026, 10, 10), fetch=fetch) == 2
    assert calls == []
    assert "has not started" in capsys.readouterr().err


def test_fetch_without_modal_says_how_to_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(SystemExit, match="--with modal"):
        modal_spend.fetch_rows(date(2026, 10, 1), date(2026, 10, 10))
