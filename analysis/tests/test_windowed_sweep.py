"""The sweep harness itself: a plant that is never recovered would read as a
floor that is too high, so the harness is checked against a case with a known
answer before any sweep number is believed."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PATH = Path(__file__).resolve().parents[1] / "investigations/2026-09-22-windowed-floor/sweep.py"
_spec = importlib.util.spec_from_file_location("windowed_sweep", _PATH)
sweep = importlib.util.module_from_spec(_spec)
sys.modules["windowed_sweep"] = sweep
_spec.loader.exec_module(sweep)

START = pd.Timestamp("2026-09-01", tz="UTC")


def _background(days=5, users=80, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, pool in (("co_retweet", 300), ("co_url", 100), ("hashtag_sequence", 60),
                       ("fast_retweet", 40)):
        n = days * users * 3
        out[name] = pd.DataFrame({
            "user_id": [f"u{i}" for i in rng.integers(0, users, n)],
            "entity": [f"{name}{i}" for i in rng.integers(0, pool, n)],
            "created_at": START + pd.to_timedelta(rng.uniform(0, days * 86400, n), unit="s"),
        })
    return out


def test_a_plant_never_touches_its_input():
    rows = _background()
    before = {n: t.copy() for n, t in rows.items()}
    sweep.plant(rows, sweep.HEAVY, "2026-09-03", seed=1)
    for name in rows:
        pd.testing.assert_frame_equal(rows[name], before[name])


def test_planted_actions_sit_inside_the_plant_day_and_use_the_campaign():
    rows = _background()
    out, ids = sweep.plant(rows, sweep.LIGHT, "2026-09-03", seed=2)
    rt = out["co_retweet"]
    campaign = rt[rt["entity"].str.startswith("camp2_")]

    assert len(ids) == sweep.LIGHT.k
    assert set(campaign["user_id"]) == set(ids)
    assert (campaign["created_at"].dt.strftime("%Y-%m-%d") == "2026-09-03").all()
    assert (campaign.groupby("user_id")["entity"].nunique() == sweep.LIGHT.tweets_each).all()


def test_a_heavy_plant_is_recovered_as_one_community():
    """The known answer: twenty accounts sharing ten of fifteen tweets in six
    hours, against random background, at floor 5. Not all twenty land in one
    community even here - each carries a donor's organic actions, which tie it
    to background accounts - but the campaign's community must lead the
    window's report."""
    rows = _background()
    out, ids = sweep.plant(rows, sweep.HEAVY, "2026-09-03", seed=3)
    frame = sweep.run_real(out, width="day", floor=5, windows=["2026-09-03"])
    got = sweep.plant_metrics(frame, ids, "2026-09-03")

    assert got["present"] == 1.0
    assert got["recall"] >= 0.8
    assert got["community_rank"] == 1


def test_only_whole_windows_are_scored():
    rows = _background(days=5)
    days = sweep.complete_windows(rows, "day")
    assert days[0] > "2026-09-01" or rows["co_retweet"]["created_at"].min() <= START
    assert "2026-09-05" not in days
