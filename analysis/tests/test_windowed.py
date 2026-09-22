"""Windowed v2: the same detector, one window at a time.

What these guard is the point of windowing - that activity is counted inside
the window it happened in - and the equivalences that let a windowed run be
read beside the global one.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from kma import coord2, windowed

DAY = pd.Timestamp("2026-09-02 10:00", tz="UTC")


def _rows(pairs, when=DAY):
    """(user, entity) pairs, all at `when` unless a pair carries its own time."""
    return pd.DataFrame(
        [(u, e, t if t is not None else when) for u, e, *rest in pairs for t in [rest[0] if rest else None]],
        columns=["user_id", "entity", "created_at"],
    )


def _ring(prefix, users, entities, when=DAY):
    return [(f"{prefix}{u}", f"{prefix}obj{e}", when) for u in range(users) for e in range(entities)]


def test_weeks_start_on_monday_and_days_on_the_day():
    when = pd.Series(pd.to_datetime(["2026-09-02 23:59", "2026-09-06 00:01", "2026-09-07 00:00"], utc=True))

    assert windowed.window_label(when, "day").tolist() == ["2026-09-02", "2026-09-06", "2026-09-07"]
    assert windowed.window_label(when, "week").tolist() == ["2026-08-31", "2026-08-31", "2026-09-07"]


def test_traces_keep_their_gate_shape_unless_time_is_asked_for():
    """The reproduction gate ran on (user_id, entity). Windowing adds a column
    only on request, so no existing trace changes shape."""
    con = duckdb.connect()
    con.register(
        "v",
        pd.DataFrame(
            {
                "user_id": ["a", "b"],
                "post_id": ["1", "2"],
                "created_at": pd.to_datetime(["2026-09-01", "2026-09-02"], utc=True),
                "is_retweet": [True, True],
                "retweet_post_id": ["x", "x"],
                "retweet_user_id": [None, None],
                "text": ["", ""],
                "hashtags": [["a", "b", "c"], ["a", "b", "c"]],
                "urls": [["u"], ["u"]],
            }
        ),
    )
    for extractor in windowed.TIMED_TRACES.values():
        plain = extractor(con, "SELECT * FROM v")
        timed = extractor(con, "SELECT * FROM v", with_time=True)
        assert list(plain.columns) == ["user_id", "entity"]
        assert list(timed.columns) == ["user_id", "entity", "created_at"]
        assert len(plain) == len(timed)


def test_the_floor_counts_entities_inside_the_window_only():
    """An account with two entities a day for five days has ten entities in
    total and two in any day. A daily run at floor 5 must not see it - that is
    what makes the floor a statement about a burst."""
    spread = [
        (who, f"d{d}e{e}", DAY + pd.Timedelta(days=d))
        for who in ("steady", "twin") for d in range(5) for e in range(2)
    ]
    result = windowed.run_rows(
        {"co_retweet": _rows(spread)}, width="day", min_entities=5
    )
    assert "steady" not in set(result.per_window["user_id"])

    weekly = windowed.run_rows({"co_retweet": _rows(spread)}, width="week", min_entities=5)
    assert "steady" in set(weekly.per_window["user_id"])


def test_a_one_day_ring_is_one_community_in_its_day_and_absent_elsewhere():
    ring = _ring("c", users=6, entities=6)
    background = [
        (f"n{u}", f"bg{(u * 7 + k) % 40}", DAY + pd.Timedelta(days=1 + (k % 3)))
        for u in range(30) for k in range(8)
    ]
    result = windowed.run_rows(
        {"co_retweet": _rows(ring + background)}, width="day", min_entities=5
    )
    frame = result.per_window
    planted = frame[frame["user_id"].str.startswith("c")]

    assert set(planted["window"]) == {"2026-09-02"}
    assert planted["community"].nunique() == 1
    assert int(planted["community_size"].iloc[0]) == 6


def test_percentiles_and_floor_match_the_global_builder():
    """A window holding the whole corpus must produce the global networks
    exactly: same projection, same filter, same floor."""
    ring = _ring("c", users=5, entities=4) + _ring("d", users=4, entities=3)
    rows = _rows(ring)
    networks, _ = windowed.window_networks({"co_retweet": rows}, min_entities=2)
    expected = coord2.similarity_network(
        rows[["user_id", "entity"]], percentile=coord2.EDGE_PERCENTILE["co_retweet"], min_entities=2
    )
    pd.testing.assert_frame_equal(networks["co_retweet"], expected)


def test_curveball_preserves_both_degree_sequences():
    rows = _rows(_ring("c", users=5, entities=4) + [("x", "cobj0"), ("y", "zz"), ("y", "cobj1")])
    shuffled = windowed.curveball(rows, burn_in=50, seed=3)
    distinct = rows[["user_id", "entity"]].drop_duplicates()

    pd.testing.assert_series_equal(
        distinct.groupby("user_id").size().sort_index(),
        shuffled.groupby("user_id").size().sort_index(),
    )
    pd.testing.assert_series_equal(
        distinct.groupby("entity").size().sort_index(),
        shuffled.groupby("entity").size().sort_index(),
    )


def test_burst_scores_come_from_the_windowed_centrality():
    ring = _ring("c", users=5, entities=5)
    later = _ring("c", users=5, entities=5, when=DAY + pd.Timedelta(days=3))
    result = windowed.run_rows({"co_retweet": _rows(ring + later)}, width="day", min_entities=5)

    got = result.burst.set_index("user_id")
    assert (got.loc[[f"c{u}" for u in range(5)], "windows"] == 2).all()
    assert np.isclose(got.loc["c0", "peak_share"], 0.5, atol=0.05)
