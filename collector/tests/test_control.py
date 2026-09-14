"""The control arm: sampling time instead of choosing what to look at.

What is tested is the sampling design, because that is the only thing the arm
has. If the window grid drifts, if a window can be sampled twice, or if an
empty window goes unrecorded, the denominator is wrong and every rate computed
from it is wrong with it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from kenya_monitor.collectors.base import Author, Post
from kenya_monitor.control import (
    HORIZON_DAYS,
    TARGET_TYPE,
    Frame,
    collect_control,
    load_frame,
    load_state,
    sample_windows,
    save_state,
    window_key,
    window_population,
)

NOW = datetime(2026, 9, 14, 12, 7, 33, tzinfo=timezone.utc)


def test_the_window_grid_is_aligned_to_the_epoch_not_to_now():
    """Two passes minutes apart must see the SAME windows, or "already sampled"
    is unanswerable and the sample is drawn from two different populations."""
    first = window_population(NOW, minutes=15)
    later = window_population(NOW + timedelta(minutes=7), minutes=15)

    assert all(w.minute % 15 == 0 and w.second == 0 for w in first)
    assert set(first) <= set(later)


def test_the_population_covers_the_search_horizon_and_stops_short_of_now():
    pop = window_population(NOW, minutes=15)

    assert pop[0] >= NOW - timedelta(days=HORIZON_DAYS)
    # One settle window: a window that has not finished cannot be censused.
    assert pop[-1] + timedelta(minutes=15) <= NOW
    assert len(pop) == pytest.approx(HORIZON_DAYS * 24 * 4, abs=2)


def test_a_window_is_never_sampled_twice():
    """A window censused twice is the same observation, not a second one."""
    pop = window_population(NOW, minutes=15)
    first = sample_windows(pop, 10, seed=1)
    second = sample_windows(pop, 10, seed=2, done=[window_key(w) for w in first])

    assert not set(first) & set(second)


def test_the_draw_is_reproducible_from_its_seed():
    pop = window_population(NOW, minutes=15)

    assert sample_windows(pop, 5, seed=3) == sample_windows(pop, 5, seed=3)
    assert sample_windows(pop, 5, seed=3) != sample_windows(pop, 5, seed=4)


def test_sampling_asks_for_more_windows_than_remain():
    pop = window_population(NOW, minutes=15)
    done = [window_key(w) for w in pop[:-2]]

    assert len(sample_windows(pop, 100, seed=0, done=done)) == 2


def test_multi_word_frame_terms_are_quoted():
    """Unquoted, `county government` inside an OR-group is two ANDed terms, so
    the frame file would declare one query and the collector would run another."""
    frame = Frame(terms=("IEBC", "county government"), anchors=("Kenya",))

    assert frame.keyword == 'IEBC OR "county government"'


def test_the_shipped_frame_is_institutional_not_campaign_language():
    """A frame built from campaign terms over-represents campaign talk, which
    is the bias `targets.yaml` already has and this arm exists to avoid."""
    frame = load_frame()
    campaign = {"wantam", "ruto 2027", "#rutomustgo", "kasongo", "gachagua"}

    assert frame.terms and frame.anchors
    assert not {t.lower() for t in frame.terms} & campaign


class _StubCollector:
    platform = "x"

    def __init__(self, per_window: list[int]) -> None:
        self.per_window = list(per_window)
        self.queries: list[tuple[str, str, str]] = []

    async def search(self, keyword, limit, since=None, until=None, product=None, anchors=None, **kw):
        self.queries.append((keyword, since, until))
        n = self.per_window.pop(0) if self.per_window else 0
        for i in range(min(n, limit)):
            yield Post(
                platform="x",
                platform_post_id=f"{since}-{i}",
                author_id="a1",
                author_handle="a",
                created_at=datetime.now(timezone.utc),
                text="t",
                lang="en",
                url=f"https://x.com/a/status/{i}",
                collected_at=datetime.now(timezone.utc),
            )

    def collected_authors(self):
        return [Author(platform="x", platform_user_id="a1", handle="a")]


class _StubStorage:
    def __init__(self) -> None:
        self.posts: list[Post] = []
        self.runs: list[dict] = []
        self.target_types: list[str] = []

    def write_posts(self, posts, target_type, now=None):
        self.posts.extend(posts)
        self.target_types.append(target_type)
        return "posts/key" if posts else None

    def write_control_run(self, rows, platform="x", now=None):
        self.runs.extend(rows)
        return "control_runs/key"

    def write_authors(self, authors, now=None):
        return "authors/key"


def _run(collector, storage, tmp_path, **kw):
    return asyncio.run(
        collect_control(
            collector, storage, state_path=tmp_path / "control.json", now=NOW, **kw
        )
    )


def test_an_empty_window_is_still_recorded(tmp_path):
    """The denominator is sampled WINDOWS. A window with no frame posts is an
    observation, and dropping it biases every rate upward."""
    collector = _StubCollector([0, 3])
    storage = _StubStorage()

    counts = _run(collector, storage, tmp_path, windows=2, cap=100, seed=5)

    assert counts["windows"] == 2
    assert counts["posts"] == 3
    assert len(storage.runs) == 2
    assert [r["posts"] for r in storage.runs].count(0) == 1


def test_a_window_at_the_cap_is_flagged_truncated(tmp_path):
    """At the cap the window may have held more than we took, so it is a ranked
    sample of an unknown larger set and the census claim does not cover it."""
    collector = _StubCollector([10, 2])
    storage = _StubStorage()

    counts = _run(collector, storage, tmp_path, windows=2, cap=10, seed=5)

    assert counts["truncated"] == 1
    assert [r["truncated"] for r in storage.runs] == [True, False] or [
        r["truncated"] for r in storage.runs
    ] == [False, True]


def test_control_posts_land_in_their_own_partition(tmp_path):
    """Neither baseline nor targeted. Unioned into either scope the arm is
    worthless, so it gets a partition a reader has to name deliberately."""
    storage = _StubStorage()

    _run(_StubCollector([2]), storage, tmp_path, windows=1, cap=100, seed=5)

    assert storage.target_types == [TARGET_TYPE] == ["control"]


def test_a_second_pass_samples_windows_the_first_did_not(tmp_path):
    storage = _StubStorage()
    _run(_StubCollector([1, 1]), storage, tmp_path, windows=2, cap=100, seed=5)
    first = set(load_state(tmp_path / "control.json"))

    _run(_StubCollector([1, 1]), storage, tmp_path, windows=2, cap=100, seed=5)
    both = set(load_state(tmp_path / "control.json"))

    assert len(first) == 2
    assert len(both) == 4


def test_the_window_bounds_reach_the_collector_as_the_query(tmp_path):
    collector = _StubCollector([1])
    _run(collector, _StubStorage(), tmp_path, windows=1, minutes=15, cap=100, seed=5)

    _keyword, since, until = collector.queries[0]

    assert since.endswith("_UTC") and until.endswith("_UTC")
    assert since < until
