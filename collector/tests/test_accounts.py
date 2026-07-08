from kenya_monitor.accounts import metrics_cap, posts_gap_hours
from kenya_monitor.pacing import _effective_bounds, set_pool_size


def test_metrics_cap_scales_with_pool():
    assert metrics_cap(0, 8, 200) == 200
    assert metrics_cap(10, 8, 200) == 200
    assert metrics_cap(54, 8, 200) == 432


def test_posts_gap_scales_down_with_large_pool():
    lo, hi = posts_gap_hours(50, 3.0, 5.0)
    assert lo == 1.5
    assert hi == 2.5
    lo, hi = posts_gap_hours(5, 3.0, 5.0)
    assert lo == 3.0
    assert hi == 5.0


def test_pacing_auto_scales_with_pool_size():
    set_pool_size(1)
    lo, hi = _effective_bounds(3.0, 12.0)
    assert lo == 3.0 and hi == 12.0

    set_pool_size(49)
    lo, hi = _effective_bounds(3.0, 12.0)
    assert lo < 3.0
    assert hi < 12.0
    assert lo >= 1.0
