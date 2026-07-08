"""Unit tests for kma.viz."""

import matplotlib.pyplot as plt
import numpy as np

from kma import viz


def test_grouped_hbars_handles_nan_values():
    fig, ax = viz.new_fig(6, 2)
    viz.grouped_hbars(
        ax,
        ["cluster a", "cluster b"],
        [
            ("suspicion", [1.2, 0.8], viz.BLUE),
            ("homogeneity", [np.nan, np.nan], viz.AQUA),
        ],
    )
    lo, hi = ax.get_xlim()
    assert np.isfinite(lo) and np.isfinite(hi)
    plt.close(fig)


def test_finite_xlim_all_nan():
    lo, hi = viz._finite_xlim([np.nan, np.nan])
    assert lo == -1.0 and hi == 1.0
