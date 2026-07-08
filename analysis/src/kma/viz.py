"""Shared chart tokens and helpers for the analysis notebooks.

One validated palette (dataviz-skill reference instance, light surface) and a
handful of matplotlib primitives so every notebook reads as one system: thin
marks, hairline solid grid, recessive axes, ink-token text.

Rules baked in so notebooks cannot drift:
- one hue for single-series magnitude (never a value-ramp on nominal bars)
- categorical hues in fixed slot order, never cycled or generated
- diverging = blue/red poles + neutral gray midpoint (polarity only)
- text wears ink tokens, marks wear series color
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots, fixed order (validated: worst adjacent CVD dE 24.2).
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"
CATEGORICAL = [BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE]

# Sequential blue ramp (100 -> 700) for continuous magnitude.
SEQ_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# Ordinal 3-step blue (validated --ordinal): loose -> strict.
ORDINAL_3 = ["#86b6ef", "#2a78d6", "#104281"]

# Diverging poles + neutral midpoint (polarity only, never identity).
DIV_POS = BLUE
DIV_NEG = RED
NEUTRAL = "#c8c7c0"

# Ink + chrome tokens (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DEEMPH = "#c3c2b7"  # context series in emphasis charts

SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("kma_seq", SEQ_RAMP)


def use_theme() -> None:
    """Apply the notebook chart theme once per session."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": 600,
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def new_fig(width: float = 9.0, height: float = 4.0):
    """Figure + axes with the house margins."""
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.14)
    return fig, ax


def compact(v: float) -> str:
    """1284 -> 1.3K, 4200000 -> 4.2M."""
    v = float(v)
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= cut:
            return f"{v / cut:.1f}{suffix}".replace(".0", "")
    return f"{v:,.0f}" if abs(v) >= 1 or v == 0 else f"{v:.2f}"


def _finite_xlim(values, *, default=(-1.0, 1.0)):
    """Symmetric-ish x limits for diverging bar charts; NaN-safe."""
    vals = np.asarray(list(values), dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return default
    lo, hi = float(finite.min()), float(finite.max())
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if lo == hi:
        pad = max(abs(lo) * 0.1, 0.5)
        lo, hi = lo - pad, hi + pad
    return lo, hi


def _plot_values(values):
    """Map non-finite bar values to 0 so matplotlib can render."""
    return np.where(np.isfinite(values), values, 0.0)

def hbars(ax, labels, values, color=BLUE, colors=None, thickness=0.55,
          label_tips=True, tip_fmt=compact):
    """Horizontal single-series bars: one hue, value at the tip.

    `colors` overrides per-bar color for diverging-by-sign use only - never to
    rank nominal categories."""
    labels = list(labels)
    values = _plot_values(np.asarray(list(values), dtype=float))
    ys = np.arange(len(labels))[::-1]
    ax.barh(ys, values, height=thickness, color=colors or color, linewidth=0)
    span = max(abs(values).max(), 1e-9)
    if label_tips:
        for y, v in zip(ys, values):
            off = span * 0.015
            ax.text(v + (off if v >= 0 else -off), y, tip_fmt(v),
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=9, color=INK_2)
    ax.set_yticks(ys, labels)
    lo, hi = _finite_xlim(values)
    pad = (hi - lo) * (0.14 if label_tips else 0.06) + 1e-9
    ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.grid(axis="y", visible=False)
    if (values < 0).any():
        ax.axvline(0, color=BASELINE, linewidth=1)


def vbars(ax, labels, values, color=BLUE, thickness=0.6):
    """Vertical single-series columns: one hue."""
    labels = list(labels)
    values = np.asarray(list(values), dtype=float)
    xs = np.arange(len(labels))
    ax.bar(xs, values, width=thickness, color=color, linewidth=0)
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, (values.max() if len(values) else 1) * 1.08)
    ax.grid(axis="x", visible=False)


def grouped_hbars(ax, group_labels, series, thickness=0.78, legend_loc="upper right"):
    """Grouped horizontal bars. `series` = list of (name, values, color) in
    fixed slot order; a legend is always drawn (>= 2 series rule)."""
    n_groups = len(group_labels)
    n_series = len(series)
    band = thickness / n_series
    ys = np.arange(n_groups)[::-1]
    for i, (name, vals, color) in enumerate(series):
        offs = ys + thickness / 2 - (i + 0.5) * band
        arr = _plot_values(np.asarray(list(vals), dtype=float))
        ax.barh(offs, arr, height=band * 0.9, color=color, linewidth=0, label=name)
    ax.set_yticks(ys, group_labels)
    all_vals = np.concatenate(
        [_plot_values(np.asarray(list(v), dtype=float)) for _, v, _ in series]
    )
    lo, hi = _finite_xlim(all_vals)
    pad = (hi - lo) * 0.15 + 1e-9
    ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)
    ax.set_ylim(-0.6, n_groups - 0.4)
    ax.grid(axis="y", visible=False)
    if (all_vals < 0).any():
        ax.axvline(0, color=BASELINE, linewidth=1)
    ax.legend(loc=legend_loc)


def diverging_stack(ax, rows, labels, order=("negative", "neutral", "positive"),
                    colors=(DIV_NEG, NEUTRAL, DIV_POS), thickness=0.5):
    """Diverging 100% stacked bars centered on neutral: negatives grow left,
    positives right, half the neutral share on each side. `rows` = iterable of
    dicts {order key -> count}. Segment labels only where they fit."""
    ys = np.arange(len(labels))[::-1]
    gap = 0.006  # 2px-feel surface gap
    for y, row in zip(ys, rows):
        total = max(sum(row.get(k, 0) for k in order), 1e-9)
        neg = row.get(order[0], 0) / total
        neu = row.get(order[1], 0) / total
        pos = row.get(order[2], 0) / total
        for x0, w, c in (
            (-(neg + neu / 2), neg, colors[0]),
            (-neu / 2, neu, colors[1]),
            (neu / 2, pos, colors[2]),
        ):
            if w > 2 * gap:
                ax.barh(y, w - 2 * gap, left=x0 + gap, height=thickness,
                        color=c, linewidth=0)
        for share, x in ((neg, -neu / 2 - neg / 2), (pos, neu / 2 + pos / 2)):
            if share >= 0.12:  # label only when it fits inside the segment
                ax.text(x, y, f"{share:.0%}", va="center", ha="center",
                        fontsize=9, color=SURFACE, fontweight=600)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(ys, labels)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.grid(visible=False)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["100%", "50%", "0", "50%", "100%"])


def legend_swatches(ax, items, loc="upper right"):
    """Legend from (label, color) pairs without plotting fake artists."""
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in items], loc=loc)
