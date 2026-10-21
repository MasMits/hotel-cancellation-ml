"""Shared palette and figure styling, based on the preprocessing slide figures."""

from functools import wraps
from pathlib import Path

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PRIMARY = "#A82487"
BLUE = "#2366AB"
GREEN = "#237A3B"
AMBER = "#A96508"
TEXT = "#202733"
MUTED = "#4B5563"
PREVIEW_BACKGROUND = "#F5F3F7"
PALETTE = [PRIMARY, BLUE, GREEN, AMBER]
HEATMAP = LinearSegmentedColormap.from_list("hotel_cancellation", ["#F5EAF3", "#CD83BC", "#771760"])

SLIDE_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 21, "axes.titleweight": "bold", "axes.titlepad": 18,
    "axes.labelpad": 12, "text.color": TEXT, "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.grid": False, "figure.facecolor": "none", "axes.facecolor": "none",
    "legend.facecolor": "none", "legend.edgecolor": "none", "legend.labelcolor": TEXT,
    "axes.prop_cycle": cycler(color=PALETTE), "lines.linewidth": 2.6,
}
NOTEBOOK_STYLE = {
    **SLIDE_STYLE,
    "savefig.facecolor": "none", "savefig.transparent": True, "legend.frameon": False,
}


def style_axis(axis):
    """Left-align titles and use the same quiet axes as preprocessing."""
    title = axis.get_title() or axis.get_title(loc="left")
    axis.set_title("")
    axis.set_title(title, loc="left")
    axis.grid(False, axis="both", which="both")
    axis.set_axisbelow(True)
    axis.tick_params(length=0, pad=9, colors=MUTED)
    for spine in axis.spines.values():
        spine.set_visible(False)


def finish_figure(figure):
    figure.patch.set_alpha(0)
    for axis in figure.axes:
        axis.patch.set_alpha(0)
        style_axis(axis)
    figure.tight_layout(pad=1.5)


def show_figure(figure, name):
    """Display a transparent figure and export matching PNG/SVG files for slides."""
    finish_figure(figure)
    folder = Path(__file__).resolve().parents[2] / "reports/figures"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(folder / f"{name}.{extension}", transparent=True, dpi=220,
                       bbox_inches="tight", pad_inches=.25)
    plt.show()


def themed(function):
    """Keep plotting helpers self-contained without changing global settings."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with plt.rc_context(NOTEBOOK_STYLE):
            return function(*args, **kwargs)
    return wrapped
