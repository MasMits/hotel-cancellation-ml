"""Shared plot settings and PNG export."""

from pathlib import Path

import matplotlib.pyplot as plt

FIGURE_FOLDER = Path(__file__).resolve().parents[2] / "reports/figures"
STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"}


def show_figure(figure, name):
    """Save a PNG in reports/figures and display the figure."""
    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURE_FOLDER / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.show()
