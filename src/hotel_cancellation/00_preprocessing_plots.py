"""Figures used in the preprocessing section of the main notebook."""

from calendar import month_name

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY = "#C64BAD"
BLUE = "#587FB6"
GREEN = "#29A81E"
TEXT = "#F2F8F1"
MUTED = "#B8CCB8"
PREVIEW_BACKGROUND = "#17151D"

SLIDE_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 21,
    "axes.titleweight": "bold",
    "axes.titlepad": 18,
    "axes.labelpad": 12,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": False,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "legend.facecolor": "none",
    "legend.edgecolor": "none",
    "legend.labelcolor": TEXT,
}


def style_axis(axis):
    """Use the slide style without changing other figures in the notebook."""
    title = axis.get_title()
    axis.set_title("")
    axis.set_title(title, loc="left")
    axis.grid(False, axis="both", which="both")
    axis.set_axisbelow(True)
    axis.tick_params(length=0, pad=9)


def plot_class_balance(train_bookings):
    """Show the class balance and cancellation rates of the two hotels."""
    with plt.rc_context(SLIDE_STYLE):
        figure, (balance_plot, hotel_plot) = plt.subplots(1, 2, figsize=(12, 5))
        counts = train_bookings["is_canceled"].value_counts().reindex([0, 1], fill_value=0)
        balance_plot.bar(["Not canceled", "Canceled"], counts, color=[BLUE, PRIMARY])
        balance_plot.set(title="Class balance", ylabel="Bookings")
        balance_plot.set_ylim(0, counts.max() * 1.18)
        for position, count in enumerate(counts):
            label = f"{count:,} ({count / counts.sum():.1%})"
            balance_plot.text(position, count + counts.max() * .02, label, ha="center")

        hotels = train_bookings.groupby("hotel")["is_canceled"].agg(["size", "mean"])
        hotel_plot.bar(hotels.index, hotels["mean"] * 100, color=PRIMARY)
        hotel_plot.set(title="Cancellation by hotel", ylabel="Canceled (%)", ylim=(0, 60))
        for position, row in enumerate(hotels.itertuples()):
            label = f"{row.mean:.1%}\nn={row.size:,}"
            hotel_plot.text(position, row.mean * 100 + 2, label, ha="center")
        for axis in figure.axes:
            style_axis(axis)
        figure.tight_layout()
        return figure


def plot_feature_patterns(train_bookings, training_features):
    """Show booking timing, cancellation rates and the frequency of known categories."""
    with plt.rc_context(SLIDE_STYLE):
        figure, (lead_plot, weekday_plot, frequency_plot) = plt.subplots(1, 3, figsize=(18, 5))

        lead_time_groups = pd.cut(
            train_bookings["lead_time"],
            bins=[-1, 0, 7, 30, 90, 180, 365, np.inf],
            labels=["0", "1–7", "8–30", "31–90", "91–180", "181–365", "366+"],
        )
        cancellations = train_bookings.groupby(lead_time_groups, observed=True)["is_canceled"]
        lead_summary = cancellations.agg(["size", "mean"])
        (lead_summary["mean"] * 100).plot.bar(ax=lead_plot, color=PRIMARY, rot=35)
        lead_plot.set(
            title="Cancellation by lead time", xlabel="Days before arrival",
            ylabel="Canceled (%)", ylim=(0, 85),
        )
        for position, row in enumerate(lead_summary.itertuples()):
            lead_plot.text(
                position, row.mean * 100 + 2, f"{row.size:,}", ha="center", fontsize=10,
            )

        month_numbers = {
            name: number for number, name in enumerate(month_name[1:], start=1)
        }
        arrival_dates = pd.to_datetime({
            "year": train_bookings["arrival_date_year"],
            "month": train_bookings["arrival_date_month"].map(month_numbers),
            "day": train_bookings["arrival_date_day_of_month"],
        })
        weekdays = arrival_dates.dt.day_name().str[:3]
        weekday_summary = train_bookings.groupby(weekdays)["is_canceled"].agg(["size", "mean"])
        weekday_summary = weekday_summary.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        weekday_plot.bar(weekday_summary.index, weekday_summary["mean"] * 100, color=PRIMARY)
        weekday_plot.set(title="Arrival weekday", ylabel="Canceled (%)", ylim=(0, 60))
        for position, row in enumerate(weekday_summary.itertuples()):
            weekday_plot.text(position, row.mean * 100 + 2, f"{row.size:,}", ha="center", fontsize=10)

        for column, color in zip(["agent", "company", "country"], [PRIMARY, BLUE, GREEN]):
            counts = training_features[column].fillna("Unknown").value_counts()
            counts = counts.drop("Unknown", errors="ignore")
            frequency_plot.plot(range(1, len(counts) + 1), counts, label=column, color=color, linewidth=2.8)
        frequency_plot.axhline(50, color=MUTED, linestyle="--", label="50 rows")
        frequency_plot.set(
            title="Category frequencies", xlabel="Rank within each column",
            ylabel="Bookings (log)", yscale="log",
        )
        frequency_plot.legend()
        for axis in figure.axes:
            style_axis(axis)
        figure.tight_layout()
        return figure


def plot_experiment_results(results_table):
    """Compare train/validation F1 and the number of transformed features."""
    with plt.rc_context(SLIDE_STYLE):
        validation_results = results_table[results_table["split"] == "validation"]
        validation_results = validation_results.set_index("experiment")
        figure, (score_plot, size_plot) = plt.subplots(1, 2, figsize=(13, 6))

        f1_scores = results_table.pivot(index="experiment", columns="split", values="f1")
        f1_scores = f1_scores.reindex(validation_results.index)
        f1_scores[["train", "validation"]].plot.barh(
            ax=score_plot, color=[BLUE, PRIMARY],
        )
        score_plot.set(title="F1: train vs validation", ylabel="", xlabel="F1", xlim=(0, 0.9))
        score_plot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, title=None)

        validation_results["features"].plot.barh(ax=size_plot, color=PRIMARY)
        size_plot.set(title="Feature count", xlabel="Number of features", ylabel="")
        for axis in figure.axes:
            style_axis(axis)
        figure.tight_layout()
        return figure


def save_figure(figure, path):
    """Save transparent PNG/SVG for slides and a dark PNG preview for the report."""
    figure.savefig(f"{path}.png", dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.25)
    figure.savefig(f"{path}.svg", transparent=True, bbox_inches="tight", pad_inches=0.25)
    figure.savefig(
        f"{path}_preview.png", dpi=160, facecolor=PREVIEW_BACKGROUND,
        transparent=False, bbox_inches="tight", pad_inches=0.25,
    )
