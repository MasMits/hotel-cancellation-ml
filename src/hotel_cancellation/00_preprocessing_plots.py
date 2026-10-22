"""Preprocessing plots."""

from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

arrival_dates = import_module("hotel_cancellation.0_preprocessing").arrival_dates
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def cancellation_bars(ax, summary, title, **labels):
    """Plot cancellation rates and booking counts from the size and mean columns."""
    ax.bar(summary.index.astype(str), summary["mean"] * 100)
    for x, row in enumerate(summary.itertuples()):
        ax.text(x, row.mean * 100 + 1, f"{row.size:,}", ha="center", fontsize=9)
    ax.set(title=title, ylabel="Canceled (%)", **labels)


def plot_class_balance(train_bookings):
    fig, (balance_ax, hotel_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = train_bookings["is_canceled"].value_counts().sort_index()
    balance_ax.bar(["Not canceled", "Canceled"], counts)
    for x, count in enumerate(counts):
        balance_ax.text(x, count, f"{count:,} ({count / counts.sum():.1%})", ha="center", va="bottom")
    balance_ax.set(title="Class balance", ylabel="Bookings")

    hotels = train_bookings.groupby("hotel")["is_canceled"].agg(["size", "mean"])
    cancellation_bars(hotel_ax, hotels, "Cancellation by hotel")
    return fig


def plot_feature_patterns(train_bookings, training_features):
    fig, (lead_ax, weekday_ax, frequency_ax) = plt.subplots(1, 3, figsize=(18, 4.5))
    lead_groups = pd.cut(train_bookings["lead_time"], bins=[-1, 0, 7, 30, 90, 180, 365, np.inf],
                         labels=["0", "1-7", "8-30", "31-90", "91-180", "181-365", "366+"])
    lead = train_bookings.groupby(lead_groups, observed=True)["is_canceled"].agg(["size", "mean"])
    cancellation_bars(lead_ax, lead, "Cancellation by lead time", xlabel="Days before arrival")

    weekdays = arrival_dates(train_bookings).dt.day_name().str[:3]
    weekday = train_bookings.groupby(weekdays)["is_canceled"].agg(["size", "mean"]).reindex(WEEKDAYS)
    cancellation_bars(weekday_ax, weekday, "Cancellation by arrival weekday")

    for column in ["agent", "company", "country"]:
        counts = training_features[column].value_counts().drop("Unknown", errors="ignore")
        frequency_ax.plot(range(1, len(counts) + 1), counts, label=column)
    frequency_ax.axhline(50, color="gray", linestyle="--", label="50 bookings")
    frequency_ax.set(title="Category frequencies", xlabel="Rank within column", ylabel="Bookings", yscale="log")
    frequency_ax.legend()
    return fig


def plot_numeric_distributions(features, columns):
    """Plot numeric histograms, zooming ADR and guest axes at the 99th percentile."""
    rows = int(np.ceil(len(columns) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.5 * rows), squeeze=False)
    for ax, column in zip(axes.flat, columns):
        values = features[column].dropna()
        upper = values.quantile(0.99) if column in ("adr", "total_guests") else values.max()
        zoom = values.min() < upper < values.max()
        visible_range = (values.min(), upper) if zoom else None
        ax.hist(values, bins=35, range=visible_range)
        ax.set(title=column.replace("_", " "), xlabel="Value", ylabel="Bookings")
        if zoom:
            ax.set_xlabel("Value (up to 99th percentile)")
            ax.text(0.98, 0.95, f"{(values > upper).sum():,} values above axis\nmax = {values.max():g}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    return fig


def plot_numeric_correlations(features, target, columns):
    """Plot Pearson correlations between selected features and the target."""
    correlations = features[columns].assign(is_canceled=target).corr()
    labels = correlations.columns.str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(11, 9))
    heatmap = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels, title="Numeric correlations (train)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlations.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.6 else "black", fontsize=8)
    fig.colorbar(heatmap, ax=ax, label="Pearson correlation")
    return fig


def plot_experiment_results(results_table):
    fig, (f1_ax, size_ax) = plt.subplots(1, 2, figsize=(13, 5))
    order = results_table["experiment"].unique()
    f1 = results_table.pivot(index="experiment", columns="split", values="f1").loc[order]
    f1[["train", "validation"]].plot.barh(ax=f1_ax, xlim=(0, 0.9))
    f1_ax.set(title="F1: train vs validation", xlabel="F1", ylabel="")

    features = results_table.drop_duplicates("experiment").set_index("experiment")["features"]
    features.plot.barh(ax=size_ax)
    size_ax.set(title="Encoded feature count", xlabel="Features", ylabel="")
    return fig
