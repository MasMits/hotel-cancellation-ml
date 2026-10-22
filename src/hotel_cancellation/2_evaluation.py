"""Metrics, bootstrap intervals, contact workloads and evaluation plots."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hotel_cancellation.plot_style import show_figure


def positive_proba(model, X):
    """Predicted probability of the positive class (canceled)."""
    return model.predict_proba(X)[:, 1]


def get_metrics(model, X, y):
    """Return the main classification metrics of a model on one dataset."""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return {
        "accuracy": accuracy,
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "log_loss": log_loss(y, positive_proba(model, X)),
        "error_rate": 1 - accuracy,
    }


def summarize_cv(cv_results, metrics):
    """Mean and standard deviation of each metric across folds, one row per model."""
    return pd.DataFrame([
        {"model": name, **{f"{metric}_{stat}": getattr(np, stat)(scores[f"test_{metric}"])
                           for metric in metrics for stat in ("mean", "std")}}
        for name, scores in cv_results.items()
    ]).set_index("model")


def plot_confusion_matrix(model, X, y, title="Confusion matrix", threshold=None):
    """Confusion matrix at the model's default rule or at a given probability threshold."""
    prediction = model.predict(X) if threshold is None else (positive_proba(model, X) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y, prediction, display_labels=["Not canceled", "Canceled"],
                                            cmap="Blues", colorbar=False)
    plt.title(title)
    show_figure(plt.gcf(), "confusion_matrix")


def metrics_from_proba(y, y_proba, threshold=0.5, sample_weight=None):
    """Return the main metrics when bookings with probability >= threshold are flagged as canceled."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred, sample_weight=sample_weight),
        "precision": precision_score(y, y_pred, zero_division=0, sample_weight=sample_weight),
        "recall": recall_score(y, y_pred, zero_division=0, sample_weight=sample_weight),
        "f1": f1_score(y, y_pred, zero_division=0, sample_weight=sample_weight),
        "roc_auc": roc_auc_score(y, y_proba, sample_weight=sample_weight),
        "average_precision": average_precision_score(y, y_proba, sample_weight=sample_weight),
        "brier": brier_score_loss(y, y_proba, sample_weight=sample_weight),
        "log_loss": log_loss(y, y_proba, sample_weight=sample_weight, labels=[0, 1]),
    }


def compare_proba(probas, y, threshold=0.5):
    """One row of metrics per column of a probability table."""
    return pd.DataFrame({name: metrics_from_proba(y, probas[name], threshold) for name in probas.columns}).T


def compare_errors(probas, y, threshold=0.5):
    """False positives/negatives and paired error overlap, aligning labels by index."""
    if (not probas.index.is_unique or not y.index.is_unique
            or len(probas) != len(y) or not probas.index.isin(y.index).all()):
        raise ValueError("Probabilities and target must have the same unique row indices.")
    target = y.reindex(probas.index)
    if target.isna().any():
        raise ValueError("Target labels must not be missing.")
    predictions = probas.ge(threshold)
    errors = predictions.ne(target, axis="index")
    summary = pd.DataFrame({
        "false_positives": predictions.loc[target.eq(0)].sum(),
        "false_negatives": (~predictions.loc[target.eq(1)]).sum(),
        "error_rate": errors.mean(),
    })
    pairs = pd.DataFrame([
        {"first": first, "second": second,
         "both_wrong": (errors[first] & errors[second]).sum(),
         "only_first_wrong": (errors[first] & ~errors[second]).sum(),
         "only_second_wrong": (~errors[first] & errors[second]).sum(),
         "disagreement_share": (errors[first] != errors[second]).mean()}
        for first, second in combinations(probas.columns, 2)
    ])
    return summary, pairs


def plot_roc_pr_curves(probas, y, title=""):
    """ROC curve (left) and precision-recall curve (right) for every column of a probability table."""
    _, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(15, 6))
    for name in probas.columns:
        RocCurveDisplay.from_predictions(y, probas[name], name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y, probas[name], name=name, ax=ax_pr)
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess (AUC = 0.5)")
    ax_pr.axhline(y.mean(), color="gray", linestyle="--", label=f"Random guess (AP = {y.mean():.2f})")
    for ax, name in [(ax_roc, "ROC curve"), (ax_pr, "Precision-recall curve")]:
        ax.legend()
        ax.set_title(f"{name} {title}")
    show_figure(plt.gcf(), "roc_pr_curves")


def threshold_table(y, y_proba, thresholds=None):
    """Precision, recall, F1 and the share of flagged bookings for a range of thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    y_proba = np.asarray(y_proba)
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "flagged_share": y_pred.mean(),
        })
    return pd.DataFrame(rows).set_index("threshold")


def plot_threshold_curves(table, marked=None, title="Metrics against threshold"):
    """Plot precision, recall and F1 against the threshold, with chosen thresholds as vertical lines."""
    _, ax = plt.subplots(figsize=(11, 6))
    table[["precision", "recall", "f1"]].plot(ax=ax)
    for label, threshold in (marked or {}).items():
        ax.axvline(threshold, color="gray", linestyle="--")
        ax.text(threshold, 0.04, f" {label} = {threshold:.2f}", rotation=90, va="bottom", fontsize=10)
    ax.set(xlabel="Threshold (flag as canceled if probability >= threshold)", ylim=(0, 1), title=title)
    show_figure(plt.gcf(), "threshold_curves")


def plot_calibration(probas, y, n_bins=10, title="Calibration"):
    """Reliability diagram: predicted probability against the observed share of cancellations."""
    _, ax = plt.subplots(figsize=(9, 7))
    for name in probas.columns:
        observed, predicted = calibration_curve(y, probas[name], n_bins=n_bins)
        brier = brier_score_loss(y, probas[name])
        ax.plot(predicted, observed, "o-", label=f"{name} (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfectly calibrated")
    ax.set(xlabel="Predicted probability of cancellation", ylabel="Observed share of cancellations", title=title)
    ax.legend()
    show_figure(plt.gcf(), "calibration")


def _bootstrap_weights(y, groups, n_boot, random_state):
    """Resample whole groups; weights exactly represent repeated group draws."""
    if groups is None:
        groups = np.arange(len(y))
    if len(groups) != len(y):
        raise ValueError("groups and y must have the same length and row order")
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = codes.max() + 1
    rng = np.random.default_rng(random_state)
    for _ in range(n_boot):
        # A retry is needed only when a draw contains a single outcome class.
        for _attempt in range(100):
            draws = rng.integers(0, n_groups, n_groups)
            weights = np.bincount(draws, minlength=n_groups)[codes]
            if np.unique(y[weights > 0]).size == 2:
                yield weights
                break
        else:
            raise ValueError("Unable to draw a bootstrap sample with both classes")


def bootstrap_metrics(y, y_proba, threshold=0.5, n_boot=1000, random_state=72, groups=None):
    """Percentile CIs for a fixed model; resample whole groups when supplied."""
    y, y_proba = np.asarray(y), np.asarray(y_proba)
    samples = pd.DataFrame([metrics_from_proba(y, y_proba, threshold, sample_weight=weights)
                            for weights in _bootstrap_weights(y, groups, n_boot, random_state)])
    return pd.DataFrame({
        "estimate": metrics_from_proba(y, y_proba, threshold),
        "ci_low": samples.quantile(0.025),
        "ci_high": samples.quantile(0.975),
    })


def bootstrap_auc_difference(y, proba_a, proba_b, n_boot=1000, random_state=72, groups=None):
    """Paired AUC difference on identical resampled groups for fixed models."""
    y, proba_a, proba_b = np.asarray(y), np.asarray(proba_a), np.asarray(proba_b)
    return np.array([roc_auc_score(y, proba_a, sample_weight=weights) - roc_auc_score(y, proba_b, sample_weight=weights)
                     for weights in _bootstrap_weights(y, groups, n_boot, random_state)])


def contact_workload(y, y_proba, shares=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Rank by risk and evaluate illustrative contact budgets, not business targets.

    Stable input order breaks equal-score ties without looking at outcomes.
    """
    y = np.asarray(y)
    ranking = np.argsort(-np.asarray(y_proba), kind="stable")
    rows = []
    for share in shares:
        if not 0 < share <= 1:
            raise ValueError("Contact shares must be in (0, 1]")
        n_contacts = int(np.ceil(len(y) * share))
        captured = int(y[ranking[:n_contacts]].sum())
        rows.append({
            "contact_share": n_contacts / len(y), "contacts": n_contacts,
            "cancellations_found": captured, "unnecessary_contacts": n_contacts - captured,
            "precision": captured / n_contacts,
            "recall": captured / y.sum() if y.sum() else 0.0,
        })
    return pd.DataFrame(rows)


def calibration_bins(y, y_proba, n_bins=10):
    """Reliability estimates with bin counts; no claim of universal calibration."""
    table = pd.DataFrame({"outcome": np.asarray(y), "predicted": np.asarray(y_proba)})
    table["bin"] = pd.cut(table.predicted, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    return table.groupby("bin", observed=True).agg(
        bookings=("outcome", "size"), predicted=("predicted", "mean"), observed=("outcome", "mean"))


def plot_contact_workload(table):
    """Precision and recall for fixed validation contact budgets."""
    _, ax = plt.subplots(figsize=(8, 5))
    table.plot(x="contact_share", y=["precision", "recall"], marker="o", ax=ax)
    ax.plot(table["contact_share"], table["contact_share"], "--", color="gray", label="Random selection: recall")
    ax.set(xlabel="Share of bookings contacted", ylabel="Score", ylim=(0, 1.05),
           title="Contact budget trade-off (validation)")
    ax.legend()
    show_figure(plt.gcf(), "contact_workload")


def plot_bootstrap_difference(differences, label="AUC difference"):
    """Histogram of a bootstrap distribution with its 95% interval and zero marked."""
    low, high = np.quantile(differences, [0.025, 0.975])
    _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(differences, bins=40)
    ax.axvline(0, color="gray", linestyle="--", label="No difference")
    ax.axvspan(low, high, color="C1", alpha=0.25, label=f"95% interval [{low:.4f}, {high:.4f}]")
    ax.set(title="Paired bootstrap comparison", xlabel=label, ylabel="Bootstrap samples")
    ax.legend()
    show_figure(plt.gcf(), "bootstrap_difference")


def recall_interval(y, prediction, groups, n_boot=1000, random_state=72):
    """Pointwise percentile interval from resampling whole groups within one segment."""
    positives = np.asarray(y) == 1
    counts = pd.DataFrame({"positives": positives, "found": positives & np.asarray(prediction, dtype=bool),
                           "group": np.asarray(groups)}).groupby("group", sort=False).sum().to_numpy()
    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_boot):
        total, found = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        if total:
            samples.append(found / total)
    return np.quantile(samples, [0.025, 0.975]) if samples else np.array([np.nan, np.nan])


def metrics_by_segment(X, y, y_proba, column, threshold=0.5, min_rows=200, groups=None):
    """Segment precision/recall and group-bootstrap recall intervals, largest groups first."""
    data = pd.DataFrame({
        "segment": X[column].astype(str).to_numpy(),
        "y": np.asarray(y),
        "pred": (np.asarray(y_proba) >= threshold).astype(int),
        "group": np.arange(len(y)) if groups is None else np.asarray(groups),
    })
    rows = []
    for segment, group in data.groupby("segment"):
        if len(group) < min_rows:
            continue
        low, high = recall_interval(group["y"], group["pred"], group["group"])
        rows.append({"segment": segment, "bookings": len(group), "cancellations": group["y"].sum(),
                     "cancel_rate": group["y"].mean(),
                     "precision": precision_score(group["y"], group["pred"], zero_division=0),
                     "recall": recall_score(group["y"], group["pred"], zero_division=0),
                     "recall_ci_low": low, "recall_ci_high": high})
    return pd.DataFrame(rows).set_index("segment").sort_values("bookings", ascending=False)


def plot_segment_recall(tables, overall_recall):
    """Recall points and pointwise 95% group-bootstrap intervals per segment."""
    _, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (column, table) in zip(axes, tables.items()):
        ordered = table.sort_values("recall")
        positions = np.arange(len(ordered))
        ax.hlines(positions, ordered["recall_ci_low"], ordered["recall_ci_high"], color="C0", linewidth=2)
        ax.scatter(ordered["recall"], positions, label="Recall with 95% CI", zorder=3)
        labels = [f"{segment} (canceled: {int(row.cancellations)})" for segment, row in ordered.iterrows()]
        ax.set(yticks=positions, yticklabels=labels)
        ax.tick_params(axis="y", labelsize=9)
        ax.axvline(overall_recall, color="gray", linestyle="--", label="Overall recall")
        ax.set(xlim=(0, 1), title=column.replace("_", " ").capitalize(), xlabel="Recall", ylabel="")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=2)
    show_figure(plt.gcf(), "segment_recall")


def plot_permutation_importance(model, X, y, n_repeats=5, top=15, sample=5000, random_state=72):
    """Drop in ROC AUC when one original column is shuffled, measured on (a sample of) the given data."""
    if sample is not None and len(X) > sample:
        X = X.sample(sample, random_state=random_state)
        y = y.loc[X.index]
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                    random_state=random_state, n_jobs=1)
    importance = (pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std}, index=X.columns)
                  .sort_values("mean", ascending=False).head(top))

    _, ax = plt.subplots(figsize=(8, 6))
    importance["mean"][::-1].plot.barh(ax=ax, xerr=importance["std"][::-1])
    ax.set(xlabel="Drop in ROC AUC when the column is shuffled", title=f"Permutation importance (top {top})")
    show_figure(plt.gcf(), "permutation_importance")
    return importance
