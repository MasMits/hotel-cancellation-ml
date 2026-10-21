import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hotel_cancellation import plot_style as style
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


def get_metrics(model, X, y):
    """Return the main classification metrics of a model on one dataset."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "log_loss": log_loss(y, y_proba),
        "error_rate": 1 - accuracy_score(y, y_pred),
    }


@style.themed
def plot_confusion_matrix(model, X, y, title="Confusion matrix", threshold=None):
    prediction = model.predict(X) if threshold is None else (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(
        y, prediction,
        display_labels=["Not canceled", "Canceled"],
        cmap=style.HEATMAP,
        colorbar=False,
    )
    for value, label in zip(display.confusion_matrix.flat, display.text_.flat):
        red, green, blue, _ = display.im_.cmap(display.im_.norm(value))
        label.set_color("white" if .2126 * red + .7152 * green + .0722 * blue < .5 else style.TEXT)
    plt.title(title)
    style.show_figure(plt.gcf(), "confusion_matrix")


@style.themed
def plot_roc_curve(model, X, y, title="ROC curve"):
    RocCurveDisplay.from_estimator(model, X, y, name="Logistic regression")
    plt.plot([0, 1], [0, 1], color=style.MUTED, linestyle="--", label="Random guess (AUC = 0.5)")
    plt.legend()
    plt.title(title)
    style.show_figure(plt.gcf(), "roc_curve")


def compare_models(models, X_train, y_train, X_eval, y_eval):
    """Return one row per model: metrics on the evaluation set, ROC AUC and the train/eval error gap."""
    rows = {}
    for name, model in models.items():
        scores = get_metrics(model, X_eval, y_eval)
        train = get_metrics(model, X_train, y_train)
        scores["roc_auc"] = roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])
        scores["train_error_rate"] = train["error_rate"]
        scores["gap (eval - train error)"] = scores["error_rate"] - train["error_rate"]
        rows[name] = scores

    return pd.DataFrame(rows).T


@style.themed
def plot_roc_curves(models, X, y, title="ROC curves"):
    """Draw the ROC curves of several models on one plot."""
    _, ax = plt.subplots(figsize=(9, 7))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X, y, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], color=style.MUTED, linestyle="--", label="Random guess (AUC = 0.5)")
    ax.legend()
    ax.set_title(title)
    style.show_figure(plt.gcf(), "roc_curves")


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
    return pd.DataFrame(
        {name: metrics_from_proba(y, probas[name], threshold) for name in probas.columns}
    ).T


@style.themed
def plot_roc_pr_curves(probas, y, title=""):
    """ROC curve (left) and precision-recall curve (right) for every column of a probability table."""
    _, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(15, 6))
    for name in probas.columns:
        RocCurveDisplay.from_predictions(y, probas[name], name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y, probas[name], name=name, ax=ax_pr)
    ax_roc.plot([0, 1], [0, 1], color=style.MUTED, linestyle="--", label="Random guess (AUC = 0.5)")
    ax_pr.axhline(y.mean(), color=style.MUTED, linestyle="--", label=f"Random guess (AP = {y.mean():.2f})")
    ax_roc.legend()
    ax_pr.legend()
    ax_roc.set_title(f"ROC curve {title}")
    ax_pr.set_title(f"Precision-recall curve {title}")
    style.show_figure(plt.gcf(), "roc_pr_curves")


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


@style.themed
def plot_threshold_curves(table, marked=None, title="Metrics against threshold"):
    """Plot precision, recall and F1 against the threshold, with chosen thresholds as vertical lines."""
    _, ax = plt.subplots(figsize=(11, 6))
    table[["precision", "recall", "f1"]].plot(ax=ax)
    for label, threshold in (marked or {}).items():
        ax.axvline(threshold, color=style.MUTED, linestyle="--")
        ax.text(threshold, 0.04, f" {label} = {threshold:.2f}", rotation=90, va="bottom", color=style.TEXT, fontsize=10)
    ax.set_xlabel("Threshold (flag as canceled if probability >= threshold)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    style.show_figure(plt.gcf(), "threshold_curves")


@style.themed
def plot_calibration(probas, y, n_bins=10, title="Calibration"):
    """Reliability diagram: predicted probability against the observed share of cancellations."""
    _, ax = plt.subplots(figsize=(9, 7))
    for name in probas.columns:
        observed, predicted = calibration_curve(y, probas[name], n_bins=n_bins)
        brier = brier_score_loss(y, probas[name])
        ax.plot(predicted, observed, "o-", label=f"{name} (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], color=style.MUTED, linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability of cancellation")
    ax.set_ylabel("Observed share of cancellations")
    ax.set_title(title)
    ax.legend()
    style.show_figure(plt.gcf(), "calibration")


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
        for attempt in range(100):
            draws = rng.integers(0, n_groups, n_groups)
            weights = np.bincount(draws, minlength=n_groups)[codes]
            if np.unique(y[weights > 0]).size == 2:
                yield weights
                break
        else:
            raise ValueError("Unable to draw a bootstrap sample with both classes")


def bootstrap_metrics(y, y_proba, threshold=0.5, n_boot=1000, random_state=72, groups=None):
    """Percentile CIs for a fixed model; resample whole groups when supplied."""
    y = np.asarray(y)
    y_proba = np.asarray(y_proba)
    samples = []
    for weights in _bootstrap_weights(y, groups, n_boot, random_state):
        samples.append(metrics_from_proba(y, y_proba, threshold, sample_weight=weights))
    samples = pd.DataFrame(samples)

    return pd.DataFrame({
        "estimate": metrics_from_proba(y, y_proba, threshold),
        "ci_low": samples.quantile(0.025),
        "ci_high": samples.quantile(0.975),
    })


def bootstrap_auc_difference(y, proba_a, proba_b, n_boot=1000, random_state=72, groups=None):
    """Paired AUC difference on identical resampled groups for fixed models."""
    y = np.asarray(y)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    differences = []
    for weights in _bootstrap_weights(y, groups, n_boot, random_state):
        differences.append(
            roc_auc_score(y, proba_a, sample_weight=weights)
            - roc_auc_score(y, proba_b, sample_weight=weights)
        )
    return np.array(differences)


def contact_workload(y, y_proba, shares=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Rank by risk and evaluate illustrative contact budgets, not business targets.

    Stable input order breaks equal-score ties without looking at outcomes.
    """
    y = np.asarray(y)
    proba = np.asarray(y_proba)
    ranking = np.argsort(-proba, kind="stable")
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
        bookings=("outcome", "size"), predicted=("predicted", "mean"),
        observed=("outcome", "mean"),
    )


@style.themed
def plot_bootstrap_difference(differences, label="AUC difference"):
    """Histogram of a bootstrap distribution with its 95% interval and zero marked."""
    low, high = np.quantile(differences, [0.025, 0.975])
    _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(differences, bins=40, color=style.PRIMARY, alpha=0.7)
    ax.axvline(0, color=style.MUTED, linestyle="--", label="No difference")
    ax.axvspan(low, high, color=style.BLUE, alpha=0.25, label=f"95% interval [{low:.4f}, {high:.4f}]")
    ax.set_title("Paired bootstrap comparison")
    ax.set_xlabel(label)
    ax.set_ylabel("Bootstrap samples")
    ax.legend()
    style.show_figure(plt.gcf(), "bootstrap_difference")


def metrics_by_segment(X, y, y_proba, column, threshold=0.5, min_rows=200):
    """Precision and recall inside each group of one column, largest groups first."""
    data = pd.DataFrame({
        "segment": X[column].astype(str).to_numpy(),
        "y": np.asarray(y),
        "pred": (np.asarray(y_proba) >= threshold).astype(int),
    })
    rows = []
    for segment, group in data.groupby("segment"):
        if len(group) < min_rows:
            continue
        rows.append({
            "segment": segment,
            "bookings": len(group),
            "cancel_rate": group["y"].mean(),
            "precision": precision_score(group["y"], group["pred"], zero_division=0),
            "recall": recall_score(group["y"], group["pred"], zero_division=0),
        })
    return pd.DataFrame(rows).set_index("segment").sort_values("bookings", ascending=False)


@style.themed
def plot_segment_recall(tables, overall_recall):
    """Horizontal bars of recall per segment, one panel per column."""
    _, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4))
    axes = np.atleast_1d(axes)
    for ax, (column, table) in zip(axes, tables.items()):
        table["recall"].sort_values().plot.barh(ax=ax, color=style.PRIMARY)
        ax.axvline(overall_recall, color=style.MUTED, linestyle="--", label="Overall recall")
        ax.set_xlim(0, 1)
        ax.set_title(column.replace("_", " ").capitalize())
        ax.set_xlabel("Recall")
        ax.set_ylabel("")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=2)
    style.show_figure(plt.gcf(), "segment_recall")


@style.themed
def plot_permutation_importance(model, X, y, n_repeats=5, top=15, sample=5000, random_state=72):
    """Drop in ROC AUC when one original column is shuffled, measured on (a sample of) the given data."""
    if sample is not None and len(X) > sample:
        X = X.sample(sample, random_state=random_state)
        y = y.loc[X.index]
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=n_repeats,
        random_state=random_state, n_jobs=1,
    )
    importance = pd.DataFrame(
        {"mean": result.importances_mean, "std": result.importances_std},
        index=X.columns,
    ).sort_values("mean", ascending=False).head(top)

    _, ax = plt.subplots(figsize=(8, 6))
    importance["mean"][::-1].plot.barh(ax=ax, xerr=importance["std"][::-1], color=style.PRIMARY, ecolor=style.MUTED)
    ax.set_xlabel("Drop in ROC AUC when the column is shuffled")
    ax.set_title(f"Permutation importance (top {top})")
    style.show_figure(plt.gcf(), "permutation_importance")
    return importance
