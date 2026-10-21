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
    y_proba = model.predict_proba(X)[:, 1]  # probability of "canceled"

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "log_loss": log_loss(y, y_proba),
        "error_rate": 1 - accuracy_score(y, y_pred),
    }


def plot_confusion_matrix(model, X, y, title="Confusion matrix"):
    ConfusionMatrixDisplay.from_estimator(
        model, X, y,
        display_labels=["Not canceled", "Canceled"],
        cmap="Purples",
        colorbar=False,
    )
    plt.title(title)
    plt.show()


def plot_roc_curve(model, X, y, title="ROC curve"):
    RocCurveDisplay.from_estimator(model, X, y, name="Logistic regression")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.5)")
    plt.legend()
    plt.title(title)
    plt.show()


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


def plot_roc_curves(models, X, y, title="ROC curves"):
    """Draw the ROC curves of several models on one plot."""
    _, ax = plt.subplots(figsize=(6, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X, y, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.5)")
    ax.legend()
    ax.set_title(title)
    plt.show()


def metrics_from_proba(y, y_proba, threshold=0.5):
    """Return the main metrics when bookings with probability >= threshold are flagged as canceled."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "brier": brier_score_loss(y, y_proba),
    }


def compare_proba(probas, y, threshold=0.5):
    """One row of metrics per column of a probability table."""
    return pd.DataFrame(
        {name: metrics_from_proba(y, probas[name], threshold) for name in probas.columns}
    ).T


def plot_roc_pr_curves(probas, y, title=""):
    """ROC curve (left) and precision-recall curve (right) for every column of a probability table."""
    _, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name in probas.columns:
        RocCurveDisplay.from_predictions(y, probas[name], name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y, probas[name], name=name, ax=ax_pr)
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.5)")
    ax_pr.axhline(y.mean(), color="k", linestyle="--", label=f"Random guess (AP = {y.mean():.2f})")
    ax_roc.legend()
    ax_pr.legend()
    ax_roc.set_title(f"ROC curve {title}")
    ax_pr.set_title(f"Precision-recall curve {title}")
    plt.tight_layout()
    plt.show()


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
    _, ax = plt.subplots(figsize=(8, 5))
    table[["precision", "recall", "f1"]].plot(ax=ax)
    for label, threshold in (marked or {}).items():
        ax.axvline(threshold, color="grey", linestyle="--")
        ax.text(threshold, 0.02, f" {label} = {threshold:.2f}", rotation=90, va="bottom")
    ax.set_xlabel("Threshold (flag as canceled if probability >= threshold)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.show()


def plot_calibration(probas, y, n_bins=10, title="Calibration"):
    """Reliability diagram: predicted probability against the observed share of cancellations."""
    _, ax = plt.subplots(figsize=(6, 6))
    for name in probas.columns:
        observed, predicted = calibration_curve(y, probas[name], n_bins=n_bins)
        brier = brier_score_loss(y, probas[name])
        ax.plot(predicted, observed, "o-", label=f"{name} (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability of cancellation")
    ax.set_ylabel("Observed share of cancellations")
    ax.set_title(title)
    ax.legend()
    plt.show()


def bootstrap_metrics(y, y_proba, threshold=0.5, n_boot=1000, random_state=72):
    """Point estimate and 95% bootstrap confidence interval of each metric.

    The test set is resampled with replacement n_boot times; the model is not retrained.
    """
    y = np.asarray(y)
    y_proba = np.asarray(y_proba)
    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(y), len(y))
        samples.append(metrics_from_proba(y[rows], y_proba[rows], threshold))
    samples = pd.DataFrame(samples)

    return pd.DataFrame({
        "estimate": metrics_from_proba(y, y_proba, threshold),
        "ci_low": samples.quantile(0.025),
        "ci_high": samples.quantile(0.975),
    })


def bootstrap_auc_difference(y, proba_a, proba_b, n_boot=1000, random_state=72):
    """Bootstrap distribution of AUC(a) - AUC(b), both models scored on the same resampled rows."""
    y = np.asarray(y)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    rng = np.random.default_rng(random_state)
    differences = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(y), len(y))
        differences.append(roc_auc_score(y[rows], proba_a[rows]) - roc_auc_score(y[rows], proba_b[rows]))
    return np.array(differences)


def plot_bootstrap_difference(differences, label="AUC difference"):
    """Histogram of a bootstrap distribution with its 95% interval and zero marked."""
    low, high = np.quantile(differences, [0.025, 0.975])
    _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(differences, bins=40, color="tab:purple", alpha=0.7)
    ax.axvline(0, color="k", linestyle="--", label="No difference")
    ax.axvspan(low, high, color="grey", alpha=0.2, label=f"95% interval [{low:.4f}, {high:.4f}]")
    ax.set_xlabel(label)
    ax.set_ylabel("Bootstrap samples")
    ax.legend()
    plt.show()


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


def plot_segment_recall(tables, overall_recall):
    """Horizontal bars of recall per segment, one panel per column."""
    _, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4))
    axes = np.atleast_1d(axes)
    for ax, (column, table) in zip(axes, tables.items()):
        table["recall"].sort_values().plot.barh(ax=ax, color="tab:purple")
        ax.axvline(overall_recall, color="k", linestyle="--", label="Overall recall")
        ax.set_xlim(0, 1)
        ax.set_title(column)
        ax.set_xlabel("Recall")
        ax.set_ylabel("")
    axes[0].legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_permutation_importance(model, X, y, n_repeats=5, top=15, sample=5000, random_state=72):
    """Drop in ROC AUC when one original column is shuffled, measured on (a sample of) the given data."""
    if sample is not None and len(X) > sample:
        X = X.sample(sample, random_state=random_state)
        y = y.loc[X.index]
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1,
    )
    importance = pd.DataFrame(
        {"mean": result.importances_mean, "std": result.importances_std},
        index=X.columns,
    ).sort_values("mean", ascending=False).head(top)

    _, ax = plt.subplots(figsize=(8, 6))
    importance["mean"][::-1].plot.barh(ax=ax, xerr=importance["std"][::-1], color="tab:purple")
    ax.set_xlabel("Drop in ROC AUC when the column is shuffled")
    ax.set_title(f"Permutation importance (top {top})")
    plt.show()
    return importance
