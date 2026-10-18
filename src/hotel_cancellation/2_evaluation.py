import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
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


def compare_models(models, X_train, y_train, X_test, y_test):
    """Return one row per model: test metrics, ROC AUC and the train/test error gap."""
    rows = {}
    for name, model in models.items():
        test = get_metrics(model, X_test, y_test)
        train = get_metrics(model, X_train, y_train)
        test["roc_auc"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        test["train_error_rate"] = train["error_rate"]
        test["gap (test - train error)"] = test["error_rate"] - train["error_rate"]
        rows[name] = test

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
