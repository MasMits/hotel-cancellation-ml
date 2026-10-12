import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
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
