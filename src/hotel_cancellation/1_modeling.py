"""Baseline pipelines and compact reporting for the grouped parameter searches."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from hotel_cancellation import plot_style as style


def train_logistic_regression(X_train, y_train, preprocessor, max_iter=1000, random_state=72):
    model = make_pipeline(preprocessor, LogisticRegression(max_iter=max_iter, random_state=random_state))
    return model.fit(X_train, y_train)


def train_decision_tree(X_train, y_train, preprocessor, max_depth=8, min_samples_leaf=20, random_state=72):
    model = make_pipeline(preprocessor, DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state,
    ))
    return model.fit(X_train, y_train)


def train_random_forest(X_train, y_train, preprocessor, n_estimators=100,
                        max_depth=12, min_samples_leaf=10, random_state=72):
    model = make_pipeline(preprocessor, RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        random_state=random_state, n_jobs=2,
    ))
    return model.fit(X_train, y_train)


def search_space_table(grids):
    """Describe the actual grids without printing estimator objects or long dictionaries."""
    reasons = {
        "columntransformer": "Lead-time transformation", "C": "Regularization strength",
        "class_weight": "Class weighting", "max_depth": "Tree complexity",
        "min_samples_leaf": "Minimum leaf support", "max_features": "Feature subsampling",
        "min_count": "Rare-category grouping", "n_estimators": "Ensemble size",
        "learning_rate": "Contribution of each stage",
    }
    rows = []
    for model, grid in grids.items():
        for path, values in grid.items():
            parameter = path.split("__")[-1]
            if parameter == "columntransformer":
                parameter = "lead_time"
                shown = ["log1p" if "log_lead" in value.transformers[0][1].named_steps else "raw"
                         for value in values]
            else:
                shown = ["disabled" if parameter == "min_count" and value is None
                         else "all" if parameter == "max_features" and value is None
                         else str(value) for value in values]
            rows.append({"model": model, "parameter": parameter, "values": ", ".join(shown),
                         "role": "Varied" if len(values) > 1 else "Fixed",
                         "purpose": reasons.get(path.split("__")[-1], "")})
    return pd.DataFrame(rows)


def configuration_label(model):
    """Short, explicit description of a fitted pipeline's selected configuration."""
    classifier = model[-1]
    if isinstance(classifier, LogisticRegression):
        lead = "log lead" if "log_lead" in model[0].named_transformers_["numeric"].named_steps else "raw lead"
        return f"C={classifier.C:g}, weights={classifier.class_weight}, {lead}"
    if isinstance(classifier, DecisionTreeClassifier):
        return f"depth={classifier.max_depth}, leaf={classifier.min_samples_leaf}"
    if isinstance(classifier, RandomForestClassifier):
        rare = model[0].named_transformers_["categorical"].named_steps["rare_categories"].min_count
        return (f"trees={classifier.n_estimators}, depth={classifier.max_depth}, "
                f"leaf={classifier.min_samples_leaf}, features={classifier.max_features}, rare={rare}")
    return (f"stages={classifier.n_estimators}, depth={classifier.max_depth}, "
            f"rate={classifier.learning_rate}, features={classifier.max_features if classifier.max_features is not None else 'all'}")


@style.themed
def plot_forest_search(cv_results):
    """Compare train/CV AUC for depth and leaf size within each grouping variant."""
    results = pd.DataFrame(cv_results)
    rare_key = "param_columntransformer__categorical__rare_categories__min_count"
    depth_key = "param_randomforestclassifier__max_depth"
    leaf_key = "param_randomforestclassifier__min_samples_leaf"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, rare, title in zip(axes, [None, 50], ["No rare grouping", "Rare categories grouped"]):
        mask = results[rare_key].isna() if rare is None else results[rare_key].eq(rare)
        for leaf, color in zip([5, 10], [style.PRIMARY, style.BLUE]):
            subset = results[mask & results[leaf_key].eq(leaf)].sort_values(depth_key)
            depths = subset[depth_key].astype(int)
            ax.plot(depths, subset.mean_test_roc_auc, color=color, marker="o", label=f"CV, leaf ≥ {leaf}")
            ax.plot(depths, subset.mean_train_roc_auc, color=color, linestyle="--", alpha=.65,
                    label=f"Train, leaf ≥ {leaf}")
        ax.set(title=title, xlabel="Maximum tree depth", xticks=[12, 24])
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2, fontsize=10)
    axes[0].set_ylabel("ROC AUC")
    style.show_figure(fig, "forest_complexity")
