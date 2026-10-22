"""Baseline pipelines and compact reporting for the grouped parameter searches."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from hotel_cancellation.plot_style import show_figure


def train_logistic_regression(X_train, y_train, preprocessor, max_iter=1000, random_state=72):
    classifier = LogisticRegression(max_iter=max_iter, random_state=random_state)
    return make_pipeline(preprocessor, classifier).fit(X_train, y_train)


def train_decision_tree(X_train, y_train, preprocessor, max_depth=8, min_samples_leaf=20, random_state=72):
    classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                        random_state=random_state)
    return make_pipeline(preprocessor, classifier).fit(X_train, y_train)


def train_random_forest(X_train, y_train, preprocessor, n_estimators=100,
                        max_depth=12, min_samples_leaf=10, random_state=72):
    # n_jobs=-1 uses every CPU core; with a fixed random_state the fitted forest is the same.
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1)
    return make_pipeline(preprocessor, classifier).fit(X_train, y_train)


def train_gradient_boosting(X_train, y_train, preprocessor, n_estimators=60, learning_rate=0.1,
                            max_depth=2, max_features="sqrt", random_state=72):
    classifier = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                            max_depth=max_depth, max_features=max_features,
                                            random_state=random_state)
    return make_pipeline(preprocessor, classifier).fit(X_train, y_train)


def grouped_grid_search(model, grid, X, y, folds):
    """Grid search on fixed grouped folds; refit the configuration with the best mean ROC AUC."""
    return GridSearchCV(
        model, grid, scoring={"roc_auc": "roc_auc", "f1": "f1", "neg_log_loss": "neg_log_loss"},
        refit="roc_auc", cv=folds, n_jobs=1, return_train_score=True,
    ).fit(X, y)


def search_summary(searches):
    """Train and CV scores of each search's selected configuration, with the train-CV ROC AUC gap."""
    best_scores = {"train_auc": "mean_train_roc_auc", "cv_auc": "mean_test_roc_auc",
                   "cv_auc_std": "std_test_roc_auc", "cv_f1_at_0.5": "mean_test_f1"}
    summary = pd.DataFrame([
        {"model": name, "candidates": len(search.cv_results_["params"]), "folds": search.n_splits_,
         **{label: search.cv_results_[key][search.best_index_] for label, key in best_scores.items()}}
        for name, search in searches.items()
    ]).set_index("model")
    summary["auc_gap"] = summary["train_auc"] - summary["cv_auc"]
    return summary


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
    none_labels = {"min_count": "disabled", "max_features": "all"}
    for model, grid in grids.items():
        for path, values in grid.items():
            parameter = path.split("__")[-1]
            if parameter == "columntransformer":
                shown = ["log1p" if "log_lead" in value.transformers[0][1].named_steps else "raw"
                         for value in values]
            else:
                shown = [none_labels.get(parameter, "None") if value is None else str(value) for value in values]
            rows.append({"model": model, "parameter": "lead_time" if parameter == "columntransformer" else parameter,
                         "values": ", ".join(shown), "role": "Varied" if len(values) > 1 else "Fixed",
                         "purpose": reasons.get(parameter, "")})
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
            f"rate={classifier.learning_rate}, features={classifier.max_features or 'all'}")


def feature_importance_table(models, top=10):
    """Top encoded inputs by absolute LR coefficient or tree impurity importance."""
    tables = []
    for name, model in models.items():
        classifier = model[-1]
        linear = isinstance(classifier, LogisticRegression)
        values = classifier.coef_[0] if linear else classifier.feature_importances_
        table = pd.DataFrame({"feature": model[:-1].get_feature_names_out(), "value": values})
        table = table.loc[table["value"].abs().nlargest(top).index]
        table.insert(0, "model", name)
        table["measure"] = "coefficient" if linear else "impurity importance"
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def save_experiments(folder, models, searches, cv_results, probabilities, target, metadata):
    """Save available fitted models, CV/search traces and indexed validation predictions."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": models,
        "baseline_cv_results": cv_results,
        "search_cv_results": {name: search.cv_results_ for name, search in searches.items()},
        "search_best_params": {name: search.best_params_ for name, search in searches.items()},
        "metadata": metadata,
    }, folder / "experiments.joblib", compress=3)
    predictions = probabilities.copy()
    predictions.insert(0, "is_canceled", target.reindex(predictions.index))
    predictions.to_csv(folder / "validation_predictions.csv", index_label="row_index")


def plot_forest_search(cv_results):
    """Compare train/CV AUC for depth and leaf size within each grouping variant."""
    results = pd.DataFrame(cv_results)
    rare_key = "param_columntransformer__categorical__rare_categories__min_count"
    depth_key = "param_randomforestclassifier__max_depth"
    leaf_key = "param_randomforestclassifier__min_samples_leaf"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, rare, title in zip(axes, [None, 50], ["No rare grouping", "Rare categories grouped"]):
        mask = results[rare_key].isna() if rare is None else results[rare_key].eq(rare)
        for leaf in sorted(results[leaf_key].astype(int).unique()):
            subset = results[mask & results[leaf_key].eq(leaf)].sort_values(depth_key)
            depths = subset[depth_key].astype(int)
            (cv_line,) = ax.plot(depths, subset.mean_test_roc_auc, marker="o", label=f"CV, leaf >= {leaf}")
            ax.plot(depths, subset.mean_train_roc_auc, "--", color=cv_line.get_color(), label=f"Train, leaf >= {leaf}")
        ax.set(title=title, xlabel="Maximum tree depth", xticks=sorted(results[depth_key].astype(int).unique()))
        ax.legend(fontsize=9)
    axes[0].set_ylabel("ROC AUC")
    show_figure(fig, "forest_complexity")
