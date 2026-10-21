# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python (hotel-cancellation)
#     language: python
#     name: hotel-cancellation
# ---

# %% [markdown]
# # Hotel cancellation prediction
#
# **Adam Nolan · Oleksandr Hakavyi · Mykhailo Mas**
#
# A retrospective comparison of cancellation classifiers for two Portuguese hotels.
# Data: [Hotel Booking Demand on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), originally published by [António, de Almeida and Nunes (2019)](https://doi.org/10.1016/j.dib.2018.11.126).

# %% [markdown]
# ## 0. Frame the problem
#
# We estimate the probability that a historical booking was canceled (`is_canceled = 1`). This is supervised binary classification on records from two hotels, with arrivals in 2015–2017.
#
# The possible business use is a ranked list for a manager to call guests and confirm their plans. Higher recall finds more eventual cancellations, but contacting more guests also creates unnecessary calls. Because no staffing capacity or cost estimates are supplied, we report recall and precision at several contact budgets because no operational precision target is available. These are retrospective scenarios: the data cannot establish whether a call would prevent cancellation.
#
# **Scope.** Booking-time snapshots of all fields are unavailable. This study compares historical associations; it does not validate a system that can be deployed at booking creation.
#
# **Protocol.** Tune and compare probability ranking using ROC AUC, with average precision, Brier score and log loss as complementary measures. Choose the model on validation. Use validation F1 to set a reproducible reference threshold, not a business optimum. Report final test metrics only after these choices. A constant training-prior model provides the trivial reference.

# %% [markdown]
# ### Related work and our contribution
#
# | Source | Approach and evidence | Relevance and comparison limits |
# |---|---|---|
# | [António et al., 2019: dataset paper](https://doi.org/10.1016/j.dib.2018.11.126) | Documents the two-hotel booking datasets and variable definitions. | Source for data interpretation, not a model benchmark. |
# | [António et al., 2017: cancellation prediction](https://scielo.pt/pdf/tms/v13n2/13n2a03.pdf) | Compares classifiers on four resort hotels; Decision Forest is strongest in accuracy, with reported results above 90%. The final assessment uses a stratified 70/30 split. | Supports comparing tree ensembles. Different hotels, features and evaluation prevent a direct score comparison. |
# | [António et al., 2017: production prototype](https://ciencia.iscte-iul.pt/publications/predicting-hotel-bookings-cancellation-with-a-machine-learning-classification-model/44810) | Daily model training and an A/B evaluation in two hotels; reports fewer cancellations among contacted bookings. | Motivates guest follow-up. This is intervention evidence from another study, not an effect estimated from our CSV. |
#
# Our contribution is the controlled comparison of four familiar models with grouped evaluation, preprocessing experiments, feature sensitivity and contact-workload analysis. We do not introduce a new algorithm or claim state-of-the-art performance.

# %% [markdown]
# Import the data preparation code from the `src` folder.

# %%
import sys
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

project_root = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "src" / "hotel_cancellation").is_dir()
)
sys.path.insert(0, str(project_root / "src"))
preprocessing = import_module("hotel_cancellation.0_preprocessing")
modeling = import_module("hotel_cancellation.1_modeling")
evaluation = import_module("hotel_cancellation.2_evaluation")
from hotel_cancellation import plot_style
plt.rcParams.update(plot_style.NOTEBOOK_STYLE)

# %% [markdown]
# ## 1. Data
#
# Load the CSV. Check the number of rows and columns and view the first five bookings.

# %%
df = preprocessing.load_data()
print(df.shape)
df.head()

# %% [markdown]
# Check column types, the five columns with the most missing values, and the share of bookings: `0` = not canceled, `1` = canceled.

# %%
df.info()
display(df.isna().sum().sort_values(ascending=False).head())
display(df["is_canceled"].value_counts(normalize=True))

# %% [markdown]
# Prepare `X` and `y`. Remove `reservation_status` and `reservation_status_date`, which reveal the outcome; convert agent/company identifiers to categories; add total nights and guests. Imputation, scaling and encoding are fitted inside each training pipeline.

# %%
X, y = preprocessing.prepare_features(df)
print(X.shape, y.shape)

# %% [markdown]
# ### 1.0 One shared train / validation / test split
#
# Use one grouped split: approximately 60% train, 20% validation and 20% test. Groups are feature-only hashes, excluding the five fields listed in `GROUPING_EXCLUDED_COLUMNS`; these exclusions also protect the five-field ablation below. They are an overlap safeguard, not verified booking IDs. Targets never enter the group key.
#
# Every section uses the same rows **and training order**. Train is used for fitting and three-fold grouped CV. Validation is reused for preprocessing comparisons, model choice and the reference threshold. Test is reserved for the fixed final comparison and diagnostics. This is a historical holdout from the same dataset, not an external or prospective test; it was also inspected in earlier project versions.

# %%
datasets = preprocessing.split_data(X, y)
X_train, y_train = datasets["train"]
X_validation, y_validation = datasets["validation"]
X_test, y_test = datasets["test"]
X_train = X_train.sample(frac=1, random_state=72)
y_train = y_train.loc[X_train.index]
datasets["train"] = (X_train, y_train)

split_groups = {name: set(preprocessing.booking_groups(part_X))
                for name, (part_X, _) in datasets.items()}
assert split_groups["train"].isdisjoint(split_groups["validation"])
assert split_groups["train"].isdisjoint(split_groups["test"])
assert split_groups["validation"].isdisjoint(split_groups["test"])
cv_folds = preprocessing.grouped_cv(X_train, y_train, n_splits=3)
train_groups = preprocessing.booking_groups(X_train)
display(pd.DataFrame([
    {"part": name, "bookings": len(part_X), "share": len(part_X) / len(X),
     "canceled_share": part_y.mean(), "groups": len(split_groups[name])}
    for name, (part_X, part_y) in datasets.items()
]).round(3))
display(pd.DataFrame([
    {"fold": i + 1, "fit_rows": len(fit_rows), "held_out_rows": len(eval_rows),
     "canceled_share": y_train.iloc[eval_rows].mean(),
     "group_overlap": len(set(train_groups.iloc[fit_rows]) & set(train_groups.iloc[eval_rows]))}
    for i, (fit_rows, eval_rows) in enumerate(cv_folds)
]).round(3))

# %% [markdown]
# ## 1.1 Preprocessing study
#
# Two focused changes are compared with the original baselines: log lead time for logistic regression, and grouping rare agent/company/country categories for the forest. All five fits use the shared training rows and order. Validation is used for these exploratory comparisons; test is not scored here.

# %%
preprocessing_plots = import_module("hotel_cancellation.00_preprocessing_plots")
preprocessing_report_folder = project_root / "reports/preprocessing"
preprocessing_report_folder.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### Data quality and feature availability
#
# Inspect missing values and unusual rows on train. Zero-length stays and zero-guest records are retained because their business meaning is uncertain; negative ADR is flagged rather than silently filtered. The main task retains the historical population, so validation/test are not cleaned to improve scores. Grouping, rather than deleting all repeated rows, avoids treating every identical record as an erroneous duplicate.

# %%
prep_X_train, prep_y_train = X_train, y_train
prep_X_validation, prep_y_validation = X_validation, y_validation
prep_train_bookings = df.loc[X_train.index]

quality_flags = pd.DataFrame({
    "negative_adr": X_train["adr"].lt(0),
    "zero_guests": X_train["total_guests"].eq(0),
    "zero_nights": X_train["total_nights"].eq(0),
    "repeated_raw_record": prep_train_bookings.duplicated(keep=False),
})
display(pd.DataFrame({"train_rows": quality_flags.sum(), "share": quality_flags.mean()}).round(4))
display(X_train.isna().sum().sort_values(ascending=False).head().rename("missing_train"))

# %% [markdown]
# **Availability audit.** Definitions follow the [original dataset paper](https://doi.org/10.1016/j.dib.2018.11.126). The CSV does not timestamp each feature update.
#
# | Fields | What may change / why this matters | Treatment |
# |---|---|---|
# | `reservation_status`, `reservation_status_date` | Outcome and its date directly reveal the answer | Excluded from predictors; date used only for temporal label-availability checks |
# | `assigned_room_type`, `booking_changes` | Assignment and accumulated modifications can be recorded after booking creation | Retained for retrospective study; included in five-field removal experiment |
# | `days_in_waiting_list`, `adr`, `deposit_type` | Waiting history, price and payment/deposit information need a defined observation date | Same removal experiment; no booking-time availability claim |
# | `required_car_parking_spaces`, `total_of_special_requests` | Requests may change over the booking lifetime | Retained; historical values do not establish information available at creation |
# | Remaining predictors and derived totals | Definitions are understandable, but creation-time snapshots are not verified | Used retrospectively; not certified deployment-safe |
#
# Removing only five fields is a sensitivity check, not proof that the remaining set is safe for prospective use.

# %% [markdown]
# Log lead time compresses a long-tailed variable and may help a linear model. Rare-category grouping reduces the one-hot matrix; categories occurring fewer than 50 times in each fit become `Other`, while missing values remain `Unknown`. Both variants are later included in grouped tuning, so the preprocessing study is connected to final model selection.

# %% [markdown]
# ### Train-only exploration
#
# Class balance, hotel differences and booking timing motivate the experiments. These are descriptive associations, not causal effects.

# %%
prep_figure = preprocessing_plots.plot_class_balance(prep_train_bookings)
plot_style.show_figure(prep_figure, "class_balance")

# %% [markdown]
# Earlier bookings and high-cardinality categories are useful candidates for preprocessing. Category frequencies and group sizes help distinguish common patterns from small samples.

# %%
prep_figure = preprocessing_plots.plot_feature_patterns(prep_train_bookings, prep_X_train)
plot_style.show_figure(prep_figure, "feature_patterns")

# %% [markdown]
# ### Baselines and preprocessing variants
#
# Fit five pipelines on the shared train set; compare their validation metrics and encoded feature counts.

# %%
prep_results = []


def record_preprocessing_results(name, model, training_features, validation_features):
    train_metrics = evaluation.get_metrics(model, training_features, prep_y_train)
    validation_metrics = evaluation.get_metrics(model, validation_features, prep_y_validation)
    feature_count = len(model[:-1].get_feature_names_out())

    prep_results.append({
        "experiment": name, "split": "train", "features": feature_count,
        **train_metrics,
    })
    prep_results.append({
        "experiment": name, "split": "validation", "features": feature_count,
        **validation_metrics,
    })


# %% [markdown]
# **Logistic regression baseline.** Median imputation and scaling for numbers; missing-category imputation and one-hot encoding for categories.

# %%
prep_logistic_baseline = modeling.train_logistic_regression(
    prep_X_train, prep_y_train,
    preprocessor=preprocessing.build_preprocessor(),
)
record_preprocessing_results("LR baseline", prep_logistic_baseline, prep_X_train, prep_X_validation)

# %% [markdown]
# **Log lead time.** Apply `log1p` inside the numeric pipeline. The saved model therefore contains the transformation.

# %%
prep_logistic_log = modeling.train_logistic_regression(
    prep_X_train, prep_y_train,
    preprocessor=preprocessing.build_preprocessor(log_lead=True),
)
record_preprocessing_results("LR + log lead", prep_logistic_log, prep_X_train, prep_X_validation)

# %% [markdown]
# **Tree and forest baselines.** Use the existing depth/leaf constraints, fitted on the same rows.

# %%
prep_tree_baseline = modeling.train_decision_tree(
    prep_X_train, prep_y_train,
    preprocessor=preprocessing.build_preprocessor(),
)
record_preprocessing_results("Tree baseline", prep_tree_baseline, prep_X_train, prep_X_validation)

prep_forest_baseline = modeling.train_random_forest(
    prep_X_train, prep_y_train,
    preprocessor=preprocessing.build_preprocessor(),
)
record_preprocessing_results("Forest baseline", prep_forest_baseline, prep_X_train, prep_X_validation)

# %% [markdown]
# **Rare-category forest.** Learn frequent categories only from the training input of each fit.

# %%
prep_forest_rare = modeling.train_random_forest(
    prep_X_train, prep_y_train,
    preprocessor=preprocessing.build_preprocessor(rare_min_count=50),
)
record_preprocessing_results("Forest + rare", prep_forest_rare, prep_X_train, prep_X_validation)

# %% [markdown]
# ### Preprocessing results
#
# Compare each variant with its own baseline. These untuned F1 results do not by themselves determine the final ranking model.

# %%
prep_results_table = pd.DataFrame(prep_results)
prep_results_table.to_csv(preprocessing_report_folder / "results.csv", index=False)

prep_validation_results = prep_results_table[prep_results_table["split"] == "validation"]
prep_validation_results = prep_validation_results.set_index("experiment")
prep_metric_columns = ["features", "accuracy", "precision", "recall", "f1", "log_loss"]
display(prep_validation_results[prep_metric_columns].round(4))

# %% [markdown]
# #### Which changes help, and do they beat the tree?
#
# Compare F1 and the number of transformed columns. The results table also includes
# precision, recall and log loss, so F1 is not the only evidence available.

# %%
prep_figure = preprocessing_plots.plot_experiment_results(prep_results_table)
plot_style.show_figure(prep_figure, "preprocessing_comparison")

# %%
lr_gain = prep_validation_results.loc["LR + log lead", "f1"] - prep_validation_results.loc["LR baseline", "f1"]
rf_gain = prep_validation_results.loc["Forest + rare", "f1"] - prep_validation_results.loc["Forest baseline", "f1"]
feature_reduction = 1 - prep_validation_results.loc["Forest + rare", "features"] / prep_validation_results.loc["Forest baseline", "features"]
display(Markdown(
    f"Log lead time changes LR validation F1 by **{lr_gain:+.3f}**. "
    f"Rare grouping changes forest F1 by **{rf_gain:+.3f}** and reduces encoded features by **{feature_reduction:.1%}**. "
    "The tuned comparisons below test whether these choices also help probability ranking."
))

# %% [markdown]
# ## 2. Training

# %% [markdown]
# ### 2.1 Experimental setup
#
# Compare four model families: logistic regression, decision tree, random forest and gradient boosting. Use the same three grouped folds throughout. Fit preprocessing inside every fold. Select hyperparameters by mean CV ROC AUC, matching the final ranking objective; report F1 at 0.5 as a diagnostic.
#
# Searches are deliberately small for reproducibility on a laptop. They are comparisons of tested configurations, not exhaustive rankings of algorithm families.

# %% [markdown]
# ### 2.2 Logistic regression
#
# A linear reference model estimates cancellation probability as
#
# $$P(Y=1\mid x)=\sigma(\beta_0+\beta^\top x).$$
#
# The features here are the **encoded** inputs. Coefficients are fitted using regularized log loss. We test `C`, class weighting and log lead time. Coefficient signs describe conditional changes in predicted log odds; they are not causal effects or general feature-importance scores.

# %%
logistic_model = prep_logistic_baseline
print("Encoded baseline features:", len(logistic_model[:-1].get_feature_names_out()))

# %% [markdown]
# ### 2.3 Decision tree
#
# A tree learns feature thresholds that reduce weighted Gini impurity,
#
# $$G(S)=1-\sum_k p_k^2.$$
#
# Unlike a linear model, it captures nonlinear interactions. The original baseline limits depth to 8 and requires 20 observations per leaf. The plot shows its first three levels; tuning checks a small set of alternative constraints.

# %%
tree_model = prep_tree_baseline
tree_classifier = tree_model[-1]

# %%
from sklearn.tree import plot_tree

tree_feature_names = tree_model[:-1].get_feature_names_out()
fig, ax = plt.subplots(figsize=(18, 7))
annotations = plot_tree(
    tree_classifier, feature_names=tree_feature_names,
    class_names=["Not canceled", "Canceled"],
    filled=False, rounded=True, max_depth=2, fontsize=10, ax=ax,
)
for annotation in annotations:
    annotation.set_color(plot_style.TEXT)
    box = annotation.get_bbox_patch()
    if box is not None:
        accent = plot_style.PRIMARY if "class = Canceled" in annotation.get_text() else plot_style.BLUE
        box.set_facecolor(plot_style.PREVIEW_BACKGROUND)
        box.set_edgecolor(accent)
        box.set_linewidth(1.8)
    if annotation.arrow_patch is not None:
        annotation.arrow_patch.set_color(plot_style.MUTED)
ax.set_title("Decision tree: first three levels")
plot_style.show_figure(fig, "decision_tree")

# %% [markdown]
# **Optional geometric illustration.** On the same two-dimensional toy data, logistic regression learns a linear boundary and a tree learns axis-aligned regions. This illustration is separate from the hotel experiments.

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier



X_toy, y_toy = make_moons(
    n_samples=400,
    noise = 0.20,
    random_state=72
)

models = {
    "Logistic Regression ": LogisticRegression(),
    "Decision Tree ": DecisionTreeClassifier(
        max_depth=5,
        random_state=72
    )
}


x_min, x_max = X_toy[:, 0].min() - 0.5, X_toy[:, 0].max() + 0.5
y_min, y_max = X_toy[:, 1].min() - 0.5, X_toy[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_toy, y_toy)

    predictions = model.predict(grid).reshape(xx.shape)

    ax.contourf(
        xx, yy, predictions,
        alpha=0.25,
        cmap=ListedColormap([plot_style.BLUE, plot_style.PRIMARY])
    )

    ax.contour(
        xx, yy, predictions,
        levels=[0.5],
        colors=plot_style.MUTED,
        linewidths=1.5
    )

    ax.scatter(
        X_toy[:, 0], X_toy[:, 1],
        c=y_toy,
        cmap=ListedColormap([plot_style.BLUE, plot_style.PRIMARY]),
        edgecolors=plot_style.MUTED,
        s=25
    )

    ax.set_title(name)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

fig.suptitle("Decision boundaries", x=0.02, ha="left", fontsize=23, fontweight="bold")
plot_style.show_figure(fig, "decision_boundaries")

# %% [markdown]
# ### 2.4 Random forest
#
# Random Forest averages probabilities from trees fitted on bootstrap samples, with random subsets of candidate features at each split. This can reduce the instability of a single tree. The original baseline uses 100 trees, depth 12 and at least 10 rows per leaf. Tuning compares depths 12/24, leaf sizes 5/10 and rare-category grouping with fixed ensemble size and feature subsampling.

# %%
forest_model = prep_forest_baseline

# %% [markdown]
# ### 2.5 Gradient boosting
#
# Boosting adds trees sequentially to improve the current model under log loss. Learning rate controls each tree's contribution; depth and the number of stages control complexity. We retain the original shallow baseline and test both `max_features="sqrt"` and `None`, rather than fixing feature subsampling throughout tuning.

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

boosting_model = Pipeline([
    ("preprocessor", preprocessing.build_preprocessor()),
    ("classifier", GradientBoostingClassifier(
        n_estimators=60, learning_rate=0.1, max_depth=2,
        max_features="sqrt", random_state=72,
    )),
]).fit(X_train, y_train)
models = {
    "Logistic Regression": logistic_model, "Decision Tree": tree_model,
    "Random Forest": forest_model, "Gradient Boosting": boosting_model,
}
baseline_configurations = pd.DataFrame({
    "Baseline configuration": {name: modeling.configuration_label(model) for name, model in models.items()}
})
with pd.option_context("display.max_colwidth", None):
    display(baseline_configurations)

# %% [markdown]
# ### 2.6 Baseline models: grouped CV on train
#
# The four baseline pipelines use the same three `cv_folds` from §1.0. No feature group crosses a fold boundary. ROC AUC evaluates ranking; precision, recall and F1 use threshold 0.5. Fold standard deviations describe variation across these partitions and are not deployment confidence intervals.

# %%
from sklearn.model_selection import cross_validate

models = {
    "Logistic Regression": logistic_model, "Decision Tree": tree_model,
    "Random Forest": forest_model, "Gradient Boosting": boosting_model,
}
scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision",
           "f1": "f1", "recall": "recall", "precision": "precision"}
cv_results = {}
for name, model in models.items():
    cv_results[name] = cross_validate(
        model, X_train, y_train, scoring=scoring, cv=cv_folds, n_jobs=1,
    )

# %%
cv_summary = pd.DataFrame([
    {"model": name, **{f"{metric}_{stat}": getattr(np, stat)(scores[f"test_{metric}"])
                      for metric in scoring for stat in ("mean", "std")}}
    for name, scores in cv_results.items()
]).set_index("model")
display(cv_summary[["roc_auc_mean", "roc_auc_std", "precision_mean", "recall_mean", "f1_mean"]].rename(columns={
    "roc_auc_mean": "CV ROC AUC", "roc_auc_std": "CV std",
    "precision_mean": "Precision at 0.5", "recall_mean": "Recall at 0.5", "f1_mean": "F1 at 0.5",
}).round(4))

# %% [markdown]
# ROC AUC measures ranking across thresholds; F1, precision and recall here use 0.5. A model with low recall at that threshold can still rank bookings well. The next step tests hyperparameters before comparing the tuned candidates on validation.

# %% [markdown]
# ### 2.7 Hyperparameter search
#
# All searches select mean ROC AUC on the same grouped CV folds. The table distinguishes varied parameters from fixed settings. Regularization and leaf support limit complexity; preprocessing choices test whether transformations improve ranking. The grids are deliberately small and do not establish the best possible configuration of each algorithm.
#
# The RF search varies depth, leaf size and rare grouping. It fixes 100 trees and `max_features=0.5`, which performed well in the preceding exploratory search. Current results therefore compare these eight RF configurations, not every feature-subsampling strategy.

# %%
from sklearn.model_selection import GridSearchCV

logistic_param_grid = {
    "columntransformer": [preprocessing.build_preprocessor(log_lead=value) for value in (False, True)],
    "logisticregression__C": [0.1, 1.0],
    "logisticregression__class_weight": [None, "balanced"],
}
decision_tree_param_grid = {
    "decisiontreeclassifier__max_depth": [8, 16],
    "decisiontreeclassifier__min_samples_leaf": [5, 20],
}
forest_param_grid = {
    "columntransformer__categorical__rare_categories__min_count": [None, 50],
    "randomforestclassifier__max_depth": [12, 24],
    "randomforestclassifier__min_samples_leaf": [5, 10],
    "randomforestclassifier__max_features": [0.5],
    "randomforestclassifier__n_estimators": [100],
}
boosting_param_grid = {
    "classifier__learning_rate": [0.1, 0.2], "classifier__max_features": [None, "sqrt"],
    "classifier__n_estimators": [60], "classifier__max_depth": [3],
}
parameter_grids = {"Logistic Regression": logistic_param_grid, "Decision Tree": decision_tree_param_grid,
                   "Random Forest": forest_param_grid, "Gradient Boosting": boosting_param_grid}
display(modeling.search_space_table(parameter_grids))
search_scoring = {"roc_auc": "roc_auc", "f1": "f1", "neg_log_loss": "neg_log_loss"}

# %% [markdown]
# #### Logistic Regression Tuning

# %%
logistic_search = GridSearchCV(
    logistic_model, logistic_param_grid, scoring=search_scoring, refit="roc_auc",
    cv=cv_folds, n_jobs=1, return_train_score=True,
)
logistic_search.fit(X_train, y_train)
best_logistic_model = logistic_search.best_estimator_

# %% [markdown]
# #### Decision Tree Tuning

# %%
tree_search = GridSearchCV(
    tree_model, decision_tree_param_grid, scoring=search_scoring, refit="roc_auc",
    cv=cv_folds, n_jobs=1, return_train_score=True,
)
tree_search.fit(X_train, y_train)
best_tree_model = tree_search.best_estimator_

# %% [markdown]
# #### Random Forest Tuning

# %%
forest_search = GridSearchCV(
    forest_model, forest_param_grid, scoring=search_scoring, refit="roc_auc",
    cv=cv_folds, n_jobs=1, return_train_score=True,
)
forest_search.fit(X_train, y_train)
best_forest_model = forest_search.best_estimator_

# %% [markdown]
# **RF complexity comparison.** Solid curves show held-out CV ROC AUC and dashed curves show training AUC for the same fits. Each panel fixes the rare-grouping choice. This separates changes in depth and leaf support without changing folds or ensemble size.

# %%
modeling.plot_forest_search(forest_search.cv_results_)

# %% [markdown]
# #### Gradient Boosting tuning

# %%
boosting_search = GridSearchCV(
    boosting_model, boosting_param_grid, scoring=search_scoring, refit="roc_auc",
    cv=cv_folds, n_jobs=1, return_train_score=True,
)
boosting_search.fit(X_train, y_train)
best_boosting_model = boosting_search.best_estimator_

# %%
searches = {"Logistic Regression": logistic_search, "Decision Tree": tree_search,
            "Random Forest": forest_search, "Gradient Boosting": boosting_search}
search_rows = []
for name, search in searches.items():
    idx = search.best_index_
    search_rows.append({
        "model": name, "candidates": len(search.cv_results_["params"]), "folds": len(cv_folds),
        "train_auc": search.cv_results_["mean_train_roc_auc"][idx],
        "cv_auc": search.cv_results_["mean_test_roc_auc"][idx],
        "cv_auc_std": search.cv_results_["std_test_roc_auc"][idx],
        "cv_f1_at_0.5": search.cv_results_["mean_test_f1"][idx],
    })
search_summary = pd.DataFrame(search_rows).set_index("model")
search_summary["auc_gap"] = search_summary["train_auc"] - search_summary["cv_auc"]
display(search_summary[["candidates", "folds", "train_auc", "cv_auc", "cv_auc_std", "auc_gap"]]
        .sort_values("cv_auc", ascending=False).rename(columns={
            "train_auc": "Train ROC AUC", "cv_auc": "CV ROC AUC",
            "cv_auc_std": "CV std", "auc_gap": "Train–CV gap",
        }).round(4))

# %% [markdown]
# The summary describes tuned candidates on train-only CV. A larger train–CV gap suggests sensitivity to complexity. Validation scores are reported separately in §3.3 after fitting each selected configuration on the full train set.

# %%
optimized_models = {
    "Logistic Regression": best_logistic_model, "Decision Tree": best_tree_model,
    "Random Forest": best_forest_model, "Gradient Boosting": best_boosting_model,
}

# %% [markdown]
# ## 3. Evaluation
#
# Choose the model, reference threshold and diagnostics on validation. Then report fixed test scores, paired group-bootstrap uncertainty and subgroup errors. The final temporal check is an exploratory sensitivity experiment, not another tuning source.

# %% [markdown]
# ### 3.1 Metrics
#
# Canceled (`1`) is the positive class.
#
# | Metric | Purpose |
# |---|---|
# | ROC AUC | Main ranking criterion; independent of a decision threshold |
# | Average precision | Precision–recall summary; compare with the cancellation prevalence |
# | Brier score / log loss | Quality of probability predictions; lower is better |
# | Precision | Share of contacted/flagged bookings that actually cancel |
# | Recall | Share of actual cancellations that are contacted/flagged |
# | F1 | Reference balance of precision and recall, not an economic optimum |
# | Accuracy | Overall correct share; interpret alongside the prior baseline |
#
# Brier score also reflects discrimination and outcome uncertainty, so a lower score alone does not prove better calibration.

# %% [markdown]
# ### 3.2 Baseline models on validation, threshold 0.5
#
# Evaluate the four initial pipelines on the shared validation set. The constant training-prior model predicts the majority class at 0.5. These results describe the baselines, before selecting tuned configurations.

# %%
from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(strategy="prior").fit(X_train, y_train)
baseline_proba = pd.DataFrame({name: model.predict_proba(X_validation)[:, 1]
                              for name, model in models.items()})
baseline_proba["Training prior"] = dummy_model.predict_proba(X_validation)[:, 1]
baseline_comparison = evaluation.compare_proba(baseline_proba, y_validation)
display(baseline_comparison[["roc_auc", "precision", "recall", "f1"]].rename(columns={
    "roc_auc": "Validation ROC AUC", "precision": "Precision at 0.5",
    "recall": "Recall at 0.5", "f1": "F1 at 0.5",
}).round(3))

# %% [markdown]
# Low recall at 0.5 does not by itself show poor ranking or underfitting. Compare ROC AUC and probability losses, then examine thresholds for the selected tuned model. Training error includes irreducible noise and possible missing information.

# %% [markdown]
# ### 3.3 Tuned models: CV and validation
#
# **CV ROC AUC** is the mean held-out score inside train, used to choose parameters. **Validation ROC AUC** evaluates the selected configuration fitted on all train rows, on the separate validation set. Choose the model by validation ROC AUC before selecting a threshold.
#
# ROC AUC has no decision threshold. The selected configurations below differ from the baseline configurations in §2.5–2.6.

# %%
model_names = list(optimized_models)
validation_proba = pd.DataFrame(
    {name: model.predict_proba(X_validation)[:, 1] for name, model in optimized_models.items()},
    index=X_validation.index,
)
validation_proba["Training prior"] = dummy_model.predict_proba(X_validation)[:, 1]
validation_comparison = evaluation.compare_proba(validation_proba, y_validation).sort_values("roc_auc", ascending=False)
selected_configurations = pd.DataFrame({
    "Selected configuration": {name: modeling.configuration_label(model) for name, model in optimized_models.items()},
    "CV ROC AUC": search_summary["cv_auc"],
    "CV std": search_summary["cv_auc_std"],
    "Validation ROC AUC": validation_comparison.loc[model_names, "roc_auc"],
}).sort_values("Validation ROC AUC", ascending=False)
with pd.option_context("display.max_colwidth", None):
    display(selected_configurations.round(4))

# %%
evaluation.plot_roc_pr_curves(validation_proba[model_names], y_validation, title="(validation set)")

# %%
final_name = validation_comparison.loc[model_names, "roc_auc"].idxmax()
final_proba_validation = validation_proba[final_name]
print("Selected tuned model:", final_name)

# %%
final_model = optimized_models[final_name]

# %% [markdown]
# ### 3.4 Threshold and manager workload
#
# Use the best validation F1 threshold on a 0.05–0.95 grid (step 0.01) as a reproducible classification reference. It is not an optimal hotel policy: contact capacity, costs and the effect of calls are unknown. The separate ranked-budget table shows how recall changes when contacting 10–50% of bookings. These percentages are sensitivity scenarios, not staffing requirements. Equal scores are resolved by input order, without using outcomes.

# %%
threshold_metrics = evaluation.threshold_table(y_validation, final_proba_validation)
final_threshold = float(threshold_metrics["f1"].idxmax())
evaluation.plot_threshold_curves(
    threshold_metrics, marked={"default": 0.5, "validation F1 reference": final_threshold},
    title=f"{final_name}: validation threshold trade-off",
)
display(threshold_metrics.loc[sorted({0.3, 0.4, 0.5, 0.6, 0.7, final_threshold})].round(3))
validation_workload = evaluation.contact_workload(y_validation, final_proba_validation)
display(validation_workload.round(3))
print(f"Reference threshold: {final_threshold:.2f}")

# %% [markdown]
# ### 3.5 Calibration on validation
#
# Inspect reliability bins with sample counts for the tuned models. Bin means depend on their population and binning; they do not guarantee calibration in a different period or subgroup.

# %%
evaluation.plot_calibration(validation_proba[model_names], y_validation, title="Calibration: validation")
display(evaluation.calibration_bins(y_validation, final_proba_validation).round(3))

# %% [markdown]
# ### 3.6 Final historical test score
#
# Freeze the model and reference threshold. Use 1,000 bootstrap resamples of whole feature groups for 95% percentile intervals. Repeated draws retain every row of the selected group; the paired AUC comparison uses the same draws for both models. These intervals condition on the fitted models and the grouping assumptions. They do not include retraining variability or future distribution shift.

# %%
test_proba = final_model.predict_proba(X_test)[:, 1]
test_groups = preprocessing.booking_groups(X_test)
final_results = evaluation.bootstrap_metrics(
    y_test, test_proba, threshold=final_threshold, groups=test_groups,
)
final_results["at default 0.5"] = pd.Series(evaluation.metrics_from_proba(y_test, test_proba, 0.5))
final_results["training prior"] = pd.Series(
    evaluation.metrics_from_proba(y_test, dummy_model.predict_proba(X_test)[:, 1])
)
display(final_results.round(3))
display(evaluation.contact_workload(y_test, test_proba).round(3))

# %%
test_pred = (test_proba >= final_threshold).astype(int)
evaluation.plot_confusion_matrix(
    final_model, X_test, y_test, threshold=final_threshold,
    title=f"{final_name}: test set, threshold {final_threshold:.2f}",
)

# %%
display(Markdown(
    f"Historical holdout: **{final_name}**, reference threshold **{final_threshold:.2f}**; "
    f"ROC AUC **{final_results.loc['roc_auc', 'estimate']:.3f}**, "
    f"recall **{final_results.loc['recall', 'estimate']:.3f}**, "
    f"precision **{final_results.loc['precision', 'estimate']:.3f}**. "
    "The workload table evaluates fixed ranking scenarios; no policy is selected from test."
))

# %% [markdown]
# ### 3.7 Paired model comparison
#
# Compare the final model with the runner-up selected on validation. Neither model nor threshold is changed after this test comparison.

# %%
second_name = validation_comparison.loc[model_names, "roc_auc"].sort_values(ascending=False).index[1]
second_proba = optimized_models[second_name].predict_proba(X_test)[:, 1]
auc_differences = evaluation.bootstrap_auc_difference(
    y_test, test_proba, second_proba, groups=test_groups,
)
evaluation.plot_bootstrap_difference(auc_differences, label=f"Test AUC: {final_name} - {second_name}")
low, high = np.quantile(auc_differences, [0.025, 0.975])
display(Markdown(
    f"Paired group-bootstrap 95% interval for AUC difference: **[{low:.3f}, {high:.3f}]**. "
    f"The interval {'excludes' if low > 0 or high < 0 else 'includes'} zero under the stated resampling assumptions. "
    "This is not evidence of superiority on future hotels or periods."
))

# %% [markdown]
# ### 3.8 Model interpretation and limitations
#
# Explain the selected tuned model using validation permutation importance and joint feature removal, then describe its errors on test. The analyses measure model reliance and predictive behavior, not causal effects.
#
# #### 3.8.1 Permutation importance on validation
#
# Shuffle one original column at a time and measure the drop in ROC AUC. Correlated columns can mask each other's contribution. Joint removal below checks five fields together; neither analysis certifies booking-time availability.

# %%
permutation_table = evaluation.plot_permutation_importance(
    final_model, X_validation, y_validation, sample=3000, n_repeats=3,
)
display(permutation_table.round(4))

# %% [markdown]
# #### 3.8.2 Joint removal of five fields
#
# Remove `assigned_room_type`, `booking_changes`, `days_in_waiting_list`, `adr` and `deposit_type` from a clone of the selected pipeline and refit on train. The split's grouping key excludes these fields, protecting this comparison from identical remaining-input records across splits. Parameters stay fixed to isolate the removal; the reduced model is not independently tuned.

# %%
from sklearn.base import clone

review_fields = preprocessing.GROUPING_EXCLUDED_COLUMNS
reduced_model = clone(final_model).fit(X_train.drop(columns=review_fields), y_train)
reduced_proba = reduced_model.predict_proba(X_validation.drop(columns=review_fields))[:, 1]
feature_sensitivity = pd.DataFrame({
    "full inputs": evaluation.metrics_from_proba(y_validation, final_proba_validation),
    "five fields removed": evaluation.metrics_from_proba(y_validation, reduced_proba),
}).T
display(feature_sensitivity.round(3))

# %% [markdown]
# #### 3.8.3 Errors by test segment
#
# Report precision and recall for groups with at least 200 test bookings. Interpret recall alongside group size and cancellation prevalence. These diagnostics describe weaknesses and do not change the fitted model or reference threshold.

# %%
X_test_segments = X_test.assign(
    lead_time_group=pd.cut(
        X_test["lead_time"],
        bins=[-1, 7, 30, 90, 180, 365, np.inf],
        labels=["0-7", "8-30", "31-90", "91-180", "181-365", "365+"],
    )
)
segment_tables = {
    column: evaluation.metrics_by_segment(X_test_segments, y_test, test_proba, column, threshold=final_threshold)
    for column in ["hotel", "market_segment", "deposit_type", "lead_time_group"]
}
evaluation.plot_segment_recall(segment_tables, overall_recall=final_results.loc["recall", "estimate"])
pd.concat(segment_tables).round(3)

# %%
segment_results = pd.concat(segment_tables)
lowest_recall = segment_results.sort_values("recall").head(3)
display(Markdown(
    "Lowest observed recall among the displayed segments: " + "; ".join(
        f"**{column}: {segment} ({row['recall']:.3f})**"
        for (column, segment), row in lowest_recall.iterrows()
    ) + ". Additional payment history or a different contact budget might help, but neither effect is established here."
))

# %% [markdown]
# ### 3.9 Exploratory temporal check
#
# Use fixed original LR/RF baselines, not the configurations tuned using mixed years. Train on arrivals before July 2016 whose outcome status date is already known before July 1; use July–December 2016 arrivals with labels known before January 2017 for model/threshold choice; evaluate on 2017 arrivals. Do not refit or retune using these test outcomes.
#
# This is a time-ordered sensitivity analysis with fewer training rows, not a replacement for the shared split. Per-feature historical snapshots remain unavailable; 2017 has incomplete seasonal coverage. These same records were inspected in the historical experiment, so this is not an independent external test or proof of booking-time performance.

# %%
from calendar import month_name

month_numbers = {name: i for i, name in enumerate(month_name) if name}
arrival = pd.to_datetime({"year": df.arrival_date_year,
                          "month": df.arrival_date_month.map(month_numbers),
                          "day": df.arrival_date_day_of_month})
status_date = pd.to_datetime(df.reservation_status_date)
temporal_masks = {
    "train": (arrival < "2016-07-01") & (status_date < "2016-07-01"),
    "validation": (arrival >= "2016-07-01") & (arrival < "2017-01-01") & (status_date < "2017-01-01"),
    "test": arrival >= "2017-01-01",
}
temporal_groups = {name: set(preprocessing.booking_groups(X.loc[mask]))
                   for name, mask in temporal_masks.items()}
assert temporal_groups["train"].isdisjoint(temporal_groups["validation"] | temporal_groups["test"])
assert temporal_groups["validation"].isdisjoint(temporal_groups["test"])
display(pd.DataFrame([
    {"part": name, "rows": int(mask.sum()), "cancel_rate": y.loc[mask].mean(),
     "arrival_start": str(arrival.loc[mask].min().date()),
     "arrival_end": str(arrival.loc[mask].max().date())}
    for name, mask in temporal_masks.items()
]))
time_train, time_val, time_test = (temporal_masks[name] for name in ("train", "validation", "test"))
temporal_models = {name: clone(model).fit(X.loc[time_train], y.loc[time_train])
                   for name, model in {"Logistic Regression": logistic_model, "Random Forest": forest_model}.items()}
time_val_proba = pd.DataFrame({name: model.predict_proba(X.loc[time_val])[:, 1]
                              for name, model in temporal_models.items()})
time_comparison = evaluation.compare_proba(time_val_proba, y.loc[time_val])
display(time_comparison.round(3))
time_name = time_comparison.roc_auc.idxmax()
time_threshold = evaluation.threshold_table(y.loc[time_val], time_val_proba[time_name]).f1.idxmax()
time_proba = temporal_models[time_name].predict_proba(X.loc[time_test])[:, 1]
time_prior = DummyClassifier(strategy="prior").fit(X.loc[time_train], y.loc[time_train])
time_results = pd.DataFrame({
    time_name: evaluation.metrics_from_proba(y.loc[time_test], time_proba, time_threshold),
    "training prior": evaluation.metrics_from_proba(y.loc[time_test], time_prior.predict_proba(X.loc[time_test])[:, 1]),
}).T
display(time_results.round(3))
print("Temporal model:", time_name, "validation reference threshold:", time_threshold)
display(evaluation.contact_workload(y.loc[time_test], time_proba).round(3))

# %% [markdown]
# ### 3.10 Save the fitted pipeline
#
# Save the historical model, reference threshold, input schema and evaluation settings. The F1 threshold is not a recommended calling policy. All learned preprocessing, including a selected log transform or category grouping, is inside the pipeline.

# %%
import joblib
import sklearn

model_dir = project_root / "models"
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "final_model.joblib"
joblib.dump({
    "name": final_name, "model": final_model, "threshold": final_threshold,
    "threshold_role": "validation F1 reference; no operational capacity specified",
    "feature_columns": X_train.columns.tolist(), "sklearn_version": sklearn.__version__,
    "scope": "retrospective historical prediction", "cv_folds": len(cv_folds), "seed": 72,
}, model_path, compress=3)
print("Saved:", model_path.relative_to(project_root))

# %% [markdown]
# ## 4. Conclusion
#
# We compared four model families on one shared historical split. Grouped CV keeps repeated records together, and tuning includes preprocessing choices. ROC AUC selects ranking models; probability losses and calibration bins describe their probability predictions. The reference threshold and contact-budget scenarios make the recall/workload trade-off explicit without assuming a precision target or claiming revenue gains.
#
# The selected model and its test scores are reported in §3.6. The five-field removal check has a modest effect on historical ranking, which does not certify the remaining fields as booking-time inputs. The temporal baseline comparison is weaker than the mixed-period comparison; differences in training size, model configuration and seasonal coverage prevent attributing the whole gap to time.
#
# **Limits and next steps.** This dataset has no complete booking-time change history and represents only two hotels in 2015–2017. The temporal check is exploratory; group-bootstrap intervals do not capture all dependencies or deployment uncertainty. Before manager calls could be evaluated prospectively, we would need verified snapshots, a real contact budget and an intervention study. More extensive tuning or new data could help; this notebook does not establish that additional complexity is necessary.
