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
#     display_name: .venv (3.14.7)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hotel cancellation prediction
#
# Explore hotel bookings and prepare data to predict cancellations.

# %% [markdown]
# ## 0. Frame the problem
#
# **Business objective.** A canceled booking can leave a room empty and lose revenue. About 37% of the bookings in our data are canceled (section 1). The hotel wants an estimate, before the guest arrives, of how likely each booking is to be canceled.
#
# **Intended use.** The model gives each booking a cancellation probability. The hotel could use it to plan overbooking, to contact risky guests, or to forecast how many rooms will really be occupied.
#
# **ML framing.** Supervised binary classification: the label `is_canceled` (1 = canceled, 0 = not canceled) is known for past bookings. Batch (offline) learning on one fixed file of bookings from two hotels (City Hotel and Resort Hotel), with arrivals from 2015 to 2017.
#
# **Performance measure.** The two errors have different costs: a missed cancellation can leave a room empty, while a false alarm can lead to overbooking. So accuracy alone is not enough (section 1.1.2). We report precision and recall separately, tune the models on F1 (2.8), and compare them with ROC AUC, which does not depend on the decision threshold (3.4).
#
# **Minimum performance:** the model must clearly beat the trivial model that always predicts "not canceled" (accuracy 0.64, recall 0) and random ranking (ROC AUC 0.5). The decision threshold is chosen in 3.5.

# %% [markdown]
# Import the data preparation code from the `src` folder.

# %%
import sys
from pathlib import Path
from importlib import import_module

import pandas as pd

# Import the local source when running from the project or notebook directory.
project_root = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "src" / "hotel_cancellation").is_dir()
)
sys.path.insert(0, str(project_root / "src"))

preprocessing = import_module("hotel_cancellation.0_preprocessing")

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
# Prepare `X` (information for the model) and `y` (whether the booking was canceled). Remove status columns that reveal the answer, clean text, and add total nights and guests.

# %%
X, y = preprocessing.prepare_features(df)
print(X.shape, y.shape)

# %%
# Inspect the feature matrix
print("Number of observations:", X.shape[0])
print("Number of features:", X.shape[1])

# Separate numerical and categorical columns
numerical_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(exclude="number").columns

print("\nNumerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

# %% [markdown]
# **Dimension of Feature Space** : $$ X \in \mathbb{R}^{119390 \times 31} $$

# %% [markdown]
# ### 1.0 Train / validation / test split
#
# The whole notebook uses **one** split, made here with `preprocessing.split_data`: about **60% train, 20% validation and 20% test**. Identical bookings are kept in the same part, so a booking the model was trained on can never appear again in validation or test (the raw data has about 32,000 duplicate rows). The training rows are shuffled once, because the file is sorted by hotel and booking status and cross-validation folds would otherwise not be random.
#
# | Part | Used for | Sections |
# |---|---|---|
# | Train | fitting models, cross-validation and tuning | 1.1, 2 |
# | Validation | comparing preprocessing options, choosing the model and the threshold | 1.1, 3.2 – 3.6 |
# | Test | final score, only after everything is fixed | 3.7 – 3.10 |

# %%
datasets = preprocessing.split_data(X, y)
X_train, y_train = datasets["train"]
X_validation, y_validation = datasets["validation"]
X_test, y_test = datasets["test"]

# split_data keeps the file order, which is sorted by hotel and booking status.
# Shuffle the training rows so that cross-validation folds in section 2 are random samples.
X_train = X_train.sample(frac=1, random_state=72)
y_train = y_train.loc[X_train.index]

display(pd.DataFrame([
    {"part": name, "bookings": len(part_X), "share": len(part_X) / len(X), "canceled_share": part_y.mean()}
    for name, (part_X, part_y) in datasets.items()
]).round(3))

# %% [markdown]
# ## 1.1 Preprocessing study: two useful changes
#
# This section is the preprocessing contribution. It reuses the existing training and
# metric functions to check log lead time for regression and rare-category grouping
# for the forest. The five training calls are explicit; plotting details live in `src`.
#
# **Comparison rules:** these experiments use the shared grouped train/validation/test
# split from section 1.0 (about 60/20/20), so identical prepared records stay together.
# Sections 2 and 3 use the same split. The test part is not a pristine external test;
# it reuses the same historical data. This section does not score the test part.
#
# The dataset reflects information near arrival, not necessarily at booking creation.
# These are retrospective learning experiments, not a verified booking-time system.

# %%
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload
preprocessing_plots = import_module("hotel_cancellation.00_preprocessing_plots")

modeling = import_module("hotel_cancellation.1_modeling")
evaluation = import_module("hotel_cancellation.2_evaluation")
preprocessing_report_folder = project_root / "reports/preprocessing"
preprocessing_report_folder.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### 1.1.1 Keep identical records in one split
#
# The split is made once in section 1.0 (about 60% train, 20% validation and 20% test);
# this section reuses its train and validation parts. Groups are based on features, never on outcomes. Excluding the five
# review fields from the group key also protects comparisons that omit those fields;
# they remain in the model inputs. This is not a complete audit of booking-time availability.
#
# The grouping and overlap checks are implemented in `preprocessing.split_data`.

# %%
# Reuse the shared split from section 1.0 (same rows; original file order, as produced by split_data)
prep_X_train, prep_y_train = datasets["train"]
prep_X_validation, prep_y_validation = datasets["validation"]

# %% [markdown]
# ### 1.1.2 Why these two changes?
#
# The cancellation rate changes with days before arrival. A log transformation can help
# regression represent this relationship. Agency/company/country contain many small
# categories; grouping these may simplify the forest's input. Use train only for exploration.

# %%
prep_train_bookings = df.loc[prep_X_train.index]
prep_category_columns = ["agent", "company", "country"]
prep_category_counts = prep_X_train[prep_category_columns].fillna("Unknown").nunique()
display(prep_category_counts.rename("categories_before_grouping"))

# %% [markdown]
# #### Class balance and hotel differences
#
# Cancellation is the smaller class, but it is not rare. Always predicting "not canceled"
# would already get about 63% accuracy, which is why accuracy alone is insufficient.
# The two hotels have different cancellation rates; sample sizes are shown above the bars.
# These plots use train only and help describe the task, without proving causation.

# %%
reload(preprocessing_plots)

prep_figure = preprocessing_plots.plot_class_balance(prep_train_bookings)
preprocessing_plots.save_figure(prep_figure, preprocessing_report_folder / "class_balance")
prep_figure.set_facecolor(preprocessing_plots.PREVIEW_BACKGROUND)
plt.show()

# %% [markdown]
# #### Booking timing and rare categories
#
# Labels above the timing bars show group sizes. Earlier bookings have higher cancellation
# rates in this train set. Rates also differ by arrival weekday, but this descriptive
# association did not establish an extra modeling benefit in the earlier experiments.
# These are associations, not causal effects. Unknown is separate from Other.

# %%
reload(preprocessing_plots)

prep_figure = preprocessing_plots.plot_feature_patterns(prep_train_bookings, prep_X_train)
preprocessing_plots.save_figure(prep_figure, preprocessing_report_folder / "data")
prep_figure.set_facecolor(preprocessing_plots.PREVIEW_BACKGROUND)
plt.show()

# %% [markdown]
# ### 1.1.3 Record results in one table
#
# This small helper **does not train models**. It measures a supplied model on train
# and validation and adds two rows to the table. The targets stay the same throughout.

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
# ### 1.1.4 Original logistic regression
#
# Train a reference model before changing lead time. Numerical gaps use the training median;
# categorical gaps remain a separate category.

# %%
prep_logistic_baseline = modeling.train_logistic_regression(
    prep_X_train, prep_y_train,
    preprocessor=preprocessing.build_preprocessor(),
)
record_preprocessing_results("LR baseline", prep_logistic_baseline, prep_X_train, prep_X_validation)

# %% [markdown]
# ### 1.1.5 Does log lead time help regression?
#
# Create separate input tables so the baseline inputs stay available. Log1p means
# ln(1 + value); it keeps zero and compresses large values. It does not learn any statistics,
# so we apply the same formula directly to train and validation.

# %%
prep_X_train_log = preprocessing.log_lead_time(prep_X_train)
prep_X_validation_log = preprocessing.log_lead_time(prep_X_validation)

prep_logistic_log = modeling.train_logistic_regression(
    prep_X_train_log, prep_y_train,
    preprocessor=preprocessing.build_preprocessor(),
)
record_preprocessing_results("LR + log lead", prep_logistic_log, prep_X_train_log, prep_X_validation_log)

# %% [markdown]
# ### 1.1.6 Compare the existing tree and forest
#
# Both calls use the original input tables and existing model settings. The tree is our
# reference for the best original F1. These calls really train the models.

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
# ### 1.1.7 Does grouping rare categories help the forest?
#
# Values of agent/company/country seen fewer than 50 times in train become Other.
# Unknown stays separate. The list is learned during fit on train and reused for validation.

# %%
prep_forest_rare = modeling.train_random_forest(
    prep_X_train, prep_y_train,
    preprocessor=preprocessing.build_preprocessor(rare_min_count=50),
)
record_preprocessing_results("Forest + rare", prep_forest_rare, prep_X_train, prep_X_validation)

# %% [markdown]
# ### 1.1.8 Read the results
#
# F1 measures cancellation classification. Accuracy is the share of all correct predictions.
# The comparison is against each model's own baseline and the original tree.

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
reload(preprocessing_plots)

prep_figure = preprocessing_plots.plot_experiment_results(prep_results_table)
preprocessing_plots.save_figure(prep_figure, preprocessing_report_folder / "experiments")
prep_figure.set_facecolor(preprocessing_plots.PREVIEW_BACKGROUND)
plt.show()

# %% [markdown]
# #### Why did PCA not help? A result from the earlier experiment
#
# PCA keeps directions with high numerical variance; it does not use cancellation labels.
# High retained variance therefore does not guarantee useful information for this target.
# In our earlier experiment, 95% PCA replaced 19 standardized numbers with 15 components,
# but left categories unchanged: total dimension only fell from 841 to 837.
#
# For regression with log lead time, F1 changed from **0.774378 to 0.774605**
# (+0.023 percentage points): no convincing gain. The original tree fell from 0.7742
# to 0.7228, and the rare-grouped forest from 0.7107 to 0.6702. PCA mixes and rotates
# input features, which can make single-feature threshold rules harder for trees;
# this experiment also discarded information, so it does not isolate the two effects.
#
# This explains why PCA was not selected. These are historical validation results,
# not a new training run in this notebook; the full setup and graphs are in
# [the full report](../reports/preprocessing/FULL_REPORT.md#stage-4).
# We have not established statistical significance or a general claim that PCA never helps.

# %% [markdown]
# ### 1.1.9 Conclusions
#
# 1. Log lead time improves logistic-regression F1 from 0.7548 to 0.7744 (+1.96 percentage points).
# 2. Grouping rare categories improves forest F1 from 0.5725 to 0.7107 (+13.83 points)
#    and reduces 841 columns to 231 (about 73% fewer).
# 3. The original tree already has F1 0.7742. Best overall F1 is practically unchanged.
#
# Earlier experiments found little benefit from missing-value flags and PCA. They are
# not rerun here; full results are in archive/preprocessing-research.zip. These are negative
# findings, not recommended transformations. Other unusual rows were not deleted.
#
# There is useful preprocessing work here, but no evidence of a major improvement over
# the best original model. Validation has been reused; stability and any tuning belong
# to the modeling/evaluation work. This experiment does not establish booking-time availability.

# %% [markdown]
# ## 2. Training
#

# %% [markdown]
# ### 2.1 Experimental Setup

# %% [markdown]
# ### 2.2 Logistic Regression
#
# We wish to establish a baseline classification model via Logistic Regression. All models in this section are trained on the **training part** of the shared split from section 1.0 (about 60% of the bookings). The validation and test parts are not used here; tuning relies on cross-validation inside the training part.

# %%
# The split is made once in section 1.0 (preprocessing.split_data); section 2 only uses the training part
print("Training observations:  ", X_train.shape[0])
print("Validation observations:", X_validation.shape[0])
print("Test observations:      ", X_test.shape[0])

# %% [markdown]
# ### Mathematical Model (Sigmoid)
#
# Estimating the probability of cancellation using the sigmoid function:
#  
# $$ 
# P(Y=1\mid X=x)=\frac{1}{1+e^{-(\beta_0+\boldsymbol{\beta}^{T}x)}}
# $$
#
# Or rather,
#
# $$
# P(Y=1\mid x)=\sigma\left(\beta_0+\sum_{j=1}^{p}\beta_j x_j\right)
# $$
#
# where each $ \beta_j $  is a learned coefficient (i.e., weight) associated with the corresponding input feature $x_j$, $\beta_{0}$ is the intercept (i.e., bias), and $p$ is the total number of features after preprocessing.

# %% [markdown]
# #### Model Training

# %%
# Import modeling module
modeling = import_module("hotel_cancellation.1_modeling")

# Train logistic regression
logistic_model = modeling.train_logistic_regression(
    X_train=X_train,
    y_train=y_train,
    preprocessor=preprocessing.build_preprocessor()
)

print("Logistic Regression Training Successfully Complete!")

# %%
import pandas as pd

# Retrieve the fitted preprocessing pipline
fitted_preprocessor = logistic_model[:-1]

# Retrieve the fitted logistic regression
classifier = logistic_model[-1]

# Names of all features after preprocessing
feature_names = fitted_preprocessor.get_feature_names_out()

# Learned Coefficients
coefficients = classifier.coef_[0]

# Organize into a readable table 
beta_table = pd.DataFrame({
    "Feature" : feature_names,
    "Coefficient (beta)" : coefficients
})

# Sort by magnitude 
beta_table = beta_table.reindex(
    beta_table["Coefficient (beta)"].abs()
    .sort_values(ascending=False).index
)

print("Intercept beta_0: ",classifier.intercept_[0])
print("Number of beta coefficients: ",len(coefficients))

display(beta_table)

# %% [markdown]
# #### Interpretation of learned Coefficients
#
# Our logistic regression model successfully learned an intercept of $\beta_{0} \approx$  -1.485 and a vector of 841 Coefficients. (i.e., $(\beta_1,\dots \beta_{841})$).
#
# Though our original dataset contained just 31 input features, categorical variables (e.g., agent, company, country) are expanded into a myriad of binary features via 1-hot encoding. As a result, the logistic model operates on 841 transformed features, each being associated with a learned coefficient $\beta_j$.
#
# The sign and magnitude of each coefficient $\beta_n$ describes how its correspoding feature influences the predicted log-odds of cancellation holding the other features constant; however, coefficient odds do not necessarily measure a feature's overall predictive performance. 

# %% [markdown]
# ### 2.3 Decision Tree

# %% [markdown]
# #### Mathematical Model (Gini Impurity)
#
# In contrast to logistic regression, which learns a weighted linear combination of features so as to carve the feature space into distinct regions via a line, plane, or hyperplane depending on the degree of dimensionality, a decision tree Classifies our observations by recursively partitioning the feature space according to a sequence of decision rules. In simpler terms, the decision tree model is a series of yes or no questions (e.g., "Is $x_{n}$ > 0.5?", where $x_n$ is the $n^{th}$ axis in $m$ dimensions) that partitions the feature space into rectangular regions in two dimensions, rectangular prisms in three, and hyperrectangular regions in higher-dimensional spaces using axis alinged cuts. 
#
# At each node, the algorithm selects a feature and a threshold that best separate the observations into increasingly homogenous subsets.
#
# Formally we can define node impurity (i.e., Gini impurity) as:
#
# $$
# G(S) = 1 - \sum_{k=1}^{K}p_k^2
# $$
#
# where $p_k$ represents the portion of observations beloning to class $k$ within node $S$.
#
# Considering our problem is a binary classification (i.e., canceled and not canceled), the equation conveiently simplifies to:
#
# $$
# G(S) = 1 - p^2 - (1-p)^2 = 2p(1-p)
# $$
#
# where $p$ is the propportion of cancled bookings at the node. 
#
# | Gini Impurity | Class Distribution | Interpretation |
# |---|---|---|
# | $G=0$ | 0% / 100% | Pure node; all bookings belong to one class |
# | $G=0.18$ | 10% / 90% | Low impurity; one class dominates |
# | $G=0.32$ | 20% / 80% | Moderate impurity |
# | $G=0.50$ | 50% / 50% | Maximum impurity; classes are equally represented |
#
# An importatant takeaway from the decision tree model is that it permits nonlinear interactions and discontinuities whereas logistic regression does not.  
#

# %% [markdown]
# #### Model Training

# %%
tree_model = modeling.train_decision_tree(
    X_train=X_train,
    y_train=y_train,
    preprocessor=preprocessing.build_preprocessor()
)

tree_classifier = tree_model[-1]

print("Tree depth: ", tree_classifier.get_depth())
print("Number of leaves: ", tree_classifier.get_n_leaves())

# %% [markdown]
# ### Geometric Interpreation (Logistic Regression vs Decision Tree)

# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 2D toy binary classification dataset to illustrate the geometric differences between the two methods

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

# Create a grid covering the feature space
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
        cmap="coolwarm"
    )

    ax.contour(
        xx, yy, predictions,
        levels=[0.5],
        colors="black",
        linewidths=1.5
    )

    ax.scatter(
        X_toy[:, 0], X_toy[:, 1],
        c=y_toy,
        cmap="coolwarm",
        edgecolors="black",
        s=25
    )

    ax.set_title(name)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

plt.suptitle("Comparison of Learned Decision Boundaries")
plt.tight_layout()
plt.show()


# %%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Extract the names of the transformed features 
feature_names = tree_model[:-1].get_feature_names_out()

plt.figure(figsize=(15,6))

plot_tree(
    tree_classifier,
    feature_names=feature_names,
    class_names=["Not canceled", "Canceled"],
    filled=True,
    rounded=True,
    max_depth=2,
    fontsize=9
)

plt.title("Decision Tree - First Three Levels")
plt.tight_layout()
plt.show()

# %%
import pandas as pd

tree_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": tree_classifier.feature_importances_
}).sort_values("Importance", ascending=False)

display(tree_importance.head(10))

# %% [markdown]
# ### Bias-Variance Tradeoff
#
# Decision trees are a perfect illustrative example of how overfitting can hinder a models ability to generalize well on unseen data. A decision tree of depth 30 (i.e., $2^{30}$), for example, yields a tree with leaves which number in the billions and depending on the number of observations captures mostly sampling noise.

# %% [markdown]
# ### 2.4 Random Forest

# %% [markdown]
# #### Mathematical Model (Ensemble Learning)
#
# Whereas a single, potentially deep, decision tree can classify observations via a hierarchy of splitting rules, a random forest combines multiple deecision trees using randomized versions of the dataset to improve the predictive robustness and reduce variance (i.e., how sensative our learned model is to changes in its training dataset). 
#
# Formally, a forest consisting of $T$ decision trees can be represented as:
#
# $$
# \hat{y}=\operatorname{mode}({h_1(x),h_2(x),\ldots,h_T(x)})
# $$
#
# where, $h_t(x)$ denote the class predicted by the $t$-th decision tree for observation $x$. 
#
# The final prediction is then determined by the most frequntly predicted class across the ensemble. 
#
# The principle idea is that one may average the imperfect behavior of many differnetly trained trees to reduce the variance associated with individual trees, thus making them less susceptible to overfitting.

# %% [markdown]
# #### Model Training

# %%
# Train Random Forest
forest_model = modeling.train_random_forest(
    X_train=X_train,
    y_train=y_train,
    preprocessor=preprocessing.build_preprocessor()
)

forest_classifier = forest_model[-1]

print("Number of Trees:", len(forest_classifier.estimators_))

print(
    "Average tree depth:",
    round(
        sum(tree.get_depth() for tree in forest_classifier.estimators_)
        / len(forest_classifier.estimators_),
        2
    )
)

# %% [markdown]
# ### 2.5 Cross Validation
#
# Though the training performance of a model provides us with insight into its ability to fit the available obervations, it doesn't necessarily indicate how well the model will generalize on unseen data. To obtain a more robust estimate of generalization performance and compare our three baseline classifiers, we employ 5-fold cross-validation.
#
# Currently, we have 71,764 training observations, representing about 60% of our original dataset (section 1.0). 
#
# Formally: 
#
# $$
# D_{train} = \{{(x_{i}, y_{i})}\}_{i=1}^{71,764}
# $$
#
# Rather than relying on a single evaluation of a validation subset, we can partition the training data into $K$ = 5 disjoint folds:
#
# $$
# D_{\mathrm{train}}=D_1\cup D_2\cup\cdots\cup D_5
# $$
#
# $$
# D_i\cap D_j=\varnothing,\quad i\ne j.
# $$
#
# i.e., no two folds contain the same observation.
#
#
#
#
#
#
#
# A larger standard deviation does not necessarily imply a that the model has a higher variance in the bias-variance sense. 

# %%

from sklearn.model_selection import cross_validate

# Our three previously trained pipelines
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model
}

# Performance metrics to evaluate
scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC-AUC": "roc_auc"
}

# Store the validation results for each model
cv_results = {}

for name, model in models.items():

    print(f"Cross-validating: {name}")

    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        scoring=scoring,
        cv=5,
        n_jobs=1
    )

    cv_results[name] = scores

    print("Completed!\n")


# %%

summary = []

for name, scores in cv_results.items():

    row = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]

        row[f"{metric} Mean"] = np.mean(values)
        row[f"{metric} Std"] = np.std(values, ddof=1)

    summary.append(row)

cv_summary = pd.DataFrame(summary)

display(cv_summary.round(4))


# %% [markdown]
# ### 2.6 Gradient Boosting 
#
#

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Construct the Gradient Boosting pipeline
boosting_model = Pipeline([
    ("preprocessor", preprocessing.build_preprocessor()),
    ("classifier", GradientBoostingClassifier(
        n_estimators=60,
        learning_rate=0.1,
        max_depth=2,
        max_features="sqrt",
        random_state=72
    ))
])

# Train on the original training dataset
boosting_model.fit(X_train, y_train)

print("Gradient Boosting Training Successfully Complete!")


boosting_classifier = boosting_model[-1]

print("Number of boosting stages:", boosting_classifier.n_estimators_)
print("Learning rate:", boosting_classifier.learning_rate)
print("Maximum tree depth:", boosting_classifier.max_depth)
print("Training accuracy:", round(
    boosting_model.score(X_train, y_train), 4
))


# Update models dictionary

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model,
    "Gradient Boosting": boosting_model
}

# %% [markdown]
# ### 2.8 Hyperparameter Tuning 

# %% [markdown]
# #### Logistic Regression Tuning

# %%
from sklearn.model_selection import GridSearchCV

'''
Hyperparameter search space

C : controls how strongly we constrain the model's coeff's.

class_weight : controls how much importance we assign to predicting cancellations vs. 
non-cancellations correctly.
'''

logistic_param_grid = {
    "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    "logisticregression__class_weight": [None, "balanced"]
}

logistic_search = GridSearchCV(
    estimator=logistic_model,
    param_grid=logistic_param_grid,
    scoring="f1",
    cv=5,
    n_jobs=1,
    return_train_score=True,
    verbose=1
)

logistic_search.fit(X_train, y_train)

print("Best hyperparameters:")
print(logistic_search.best_params_)

print("\nBest cross-validation F1-score:")
print(round(logistic_search.best_score_, 4))

best_logistic_model = logistic_search.best_estimator_


# %%
# Convert grid search results into a DataFrame
logistic_results = pd.DataFrame(logistic_search.cv_results_)

# Select and rename columns for readability
columns = {
    "param_logisticregression__C": "C (Regularization)",
    "param_logisticregression__class_weight": "Class Weight",
    "mean_test_score": "Mean CV F1",
    "std_test_score": "Std CV F1"
}

logistic_summary = (
    logistic_results[list(columns.keys())]
    .rename(columns=columns)
    .sort_values("Mean CV F1", ascending=False)
    .reset_index(drop=True)
)

display(logistic_summary.round(4))

# Training versus validation performance
best_idx = logistic_search.best_index_

train_f1 = logistic_search.cv_results_["mean_train_score"][best_idx]
validation_f1 = logistic_search.cv_results_["mean_test_score"][best_idx]

print(f"Mean training F1:   {train_f1:.4f}")
print(f"Mean validation F1: {validation_f1:.4f}")
print(f"Generalization gap: {train_f1 - validation_f1:.4f}")



# %% [markdown]
# #### Decision Tree Tuning

# %%
'''
Hyperparameter search space
max_depth : controls max depth of the tree
 
min_samples_split : specifies the minimum number of observations a node must contain 
before it can be split into child nodes

min_samples_leaf : Specifies the minimum number of observations that must remain in each leaf node after 
a split , which prevents excessively small partitions that may overfit the training data
'''

param_grid = {
    "decisiontreeclassifier__max_depth": [3, 5, 8, 12, 16],
    "decisiontreeclassifier__min_samples_split": [2, 10, 20],
    "decisiontreeclassifier__min_samples_leaf": [1, 5, 10]
}

# Initialize grid search
tree_search = GridSearchCV(
    estimator=tree_model,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=1,
    return_train_score=True
)

# Perform hyperparameter tuning
tree_search.fit(X_train, y_train)

print("Best hyperparameters:")
print(tree_search.best_params_)

print("\nBest cross-validation F1-score:")
print(round(tree_search.best_score_, 4))

# Retrieve the best fitted pipeline
best_tree_model = tree_search.best_estimator_


# %%
# Convert grid search results into a DataFrame
tree_results = pd.DataFrame(tree_search.cv_results_)

# Select and rename columns for readability
columns = {
    "param_decisiontreeclassifier__max_depth": "Max Depth",
    "param_decisiontreeclassifier__min_samples_split": "Min Samples Split",
    "param_decisiontreeclassifier__min_samples_leaf": "Min Samples Leaf",
    "mean_test_score": "Mean CV F1",
    "std_test_score": "Std CV F1"
}

tree_summary = (
    tree_results[list(columns.keys())]
    .rename(columns=columns)
    .sort_values("Mean CV F1", ascending=False)
    .reset_index(drop=True)
)

display(tree_summary.head(10).round(4))


# Training versus validation performance
best_idx = tree_search.best_index_

train_f1 = tree_search.cv_results_["mean_train_score"][best_idx]
validation_f1 = tree_search.cv_results_["mean_test_score"][best_idx]

print(f"Mean training F1:   {train_f1:.4f}")
print(f"Mean validation F1: {validation_f1:.4f}")
print(f"Generalization gap: {train_f1 - validation_f1:.4f}")



# %% [markdown]
# #### Random Forest Tuning

# %%

from sklearn.model_selection import RandomizedSearchCV

# Identify the classifier's name inside the pipeline
forest_step = forest_model.steps[-1][0]

# Hyperparameter search space
forest_param_grid = {
    f"{forest_step}__n_estimators": [50, 100, 150],
    f"{forest_step}__max_depth": [10, 16, 24, None],
    f"{forest_step}__min_samples_leaf": [1, 2, 5],
    f"{forest_step}__max_features": ["sqrt", 0.5]
}

forest_search = RandomizedSearchCV(
    estimator=forest_model,
    param_distributions=forest_param_grid,
    n_iter=8,
    scoring="f1",
    cv=3,
    n_jobs=1,
    random_state=72,
    return_train_score=True,
    verbose=1
)

forest_search.fit(X_train, y_train)

print("Best hyperparameters:")
print(forest_search.best_params_)

print("\nBest cross-validation F1-score:")
print(round(forest_search.best_score_, 4))

best_forest_model = forest_search.best_estimator_


# %%
#Analyze Results

forest_results = pd.DataFrame(forest_search.cv_results_)

# Select and rename columns for readability
columns = {
    "param_randomforestclassifier__n_estimators": "N Trees",
    "param_randomforestclassifier__max_depth": "Max Depth",
    "param_randomforestclassifier__min_samples_leaf": "Min Samples Leaf",
    "param_randomforestclassifier__max_features": "Max Features",
    "mean_test_score": "Mean CV F1",
    "std_test_score": "Std CV F1"
}

forest_summary = (
    forest_results[list(columns.keys())]
    .rename(columns=columns)
    .sort_values("Mean CV F1", ascending=False)
    .reset_index(drop=True)
)

display(forest_summary.round(4))

# Training versus validation performance
best_idx = forest_search.best_index_

train_f1 = forest_search.cv_results_["mean_train_score"][best_idx]
validation_f1 = forest_search.cv_results_["mean_test_score"][best_idx]

print(f"Mean training F1:   {train_f1:.4f}")
print(f"Mean validation F1: {validation_f1:.4f}")
print(f"Generalization gap: {train_f1 - validation_f1:.4f}")


# %% [markdown]
# ### Gradient Boost Tuning

# %%
boosting_param_grid = {
    "classifier__n_estimators": [60, 100, 150],
    "classifier__learning_rate": [0.05, 0.1, 0.2],
    "classifier__max_depth": [2, 3, 4],
    "classifier__subsample": [0.8, 1.0]
}

boosting_search = RandomizedSearchCV(
    estimator=boosting_model,
    param_distributions=boosting_param_grid,
    n_iter=6,
    scoring="f1",
    cv=3,
    n_jobs=1,
    random_state=72,
    return_train_score=True,
    verbose=1
)

boosting_search.fit(X_train, y_train)

print("Best hyperparameters:")
print(boosting_search.best_params_)

print("\nBest cross-validation F1-score:")
print(round(boosting_search.best_score_, 4))

best_boosting_model = boosting_search.best_estimator_


# %%

# Convert randomized search results into a DataFrame
boosting_results = pd.DataFrame(boosting_search.cv_results_)

# Select and rename columns for readability
columns = {
    "param_classifier__n_estimators": "N Estimators",
    "param_classifier__learning_rate": "Learning Rate",
    "param_classifier__max_depth": "Max Depth",
    "param_classifier__subsample": "Subsample",
    "mean_test_score": "Mean CV F1",
    "std_test_score": "Std CV F1"
}

boosting_summary = (
    boosting_results[list(columns.keys())]
    .rename(columns=columns)
    .sort_values("Mean CV F1", ascending=False)
    .reset_index(drop=True)
)

# Display the results without the index
display(
    boosting_summary.style
    .hide(axis="index")
    .format({
        "Mean CV F1": "{:.4f}",
        "Std CV F1": "{:.4f}",
        "Learning Rate": "{:.2f}",
        "Subsample": "{:.1f}"
    })
)

# Training versus validation performance
best_idx = boosting_search.best_index_

train_f1 = boosting_search.cv_results_["mean_train_score"][best_idx]
validation_f1 = boosting_search.cv_results_["mean_test_score"][best_idx]

print(f"Mean training F1:   {train_f1:.4f}")
print(f"Mean validation F1: {validation_f1:.4f}")
print(f"Generalization gap: {train_f1 - validation_f1:.4f}")



# %% [markdown]
# ### 2.9 Summary of Optimal Hyperparameters

# %%
# Collect completed hyperparameter searches
searches = {
    "Logistic Regression": logistic_search,
    "Decision Tree": tree_search,
    "Random Forest": forest_search,
    "Gradient Boosting": boosting_search
}

performance_summary = []
hyperparameter_summary = []

for name, search in searches.items():
    idx = search.best_index_

    train_f1 = search.cv_results_["mean_train_score"][idx]
    validation_f1 = search.cv_results_["mean_test_score"][idx]

    # Table 1: Model performance
    performance_summary.append({
        "Model": name,
        "CV Folds": search.n_splits_,
        "Training F1": train_f1,
        "Validation F1": validation_f1,
        "Generalization Gap": train_f1 - validation_f1
    })

    # Table 2: Optimal hyperparameters
    for parameter, value in search.best_params_.items():
        hyperparameter_summary.append({
            "Model": name,
            "Hyperparameter": parameter.split("__")[-1],
            "Optimal Value": str(value)
        })

# Create DataFrames
performance_df = pd.DataFrame(performance_summary)
hyperparameter_df = pd.DataFrame(hyperparameter_summary)

# Sort models by validation performance
performance_df = performance_df.sort_values(
    "Validation F1", ascending=False
).reset_index(drop=True)

print("MODEL PERFORMANCE SUMMARY")

display(
    performance_df.style
    .hide(axis="index")
    .format({
        "Training F1": "{:.4f}",
        "Validation F1": "{:.4f}",
        "Generalization Gap": "{:.4f}"
    })
)

print("\nOPTIMAL HYPERPARAMETERS")

display(
    hyperparameter_df.style.hide(axis="index")
)



# %% [markdown]
# ### Final optimized parameters for evaluation

# %%

# Final optimized models for evaluation
optimized_models = {
    "Logistic Regression": best_logistic_model,
    "Decision Tree": best_tree_model,
    "Random Forest": best_forest_model,
    "Gradient Boosting": best_boosting_model
}

# Verify that each pipeline can make predictions
for name, model in optimized_models.items():
    predictions = model.predict(X_train.iloc[:5])
    probabilities = model.predict_proba(X_train.iloc[:5])

    print(f"{name}: ready")


# %% [markdown]
# ## 3. Evaluation
#
# We choose the final model and the decision threshold on the validation set (3.2 - 3.6). The test set is used only once, at the end (3.7 - 3.10). The evaluation functions are in `src/hotel_cancellation/2_evaluation.py`.

# %% [markdown]
# ### 3.1 Metrics
#
# Canceled (`1`) is the positive class. A booking is predicted as canceled if its probability is at least the threshold (0.5 by default).
#
# | Metric | Formula | Depends on the threshold? |
# |---|---|---|
# | Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | yes |
# | Precision | $\frac{TP}{TP + FP}$, share of flagged bookings that really cancel | yes |
# | Recall | $\frac{TP}{TP + FN}$, share of cancellations that are found | yes |
# | F1 | $2 \cdot \frac{P \cdot R}{P + R}$ | yes |
# | ROC AUC | probability that a canceled booking gets a higher score than a kept one | no |
# | Brier score | $\frac{1}{n}\sum_i (\hat p_i - y_i)^2$, lower is better | no |
# | Log loss | $-\frac{1}{n}\sum_i \big[y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\big]$ | no |
# | Error rate | $1 - \text{Accuracy}$ | yes |

# %% [markdown]
# ### 3.2 Baseline: logistic regression
#
# We start with the untuned logistic regression from 2.2 and compare it with a trivial model that always predicts "not canceled".

# %%
import pandas as pd
from sklearn.dummy import DummyClassifier

evaluation = import_module("hotel_cancellation.2_evaluation")

# White plot backgrounds, independent of the editor theme
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

# Always predicts "not canceled", with probability of canceling = 0.37 (the share in the data)
dummy_model = DummyClassifier(strategy="prior").fit(X_train, y_train)

results = pd.DataFrame({
    "train": evaluation.get_metrics(logistic_model, X_train, y_train),
    "validation": evaluation.get_metrics(logistic_model, X_validation, y_validation),
    "always 'not canceled' (validation)": evaluation.get_metrics(dummy_model, X_validation, y_validation),
}).T
results.loc["gap (validation - train)"] = results.loc["validation"] - results.loc["train"]

results.round(3)

# %% [markdown]
# The model is right for 83% of the validation bookings, the trivial model for 64%. Recall (0.71) is lower than precision (0.81), so about 29% of the cancellations are missed.

# %%
from sklearn.metrics import classification_report

evaluation.plot_confusion_matrix(logistic_model, X_validation, y_validation, title="Logistic regression, validation set")

y_pred = logistic_model.predict(X_validation)
print(classification_report(y_validation, y_pred, target_names=["Not canceled", "Canceled"], digits=3))

# %%
evaluation.plot_roc_curve(logistic_model, X_validation, y_validation, title="ROC curve: logistic regression, validation set")

# %% [markdown]
# **Diagnosis.** The training error (0.163) and the validation error (0.166) are almost the same, so the model does not overfit. But it is wrong for 16% of the training bookings, so it is too simple for this data (high bias, underfitting).
#
# What can help against underfitting:
# - a more flexible model, such as a decision tree, a random forest or gradient boosting (compared in 3.3 - 3.4);
# - better features: 1.1 tests a log transform of the lead time and grouping of rare categories;
# - a lower decision threshold or class weights, to find more cancellations. The ROC AUC is already 0.911, so the model ranks bookings well and the threshold can trade precision for recall (3.5).

# %% [markdown]
# ### 3.3 Untuned models: do trees help?
#
# The decision tree (2.3) and the random forest (2.4) with default settings, compared with the baseline on the validation set.

# %%
models = {
    "Logistic regression": logistic_model,
    "Decision tree": tree_model,
    "Random forest": forest_model,
}

comparison = evaluation.compare_models(models, X_train, y_train, X_validation, y_validation)
comparison.round(3)

# %%
evaluation.plot_roc_curves(models, X_validation, y_validation, title="ROC curves: all models, validation set")

# %%
for name in ["Decision tree", "Random forest"]:
    evaluation.plot_confusion_matrix(models[name], X_validation, y_validation, title=f"{name}, validation set")

# %% [markdown]
# The decision tree is a bit better than logistic regression (F1 0.774 vs 0.755, AUC 0.919 vs 0.911), mainly because of its higher recall. The random forest has a similar AUC (0.910) but a very low recall at 0.5 (0.394), because its probabilities rarely go above 0.5. The untuned models are close, so we choose after tuning (2.8) and with a better threshold.

# %% [markdown]
# ### 3.4 Tuned models on the validation set
#
# The tuned models from 2.8 were trained on the training set only.

# %%
# Probability of "canceled" for every validation booking, one column per tuned model
validation_proba = pd.DataFrame(
    {name: model.predict_proba(X_validation)[:, 1] for name, model in optimized_models.items()},
    index=X_validation.index,
)

# Trivial reference from 3.2: always predicts the training cancellation rate, so it never flags a booking
validation_proba["Always 'not canceled'"] = dummy_model.predict_proba(X_validation)[:, 1]

validation_comparison = evaluation.compare_proba(validation_proba, y_validation).sort_values("roc_auc", ascending=False)
validation_comparison.round(3)

# %%
model_names = list(optimized_models)
evaluation.plot_roc_pr_curves(validation_proba[model_names], y_validation, title="(validation set)")

# %%
# The final model is the one that ranks bookings best (threshold-free), before any threshold is tuned
final_name = validation_comparison.loc[model_names, "roc_auc"].idxmax()
final_proba_validation = validation_proba[final_name]
print("Final model:", final_name)

# %% [markdown]
# The random forest is the best model: ROC AUC 0.946 (logistic regression 0.912, gradient boosting 0.911, decision tree 0.906), the lowest Brier score (0.089) and the best F1 (0.815). Gradient boosting has the highest precision (0.897) but a low recall (0.590). We choose the **random forest** as the final model.

# %% [markdown]
# ### 3.5 Choosing the decision threshold
#
# The threshold does not change the ROC AUC, only the balance between precision and recall. For the hotel, a missed cancellation is worse than a false alarm: a missed cancellation can leave a room empty and lose its revenue, while a false alarm only means an extra reminder to a guest who comes anyway. So we prefer a higher recall, as long as precision does not drop too much. We choose the threshold on the validation set.

# %%
threshold_metrics = evaluation.threshold_table(y_validation, final_proba_validation)
f1_threshold = threshold_metrics["f1"].idxmax()

evaluation.plot_threshold_curves(
    threshold_metrics,
    marked={"default": 0.5, "best F1": f1_threshold},
    title=f"{final_name}: metrics against threshold (validation set)",
)
threshold_metrics.loc[sorted({0.3, 0.4, 0.5, 0.6, 0.7, f1_threshold})].round(3)

# %%
# Final threshold: the F1-optimal one from 3.5 (reasoning in the results below)
final_threshold = f1_threshold
print(f"Final threshold: {final_threshold:.2f}")

# %% [markdown]
# F1 is highest at 0.40 (0.824, against 0.815 at 0.5). Going from 0.5 to 0.40 raises recall from 0.784 to 0.846 and lowers precision from 0.848 to 0.803.
#
# We use **0.40**. It finds more cancellations, which matters most for the hotel, and 80% of the flagged bookings still really cancel. A lower threshold such as 0.30 would find even more (recall 0.907), but precision would drop to 0.747, so one in four flagged guests would be contacted for nothing.

# %% [markdown]
# ### 3.6 Calibration
#
# If the model predicts 70%, do about 70% of these bookings cancel? This matters if the hotel uses the probabilities directly, for example to estimate how many rooms will be free.

# %%
from sklearn.calibration import calibration_curve

evaluation.plot_calibration(validation_proba[model_names], y_validation, title="Calibration (validation set)")

# Largest distance from the diagonal over the 10 bins
calibration_gap = {}
for name in model_names:
    observed, predicted = calibration_curve(y_validation, validation_proba[name], n_bins=10)
    calibration_gap[name] = np.abs(observed - predicted).max()
pd.Series(calibration_gap, name="largest gap: predicted vs observed").round(3)

# %% [markdown]
# The random forest is the best calibrated: its curve is at most 0.040 away from the diagonal. Logistic regression (0.125), the decision tree (0.154) and gradient boosting (0.218) are further away. For logistic regression this is partly expected, because `class_weight="balanced"` shifts its probabilities up.

# %% [markdown]
# ### 3.7 Final test score
#
# The random forest with threshold 0.40 is now scored on the test set, which was not used before. The 95% confidence intervals come from bootstrapping: we resample the test set 1,000 times and recompute the metrics.

# %%
final_model = optimized_models[final_name]
test_proba = final_model.predict_proba(X_test)[:, 1]

final_results = evaluation.bootstrap_metrics(y_test, test_proba, threshold=final_threshold)
final_results["at default 0.5"] = pd.Series(evaluation.metrics_from_proba(y_test, test_proba, 0.5))
final_results["always 'not canceled'"] = pd.Series(
    evaluation.metrics_from_proba(y_test, dummy_model.predict_proba(X_test)[:, 1])
)
final_results.round(3)

# %%
from sklearn.metrics import ConfusionMatrixDisplay

test_pred = (test_proba >= final_threshold).astype(int)
ConfusionMatrixDisplay.from_predictions(
    y_test, test_pred,
    display_labels=["Not canceled", "Canceled"],
    cmap="Purples",
    colorbar=False,
)
plt.title(f"{final_name}, test set, threshold {final_threshold:.2f}")
plt.show()

# %% [markdown]
# Test set, threshold 0.40:
#
# | Metric | Estimate | 95% CI |
# |---|---|---|
# | ROC AUC | 0.941 | 0.939 - 0.944 |
# | F1 | 0.818 | 0.812 - 0.824 |
# | Recall | 0.841 | 0.834 - 0.849 |
# | Precision | 0.795 | 0.787 - 0.804 |
# | Accuracy | 0.863 | 0.859 - 0.868 |
#
# The test scores are close to the validation scores (AUC 0.946, F1 0.824), so the choices made on the validation set also hold on new data. Compared with the default threshold 0.5, recall is higher (0.841 vs 0.785) and precision is lower (0.795 vs 0.841).

# %% [markdown]
# ### 3.8 Is the random forest really better?
#
# We compare its AUC with the second-best model from 3.4 on the same 1,000 bootstrap samples of the test set.

# %%
second_name = validation_comparison.loc[model_names, "roc_auc"].sort_values(ascending=False).index[1]
second_proba = optimized_models[second_name].predict_proba(X_test)[:, 1]

auc_differences = evaluation.bootstrap_auc_difference(y_test, test_proba, second_proba)
evaluation.plot_bootstrap_difference(auc_differences, label=f"ROC AUC: {final_name} - {second_name} (test set)")

# %% [markdown]
# The random forest's AUC is 0.031 higher than logistic regression's, with a 95% interval of [0.028, 0.034]. The interval does not include 0, so the difference is not due to chance.

# %% [markdown]
# ### 3.9 Error analysis
#
# Precision and recall for different groups of test bookings (groups with fewer than 200 bookings are left out).

# %%
X_test_segments = X_test.assign(
    lead_time_group=pd.cut(
        X_test["lead_time"],
        bins=[-1, 7, 30, 90, 180, 365, X_test["lead_time"].max()],
        labels=["0-7", "8-30", "31-90", "91-180", "181-365", "365+"],
    )
)
segment_tables = {
    column: evaluation.metrics_by_segment(X_test_segments, y_test, test_proba, column, threshold=final_threshold)
    for column in ["hotel", "market_segment", "deposit_type", "lead_time_group"]
}
evaluation.plot_segment_recall(segment_tables, overall_recall=final_results.loc["recall", "estimate"])
pd.concat(segment_tables).round(3)

# %% [markdown]
# - Short-notice bookings are the main weakness. For bookings made 0-7 days before arrival, recall is only 0.177: few of them cancel (10%), so the model rarely flags them. For long lead times recall is high (181-365 days: 0.951).
# - Direct and corporate bookings are also hard (recall 0.351 and 0.504), while groups (0.970) and offline travel agents (0.894) are easy.
# - Almost all non-refundable bookings are canceled (98.8%) and the model finds them (recall 0.999). The errors are in no-deposit bookings (recall 0.769).
# - The two hotels are similar (recall 0.843 and 0.835).
#
# To find the short-notice and direct cancellations, we would need data that is not in the dataset, for example whether the guest has already paid.

# %% [markdown]
# ### 3.10 Which columns matter?
#
# Permutation importance: we shuffle one column at a time in 5,000 test bookings and measure how much the ROC AUC drops. Unlike the coefficients (2.2) or the Gini importances (2.3), it works for any model.

# %%
permutation_table = evaluation.plot_permutation_importance(final_model, X_test, y_test)
permutation_table.round(4)

# %% [markdown]
# The most important columns are `country` (AUC drop 0.116), `agent` (0.084), `deposit_type` (0.062), `total_of_special_requests` (0.038) and `lead_time` (0.038). `arrival_date_year` is 7th (0.023), so part of what the model learned is specific to 2015-2017. `booking_changes` and `assigned_room_type`, which may be filled in close to arrival, matter little (0.005 each).

# %% [markdown]
# ### 3.11 Save the final model
#
# The pipeline and the threshold are saved with `joblib`. The `models/` folder is in `.gitignore` because the file is large.

# %%
import joblib

model_dir = project_root / "models"
model_dir.mkdir(exist_ok=True)
joblib.dump(
    {"name": final_name, "model": final_model, "threshold": final_threshold},
    model_dir / "final_model.joblib",
    compress=3,
)

# %% [markdown]
# ### 3.12 Conclusion
#
# We recommend the tuned random forest with threshold 0.40. On the test set it finds 84% of the cancellations (recall 0.841), and 80% of the bookings it flags are really canceled (precision 0.795). Its ROC AUC is 0.941, clearly better than logistic regression (+0.031) and the trivial model. Its probabilities are also the best calibrated, so the hotel can use them to estimate the number of cancellations per night.
#
# Limitations: the data describes bookings close to arrival, not at booking time (1.1). The model uses the arrival year, so it should be checked on newer data. It is weak on short-notice and direct bookings (3.9).
