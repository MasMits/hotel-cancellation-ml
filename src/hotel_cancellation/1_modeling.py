from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Logistic Model
def train_logistic_regression(
    X_train,
    y_train,
    preprocessor,
    max_iter=1000,
    random_state=72
):


    model = make_pipeline(
        preprocessor,
        LogisticRegression(
            max_iter=max_iter,
            random_state=random_state
        )
    )

    model.fit(X_train, y_train)

    return model

# Decision Tree Model
def train_decision_tree(
        X_train,
        y_train,
        preprocessor,
        max_depth=8,            # Both max_depth and min_leaf_samples are preliminary constraints 
        min_samples_leaf=20,    # Not optimzied hyperparameters
        random_state=72
):
    model = make_pipeline(
        preprocessor,
        DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
    )


    model.fit(X_train, y_train)

    return model

# Random Forest Model
def train_random_forest(
        X_train,
        y_train,
        preprocessor,
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=10,
        random_state=72
):
    model = make_pipeline(
        preprocessor,
        RandomForestClassifier(
            n_estimators=n_estimators,
            criterion="gini",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            random_state=random_state,
            n_jobs=2
        )
    )

    model.fit(X_train, y_train)

    return model
