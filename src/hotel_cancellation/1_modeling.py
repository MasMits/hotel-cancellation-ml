from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


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
