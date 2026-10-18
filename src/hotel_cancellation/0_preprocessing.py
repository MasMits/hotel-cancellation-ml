from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

DATA_PATH = Path(__file__).resolve().parents[2] / "data/raw/hotel_bookings.csv"

# Group on the same inputs even when these fields are omitted in an experiment.
# This affects only the split key, not the features passed to the model.
GROUPING_EXCLUDED_COLUMNS = [
    "assigned_room_type", "booking_changes", "days_in_waiting_list",
    "adr", "deposit_type",
]


def load_data(path=None):
    if path is None:
        path = DATA_PATH
    return pd.read_csv(path)


def prepare_features(df):
    df = df.dropna(subset=["is_canceled"])
    target = df["is_canceled"]
    if not target.isin([0, 1]).all():
        raise ValueError("is_canceled must contain only 0 or 1.")

    features = df.drop(
        columns=["is_canceled", "reservation_status", "reservation_status_date"]
    )

    text_columns = features.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        features[column] = features[column].str.strip()

    features[["agent", "company"]] = features[["agent", "company"]].astype("string")
    features[["agent", "company"]] = features[["agent", "company"]].fillna("Unknown")

    features["total_nights"] = features["stays_in_weekend_nights"] + features["stays_in_week_nights"]
    features["total_guests"] = features["adults"] + features["children"] + features["babies"]
    return features, target


def split_data(features, target, random_state=72):
    """Return train/validation/test pairs, keeping identical feature groups together."""
    group_features = features.drop(columns=GROUPING_EXCLUDED_COLUMNS)
    groups = pd.util.hash_pandas_object(group_features, index=False)

    test_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=random_state,
    )
    development_rows, test_rows = next(
        test_splitter.split(features, groups=groups)
    )

    validation_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=random_state,
    )
    train_positions, validation_positions = next(
        validation_splitter.split(
            features.iloc[development_rows], groups=groups.iloc[development_rows],
        )
    )
    train_rows = development_rows[train_positions]
    validation_rows = development_rows[validation_positions]

    train_groups = set(groups.iloc[train_rows])
    validation_groups = set(groups.iloc[validation_rows])
    test_groups = set(groups.iloc[test_rows])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)

    return {
        "train": (features.iloc[train_rows], target.iloc[train_rows]),
        "validation": (features.iloc[validation_rows], target.iloc[validation_rows]),
        "test": (features.iloc[test_rows], target.iloc[test_rows]),
    }


def log_lead_time(features):
    """Return a copy with lead_time replaced by ln(1 + lead_time)."""
    if features["lead_time"].lt(0).any():
        raise ValueError("lead_time must be non-negative for log1p.")
    transformed = features.copy()
    transformed["lead_time"] = np.log1p(transformed["lead_time"])
    return transformed


class RareCategoryGrouper(TransformerMixin, BaseEstimator):
    """Remember frequent categories in fit; replace other values in transform.

    The sklearn base classes let this step run inside a Pipeline.
    """

    def __init__(self, min_count=50):
        self.min_count = min_count

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_numpy()
        self.frequent_categories_ = {}

        for column in ["agent", "company", "country"]:
            counts = X[column].fillna("Unknown").value_counts()
            frequent = set(counts[counts >= self.min_count].index)
            # Missing values stay separate, even if they are rare in train.
            frequent.add("Unknown")
            self.frequent_categories_[column] = frequent

        return self

    def transform(self, X):
        check_is_fitted(self, "frequent_categories_")
        features = X.copy()

        for column, frequent in self.frequent_categories_.items():
            values = features[column].fillna("Unknown")
            keep_value = values.isin(frequent)
            features[column] = values.where(keep_value, "Other")

        return features

    def get_feature_names_out(self, input_features=None):
        # Grouping changes values, but keeps the same columns.
        return self.feature_names_in_.copy()


def build_preprocessor(rare_min_count=None):
    """Build separate numerical and categorical pipelines, fitted by the model."""
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    categorical_steps = []
    if rare_min_count is not None:
        categorical_steps.append(("rare_categories", RareCategoryGrouper(rare_min_count)))
    categorical_steps.extend([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), make_column_selector(dtype_include="number")),
        ("categorical", Pipeline(categorical_steps), make_column_selector(dtype_exclude="number")),
    ])
