"""Feature preparation and grouped data splits."""

from calendar import month_name
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

DATA_PATH = Path(__file__).resolve().parents[2] / "data/raw/hotel_bookings.csv"

# Exclude ablation fields so group keys remain unchanged when those fields are removed.
GROUPING_EXCLUDED_COLUMNS = ["assigned_room_type", "booking_changes", "days_in_waiting_list", "adr", "deposit_type"]
RARE_CATEGORY_COLUMNS = ["agent", "company", "country"]
MONTH_NUMBERS = {name: number for number, name in enumerate(month_name) if name}


def load_data(path=None):
    return pd.read_csv(path or DATA_PATH)


def prepare_features(df):
    df = df.dropna(subset=["is_canceled"])
    target = df["is_canceled"]
    if not target.isin([0, 1]).all():
        raise ValueError("is_canceled must contain only 0 or 1.")

    # The reservation status and its date reveal the outcome.
    features = df.drop(columns=["is_canceled", "reservation_status", "reservation_status_date"])

    text_columns = features.select_dtypes(include=["object", "string"]).columns
    features[text_columns] = features[text_columns].apply(lambda column: column.str.strip())
    features[["agent", "company"]] = features[["agent", "company"]].astype("string").fillna("Unknown")

    features["total_nights"] = features["stays_in_weekend_nights"] + features["stays_in_week_nights"]
    features["total_guests"] = features["adults"] + features["children"] + features["babies"]
    return features, target


def booking_groups(features):
    """Hash predictor values to group repeated inputs across splits and bootstrap samples."""
    return pd.util.hash_pandas_object(features.drop(columns=GROUPING_EXCLUDED_COLUMNS), index=False)


def grouped_cv(features, target, n_splits=3, random_state=72):
    """Create stratified grouped folds in the current row order."""
    groups = booking_groups(features)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(features, target, groups))
    for train_rows, validation_rows in folds:
        assert set(groups.iloc[train_rows]).isdisjoint(groups.iloc[validation_rows])
    return folds


def split_data(features, target, random_state=72):
    """Return train/validation/test pairs, keeping identical feature groups together."""
    groups = booking_groups(features)

    # Reserve 20% of groups for test, then 25% of the remaining groups for validation.
    development_rows, test_rows = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state).split(features, groups=groups))
    train_positions, validation_positions = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state).split(
            features.iloc[development_rows], groups=groups.iloc[development_rows]))
    parts = {
        "train": development_rows[train_positions],
        "validation": development_rows[validation_positions],
        "test": test_rows,
    }

    part_groups = {name: set(groups.iloc[rows]) for name, rows in parts.items()}
    assert part_groups["train"].isdisjoint(part_groups["validation"])
    assert part_groups["train"].isdisjoint(part_groups["test"])
    assert part_groups["validation"].isdisjoint(part_groups["test"])
    return {name: (features.iloc[rows], target.iloc[rows]) for name, rows in parts.items()}


def arrival_dates(bookings):
    """Build arrival dates from year, month and day."""
    return pd.to_datetime({
        "year": bookings["arrival_date_year"],
        "month": bookings["arrival_date_month"].map(MONTH_NUMBERS),
        "day": bookings["arrival_date_day_of_month"],
    })


def temporal_split(bookings, features, target):
    """Time-ordered train/validation/test parts for the exploratory temporal check.

    Train: arrivals before July 2016 whose outcome was known before July 2016.
    Validation: July-December 2016 arrivals with outcomes known before 2017.
    Test: 2017 arrivals. No feature group appears in two parts.
    """
    arrival = arrival_dates(bookings)
    status_date = pd.to_datetime(bookings["reservation_status_date"])
    masks = {
        "train": (arrival < "2016-07-01") & (status_date < "2016-07-01"),
        "validation": (arrival >= "2016-07-01") & (arrival < "2017-01-01") & (status_date < "2017-01-01"),
        "test": arrival >= "2017-01-01",
    }
    groups = {name: set(booking_groups(features.loc[mask])) for name, mask in masks.items()}
    assert groups["train"].isdisjoint(groups["validation"] | groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    return {name: (features.loc[mask], target.loc[mask]) for name, mask in masks.items()}


def quality_flags(features, bookings):
    """Flag unusual rows using features and raw bookings with matching indices."""
    return pd.DataFrame({
        "negative_adr": features["adr"].lt(0),
        "zero_guests": features["total_guests"].eq(0),
        "zero_nights": features["total_nights"].eq(0),
        "repeated_raw_record": bookings.duplicated(keep=False),
    })


def feature_overview(features):
    """Summarize each feature, counting Unknown as missing."""
    missing = features.isna() | features.eq("Unknown")
    overview = pd.DataFrame({
        "dtype": features.dtypes.astype(str),
        "missing": missing.sum(),
        "missing_pct": missing.mean() * 100,
        "unique": features.nunique(),
    })
    numeric = features.select_dtypes(include="number")
    return overview.join(numeric.agg(["min", "median", "max"]).T)


def log_lead_time(features):
    """Return a copy with lead_time replaced by ln(1 + lead_time)."""
    if features["lead_time"].lt(0).any():
        raise ValueError("lead_time must be non-negative for log1p.")
    transformed = features.copy()
    transformed["lead_time"] = np.log1p(transformed["lead_time"])
    return transformed


class RareCategoryGrouper(TransformerMixin, BaseEstimator):
    """Learn frequent categories on train and replace other values with Other.
    """

    def __init__(self, min_count=50):
        self.min_count = min_count

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_numpy()
        self.frequent_categories_ = {}
        for column in RARE_CATEGORY_COLUMNS:
            counts = X[column].fillna("Unknown").value_counts()
            frequent = counts.index if self.min_count is None else counts[counts >= self.min_count].index
            # Missing values stay separate, even if they are rare in train.
            self.frequent_categories_[column] = set(frequent) | {"Unknown"}
        return self

    def transform(self, X):
        check_is_fitted(self, "frequent_categories_")
        features = X.copy()
        for column, frequent in self.frequent_categories_.items():
            values = features[column].fillna("Unknown")
            features[column] = values.where(values.isin(frequent), "Other")
        return features

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()


def build_preprocessor(rare_min_count=None, log_lead=False):
    """Build numeric and categorical preprocessing pipelines."""
    log_step = [("log_lead", FunctionTransformer(log_lead_time, validate=False, feature_names_out="one-to-one"))]
    numeric_steps = (log_step if log_lead else []) + [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    categorical_steps = [
        ("rare_categories", RareCategoryGrouper(rare_min_count)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), make_column_selector(dtype_include="number")),
        ("categorical", Pipeline(categorical_steps), make_column_selector(dtype_exclude="number")),
    ])
