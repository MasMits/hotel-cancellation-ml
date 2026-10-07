from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[2] / "data/raw/hotel_bookings.csv"


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


def build_preprocessor():
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Unknown"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ]
    )
