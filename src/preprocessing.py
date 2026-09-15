"""
Preprocessing pipeline built with Scikit-learn's Pipeline and
ColumnTransformer.

The transformer is fit ONLY on training data (see src/train.py) to avoid
preprocessing/data leakage. Numeric columns are median-imputed; categorical
columns are most-frequent-imputed and one-hot encoded with
handle_unknown="ignore" so unseen categories at inference time do not crash
the API.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config


def build_preprocessor() -> ColumnTransformer:
    """Construct the ColumnTransformer used ahead of every regression model."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
