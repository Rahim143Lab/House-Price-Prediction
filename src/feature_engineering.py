"""
Feature engineering for the House Price Prediction project.

All engineered features are derived only from information that is also
available at prediction time from the web form, so there is no target
leakage and no reliance on columns the deployed model cannot obtain.

Engineered features
--------------------
property_age
    How old the property is relative to the year it was sold (training) or
    the current year (inference). Older properties often sell for less,
    all else being equal.
years_since_renovation
    Years since the last remodel/addition. A recently renovated older home
    often behaves more like a newer home.
total_bathrooms
    Full bathrooms plus half bathrooms counted as 0.5 each -- a single
    number is both easier for a user to reason about and a stronger
    predictor than two separate sparse counts.
total_rooms
    Total rooms above grade (renamed passthrough of TotRms AbvGrd).
living_area_per_bedroom
    Living area divided by bedroom count (+1 to avoid division by zero).
    Captures "spaciousness" independent of raw square footage.
lot_area_per_living_area
    Ratio of lot size to living area. Distinguishes large-lot/small-house
    properties from small-lot/large-house properties.
garage_indicator
    1 if the property has at least one garage space, else 0.
has_fireplace
    1 if the property has at least one fireplace, else 0.
has_deck_or_porch
    1 if the property has a wood deck or open porch, else 0.
floors
    Approximate number of floors, derived from the House Style field
    (e.g. "1Story" -> 1.0, "2Story" -> 2.0, split-level styles -> 1.5).
"""

from __future__ import annotations

import pandas as pd

from src import config


def _year_reference(df: pd.DataFrame) -> pd.Series:
    """Return the reference year used to compute property age.

    During training, the Ames dataset's own "Yr Sold" column is the
    correct reference year. At inference time the incoming dataframe will
    not have "Yr Sold" (a user filling out a form today has no sale year
    yet), so we fall back to a fixed "current year" reference supplied by
    the caller via the 'Yr Sold' column being pre-filled in predict.py.
    """
    if "Yr Sold" in df.columns:
        return df["Yr Sold"]
    raise KeyError(
        "'Yr Sold' column is required to compute property age. "
        "src.predict fills this in automatically with the current year."
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to a copy of the input dataframe."""
    df = df.copy()

    ref_year = _year_reference(df)
    df["property_age"] = (ref_year - df["Year Built"]).clip(lower=0)
    df["years_since_renovation"] = (ref_year - df["Year Remod/Add"]).clip(lower=0)

    df["total_bathrooms"] = df["Full Bath"].fillna(0) + 0.5 * df["Half Bath"].fillna(0)
    df["total_rooms"] = df["TotRms AbvGrd"]

    df["living_area_per_bedroom"] = df["Gr Liv Area"] / (df["Bedroom AbvGr"].fillna(0) + 1)
    df["lot_area_per_living_area"] = df["Lot Area"] / df["Gr Liv Area"].replace(0, 1)

    df["garage_indicator"] = (df["Garage Cars"].fillna(0) > 0).astype(int)
    df["has_fireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)

    wood_deck = df.get("Wood Deck SF", pd.Series(0, index=df.index)).fillna(0)
    open_porch = df.get("Open Porch SF", pd.Series(0, index=df.index)).fillna(0)
    df["has_deck_or_porch"] = ((wood_deck + open_porch) > 0).astype(int)

    if "House Style" in df.columns:
        df["floors"] = (
            df["House Style"]
            .map(config.HOUSE_STYLE_TO_FLOORS)
            .fillna(config.DEFAULT_FLOORS)
        )
    elif "floors" not in df.columns:
        df["floors"] = config.DEFAULT_FLOORS

    drop_cols = [c for c in config.COLUMNS_TO_DROP_AFTER_FE if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the columns the model pipeline actually consumes."""
    missing = [c for c in config.ALL_MODEL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected model columns after feature engineering: {missing}")
    return df[config.ALL_MODEL_FEATURES]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline: engineer, then select model columns."""
    df = add_engineered_features(df)
    return select_model_columns(df)
