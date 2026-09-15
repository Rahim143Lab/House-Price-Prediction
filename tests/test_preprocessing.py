"""Tests for feature engineering and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src import config, feature_engineering, preprocessing


def _sample_raw_row(**overrides):
    row = {
        "Lot Area": 8000,
        "Overall Qual": 6,
        "Overall Cond": 5,
        "Year Built": 1998,
        "Year Remod/Add": 2005,
        "Gr Liv Area": 1500,
        "Full Bath": 2,
        "Half Bath": 1,
        "Bedroom AbvGr": 3,
        "TotRms AbvGrd": 6,
        "Fireplaces": 1,
        "Garage Cars": 2,
        "Wood Deck SF": 100,
        "Open Porch SF": 0,
        "Yr Sold": 2010,
        "Neighborhood": "NAmes",
        "Bldg Type": "1Fam",
        "Central Air": "Y",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_engineered_features_present():
    df = feature_engineering.add_engineered_features(_sample_raw_row())
    for col in [
        "property_age",
        "years_since_renovation",
        "total_bathrooms",
        "total_rooms",
        "living_area_per_bedroom",
        "lot_area_per_living_area",
        "garage_indicator",
        "has_fireplace",
        "has_deck_or_porch",
    ]:
        assert col in df.columns


def test_property_age_computed_correctly():
    df = feature_engineering.add_engineered_features(_sample_raw_row())
    assert df.loc[0, "property_age"] == 2010 - 1998


def test_total_bathrooms_counts_half_bath_as_half():
    df = feature_engineering.add_engineered_features(_sample_raw_row())
    assert df.loc[0, "total_bathrooms"] == 2.5


def test_garage_indicator_zero_when_no_garage():
    df = feature_engineering.add_engineered_features(_sample_raw_row(**{"Garage Cars": 0}))
    assert df.loc[0, "garage_indicator"] == 0


def test_has_deck_or_porch_false_when_both_zero():
    df = feature_engineering.add_engineered_features(
        _sample_raw_row(**{"Wood Deck SF": 0, "Open Porch SF": 0})
    )
    assert df.loc[0, "has_deck_or_porch"] == 0


def test_property_age_never_negative():
    # Year Built after Yr Sold shouldn't happen in real data, but the
    # pipeline should not produce a negative age if it does.
    df = feature_engineering.add_engineered_features(
        _sample_raw_row(**{"Year Built": 2020, "Yr Sold": 2010})
    )
    assert df.loc[0, "property_age"] >= 0


def test_select_model_columns_returns_expected_columns():
    df = feature_engineering.build_features(_sample_raw_row())
    assert list(df.columns) == config.ALL_MODEL_FEATURES


def test_select_model_columns_raises_on_missing_columns():
    df = pd.DataFrame({"Lot Area": [8000]})
    with pytest.raises(KeyError):
        feature_engineering.select_model_columns(df)


def test_preprocessor_handles_missing_values():
    df = _sample_raw_row(**{"Garage Cars": np.nan})
    X = feature_engineering.build_features(df)
    preprocessor = preprocessing.build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert not np.isnan(transformed.toarray() if hasattr(transformed, "toarray") else transformed).any()


def test_preprocessor_handles_unknown_category_at_inference():
    train_df = pd.concat([_sample_raw_row(), _sample_raw_row(**{"Neighborhood": "Somerst"})])
    X_train = feature_engineering.build_features(train_df)
    preprocessor = preprocessing.build_preprocessor()
    preprocessor.fit(X_train)

    unseen_df = _sample_raw_row(**{"Neighborhood": "TotallyNewPlace"})
    X_unseen = feature_engineering.build_features(unseen_df)
    # Should not raise, thanks to handle_unknown="ignore"
    result = preprocessor.transform(X_unseen)
    assert result.shape[0] == 1
