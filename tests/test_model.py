"""Tests for model loading and the prediction module."""

import pytest

from src import config
from src.predict import (
    HouseFeatures,
    InvalidInputError,
    ModelNotLoadedError,
    get_model_info,
    predict_house_price,
)


def _valid_features(**overrides) -> HouseFeatures:
    defaults = dict(
        bedrooms=3,
        bathrooms=2.0,
        living_area=1500,
        lot_area=8000,
        floors=1.0,
        garage_cars=2,
        year_built=1998,
        year_remodeled=2005,
        neighborhood="NAmes",
        bldg_type="1Fam",
        overall_qual=6,
        overall_cond=5,
        total_rooms=6,
        fireplaces=1,
        central_air="Y",
        has_deck_or_porch=True,
    )
    defaults.update(overrides)
    return HouseFeatures(**defaults)


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Model must be trained first")
def test_model_file_exists():
    assert config.MODEL_PATH.exists()


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Model must be trained first")
def test_prediction_returns_number():
    result = predict_house_price(_valid_features())
    assert isinstance(result["predicted_price"], float)
    assert result["predicted_price"] > 0


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Model must be trained first")
def test_prediction_returns_expected_keys():
    result = predict_house_price(_valid_features())
    assert "predicted_price" in result
    assert "currency" in result
    assert "model" in result
    assert result["currency"] == config.CURRENCY


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Model must be trained first")
def test_larger_and_higher_quality_home_predicts_higher_price():
    small = predict_house_price(_valid_features(living_area=900, overall_qual=3))
    large = predict_house_price(_valid_features(living_area=3500, overall_qual=9))
    assert large["predicted_price"] > small["predicted_price"]


def test_invalid_input_rejected_bad_neighborhood():
    with pytest.raises(InvalidInputError):
        predict_house_price(_valid_features(neighborhood="NotARealPlace"))


def test_invalid_input_rejected_bad_bldg_type():
    with pytest.raises(InvalidInputError):
        predict_house_price(_valid_features(bldg_type="Castle"))


def test_invalid_input_rejected_year_remodeled_before_built():
    with pytest.raises(InvalidInputError):
        predict_house_price(_valid_features(year_built=2010, year_remodeled=1990))


def test_invalid_input_rejected_out_of_range_quality():
    with pytest.raises(InvalidInputError):
        predict_house_price(_valid_features(overall_qual=99))


def test_invalid_input_rejected_future_year_built():
    with pytest.raises(InvalidInputError):
        predict_house_price(_valid_features(year_built=3000))


@pytest.mark.skipif(not config.MODEL_METADATA_PATH.exists(), reason="Model must be trained first")
def test_get_model_info_contains_expected_fields():
    info = get_model_info()
    assert "model_name" in info
    assert "test_metrics" in info
    assert "valid_neighborhoods" in info
