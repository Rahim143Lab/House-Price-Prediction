"""
Prediction module.

Loads the saved model pipeline once and exposes predict_house_price(), a
reusable function the FastAPI backend (and tests) import directly rather
than duplicating loading/validation logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import joblib
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when the trained model artifact cannot be found/loaded."""


class InvalidInputError(ValueError):
    """Raised when supplied property features fail validation."""


@dataclass
class HouseFeatures:
    """Input features accepted from the web form / API."""

    bedrooms: int
    bathrooms: float
    living_area: float
    lot_area: float
    floors: float
    garage_cars: int
    year_built: int
    year_remodeled: Optional[int]
    neighborhood: str
    bldg_type: str
    overall_qual: int
    overall_cond: int
    total_rooms: int
    fireplaces: int = 0
    central_air: str = "Y"
    has_deck_or_porch: bool = False


@lru_cache(maxsize=1)
def _load_pipeline():
    if not config.MODEL_PATH.exists():
        raise ModelNotLoadedError(
            f"Model file not found at {config.MODEL_PATH}. Train a model first with "
            "`python -m src.train`."
        )
    logger.info("Loading model pipeline from %s", config.MODEL_PATH)
    return joblib.load(config.MODEL_PATH)


@lru_cache(maxsize=1)
def _load_metadata() -> dict:
    if not config.MODEL_METADATA_PATH.exists():
        return {}
    with open(config.MODEL_METADATA_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _valid_neighborhoods() -> tuple:
    """Return the set of neighborhoods seen during training, for validation."""
    if config.RAW_DATA_PATH.exists():
        df = pd.read_csv(config.RAW_DATA_PATH, usecols=["Neighborhood"])
        return tuple(sorted(df["Neighborhood"].dropna().unique().tolist()))
    return tuple()


def get_model_info() -> dict:
    """Return model metadata for the /model-info API endpoint."""
    metadata = _load_metadata()
    return {
        "model_name": metadata.get("model_name", "unknown"),
        "version": metadata.get("version", "unknown"),
        "training_date": metadata.get("training_date"),
        "test_metrics": metadata.get("metrics", {}).get("test", {}),
        "supported_features": {
            "numeric": config.NUMERIC_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
        },
        "valid_neighborhoods": list(_valid_neighborhoods()),
        "valid_bldg_types": config.VALID_BLDG_TYPES,
    }


def _validate(features: HouseFeatures) -> None:
    limits = config.INPUT_LIMITS

    def check(name, value, limit_key):
        lo, hi = limits[limit_key]
        if value is None:
            return
        if not (lo <= value <= hi):
            raise InvalidInputError(f"{name} must be between {lo} and {hi} (got {value}).")

    check("bedrooms", features.bedrooms, "bedrooms")
    check("lot_area", features.lot_area, "lot_area")
    check("living_area", features.living_area, "living_area")
    check("garage_cars", features.garage_cars, "garage_cars")
    check("year_built", features.year_built, "year_built")
    check("overall_qual", features.overall_qual, "overall_qual")
    check("overall_cond", features.overall_cond, "overall_cond")
    check("total_rooms", features.total_rooms, "total_rooms")
    check("fireplaces", features.fireplaces, "fireplaces")

    year_remod = features.year_remodeled if features.year_remodeled else features.year_built
    check("year_remodeled", year_remod, "year_remodeled")
    if year_remod < features.year_built:
        raise InvalidInputError("year_remodeled cannot be earlier than year_built.")

    if features.central_air not in config.VALID_CENTRAL_AIR:
        raise InvalidInputError(f"central_air must be one of {config.VALID_CENTRAL_AIR}.")

    if features.bldg_type not in config.VALID_BLDG_TYPES:
        raise InvalidInputError(f"bldg_type must be one of {config.VALID_BLDG_TYPES}.")

    valid_neighborhoods = _valid_neighborhoods()
    if valid_neighborhoods and features.neighborhood not in valid_neighborhoods:
        raise InvalidInputError(
            f"neighborhood '{features.neighborhood}' is not recognized. "
            f"See /model-info for the list of supported neighborhoods."
        )

    current_year = datetime.now(timezone.utc).year
    if features.year_built > current_year:
        raise InvalidInputError("year_built cannot be in the future.")


def _features_to_raw_row(features: HouseFeatures) -> pd.DataFrame:
    """Map HouseFeatures (user-facing) to the raw Ames-style columns expected
    by src.feature_engineering.build_features(), then let that module derive
    all engineered features exactly as done at training time.
    """
    year_remod = features.year_remodeled if features.year_remodeled else features.year_built

    # Reconstruct approximate Full Bath / Half Bath from total bathrooms:
    # whole number -> full baths, .5 remainder -> one half bath.
    full_bath = int(features.bathrooms)
    half_bath = 1 if (features.bathrooms - full_bath) >= 0.5 else 0

    wood_deck_sf = 100 if features.has_deck_or_porch else 0
    open_porch_sf = 0

    row = {
        "Lot Area": features.lot_area,
        "Overall Qual": features.overall_qual,
        "Overall Cond": features.overall_cond,
        "Year Built": features.year_built,
        "Year Remod/Add": year_remod,
        "Gr Liv Area": features.living_area,
        "Full Bath": full_bath,
        "Half Bath": half_bath,
        "Bedroom AbvGr": features.bedrooms,
        "TotRms AbvGrd": features.total_rooms,
        "Fireplaces": features.fireplaces,
        "Garage Cars": features.garage_cars,
        "Wood Deck SF": wood_deck_sf,
        "Open Porch SF": open_porch_sf,
        # Use the current year as the "as of" reference for property_age /
        # years_since_renovation, since a live prediction has no sale year yet.
        "Yr Sold": datetime.now(timezone.utc).year,
        "Neighborhood": features.neighborhood,
        "Bldg Type": features.bldg_type,
        "Central Air": features.central_air,
        # floors is supplied directly rather than derived from House Style,
        # since the form asks the user for floors directly.
        "floors": features.floors,
    }
    return pd.DataFrame([row])


def predict_house_price(features: HouseFeatures) -> dict:
    """Validate input, run the trained pipeline, and return a formatted result."""
    from src import feature_engineering  # local import avoids a circular import at module load

    _validate(features)
    pipeline = _load_pipeline()

    raw_row = _features_to_raw_row(features)
    X = feature_engineering.select_model_columns(
        feature_engineering.add_engineered_features(raw_row)
    )

    prediction = float(pipeline.predict(X)[0])
    prediction = max(prediction, 0.0)

    metadata = _load_metadata()
    test_metrics = metadata.get("metrics", {}).get("test", {})
    mae = test_metrics.get("MAE")

    result = {
        "predicted_price": round(prediction, 2),
        "currency": config.CURRENCY,
        "model": metadata.get("model_name", "unknown"),
    }
    if mae is not None:
        # Approximate uncertainty band: +/- the model's historical test MAE.
        # This is explicitly NOT a statistical confidence interval.
        result["estimated_range"] = {
            "low": round(max(prediction - mae, 0.0), 2),
            "high": round(prediction + mae, 2),
            "basis": "±1 mean absolute error (MAE) observed on the held-out test set",
        }
    return result
