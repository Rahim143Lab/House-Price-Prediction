"""
Centralized configuration for the House Price Prediction project.

Keeping all paths, constants, and settings in one place avoids hard-coding
values throughout the codebase and makes the project easy to reconfigure.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "AmesHousing.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "house_price_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"
BEST_PARAMS_PATH = METRICS_DIR / "best_params.json"
FINAL_METRICS_PATH = METRICS_DIR / "final_test_metrics.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Train / validation / test split ratios
# ---------------------------------------------------------------------------
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COLUMN = "SalePrice"

# ---------------------------------------------------------------------------
# Raw Ames Housing columns used by this project.
#
# The full Ames Housing dataset has 80+ columns. This project intentionally
# uses a curated, real-estate-meaningful subset that mirrors the fields a
# normal home-value estimator would ask a user for, so that every feature
# the model needs can actually be collected from a simple web form.
# ---------------------------------------------------------------------------
RAW_NUMERIC_FEATURES = [
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Gr Liv Area",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Cars",
    "Wood Deck SF",
    "Open Porch SF",
    "Yr Sold",
]

RAW_CATEGORICAL_FEATURES = [
    "Neighborhood",
    "Bldg Type",
    "House Style",
    "Central Air",
]

# Columns dropped after feature engineering (raw inputs used only to derive
# engineered features, or identifiers not useful for prediction).
COLUMNS_TO_DROP_AFTER_FE = ["Yr Sold", "House Style"]

# Engineered numeric features created in src/feature_engineering.py
ENGINEERED_NUMERIC_FEATURES = [
    "property_age",
    "years_since_renovation",
    "total_bathrooms",
    "total_rooms",
    "living_area_per_bedroom",
    "lot_area_per_living_area",
    "garage_indicator",
    "has_fireplace",
    "has_deck_or_porch",
    "floors",
]

# Final feature lists used by the preprocessing ColumnTransformer.
NUMERIC_FEATURES = [
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Garage Cars",
    "property_age",
    "years_since_renovation",
    "total_bathrooms",
    "total_rooms",
    "living_area_per_bedroom",
    "lot_area_per_living_area",
    "garage_indicator",
    "has_fireplace",
    "has_deck_or_porch",
    "floors",
]

CATEGORICAL_FEATURES = [
    "Neighborhood",
    "Bldg Type",
    "Central Air",
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Reasonable input limits used for API/frontend validation
# (prevents nonsensical or abusive input values).
# ---------------------------------------------------------------------------
INPUT_LIMITS = {
    "lot_area": (500, 250000),
    "overall_qual": (1, 10),
    "overall_cond": (1, 10),
    "living_area": (200, 15000),
    "garage_cars": (0, 5),
    "year_built": (1870, 2026),
    "year_remodeled": (1870, 2026),
    "full_bath": (0, 6),
    "half_bath": (0, 4),
    "bedrooms": (0, 12),
    "total_rooms": (1, 20),
    "fireplaces": (0, 5),
    "wood_deck_sf": (0, 2000),
    "open_porch_sf": (0, 2000),
}

VALID_NEIGHBORHOODS = None  # populated at runtime from training data (see utils.py)
VALID_BLDG_TYPES = ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"]
VALID_CENTRAL_AIR = ["Y", "N"]

# House Style -> approximate number of floors, used for feature engineering.
HOUSE_STYLE_TO_FLOORS = {
    "1Story": 1.0,
    "1.5Fin": 1.5,
    "1.5Unf": 1.5,
    "2Story": 2.0,
    "2.5Fin": 2.5,
    "2.5Unf": 2.5,
    "SFoyer": 1.5,
    "SLvl": 1.5,
}
DEFAULT_FLOORS = 1.0

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
API_TITLE = "House Price Prediction API"
API_DESCRIPTION = (
    "REST API that predicts estimated house prices from property features "
    "using a trained regression model."
)
API_VERSION = "1.0.0"
ALLOWED_ORIGINS = ["*"]  # portfolio project: open CORS, no auth/secrets involved

CURRENCY = "USD"
