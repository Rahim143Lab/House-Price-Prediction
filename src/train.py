"""
Model training pipeline.

Trains multiple regression models on the training split, evaluates each on
the validation split (to select the best model without touching the test
set), performs hyperparameter tuning on the strongest tree-based model, and
finally retrains that model's pipeline and evaluates it once on the held-out
test set.

Run with:  python -m src.train
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

from src import config, data_loader, feature_engineering, preprocessing, utils

logger = logging.getLogger(__name__)


def _prepare_xy(df: pd.DataFrame):
    """Apply feature engineering and split into X (features) / y (target)."""
    X = feature_engineering.build_features(df)
    y = df[config.TARGET_COLUMN].values
    return X, y


def get_candidate_models() -> dict:
    """Return the dictionary of baseline regression models to compare."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=100.0, random_state=config.RANDOM_STATE, max_iter=10000),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    return models


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            ("model", model),
        ]
    )


def train_and_compare(X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """Train every candidate model and evaluate on the validation set.

    Returns a comparison dataframe sorted by validation R2 (descending),
    and also reports training-set metrics so overfitting can be inspected.
    """
    rows = []
    fitted_pipelines = {}

    for name, model in get_candidate_models().items():
        logger.info("Training %s ...", name)
        pipeline = build_pipeline(model)
        start = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start

        train_pred = pipeline.predict(X_train)
        val_pred = pipeline.predict(X_val)

        train_metrics = utils.compute_regression_metrics(y_train, train_pred)
        val_metrics = utils.compute_regression_metrics(y_val, val_pred)

        rows.append(
            {
                "Model": name,
                "Train_MAE": train_metrics["MAE"],
                "Train_RMSE": train_metrics["RMSE"],
                "Train_R2": train_metrics["R2"],
                "Val_MAE": val_metrics["MAE"],
                "Val_RMSE": val_metrics["RMSE"],
                "Val_R2": val_metrics["R2"],
                "Val_MAPE": val_metrics["MAPE"],
                "Train_Time_Sec": round(elapsed, 2),
            }
        )
        fitted_pipelines[name] = pipeline
        logger.info(
            "%s -> Val R2=%.4f, Val RMSE=%.0f (train R2=%.4f)",
            name,
            val_metrics["R2"],
            val_metrics["RMSE"],
            train_metrics["R2"],
        )

    comparison_df = pd.DataFrame(rows).sort_values("Val_R2", ascending=False).reset_index(drop=True)
    return comparison_df, fitted_pipelines


def tune_best_tree_model(model_name: str, X_train, y_train):
    """Run RandomizedSearchCV for the strongest tree-based model."""
    if model_name == "Random Forest":
        base_model = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
        param_distributions = {
            "model__n_estimators": [200, 300, 400, 600],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5, 1.0],
        }
    elif model_name == "Gradient Boosting":
        base_model = GradientBoostingRegressor(random_state=config.RANDOM_STATE)
        param_distributions = {
            "model__n_estimators": [100, 200, 300, 400],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_samples_split": [2, 5, 10],
        }
    elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
        base_model = XGBRegressor(
            random_state=config.RANDOM_STATE, n_jobs=-1, objective="reg:squarederror"
        )
        param_distributions = {
            "model__n_estimators": [200, 300, 400, 600],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }
    else:
        raise ValueError(f"Tuning not configured for model: {model_name}")

    pipeline = build_pipeline(base_model)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="r2",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    logger.info("Running RandomizedSearchCV for %s ...", model_name)
    search.fit(X_train, y_train)
    logger.info("Best params for %s: %s", model_name, search.best_params_)
    return search.best_estimator_, search.best_params_, search.best_score_


def main():
    utils.setup_logging()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = data_loader.load_and_split()

    X_train, y_train = _prepare_xy(train_df)
    X_val, y_val = _prepare_xy(val_df)
    X_test, y_test = _prepare_xy(test_df)

    # --- Step 1: train & compare baseline models on train/validation ---
    comparison_df, fitted_pipelines = train_and_compare(X_train, y_train, X_val, y_val)
    comparison_df.to_csv(config.MODEL_COMPARISON_PATH, index=False)
    logger.info("Saved model comparison table to %s", config.MODEL_COMPARISON_PATH)
    print("\n=== Model Comparison (sorted by validation R2) ===")
    print(comparison_df.to_string(index=False))

    # --- Step 2: pick the strongest tree-based model for tuning ---
    tree_models = comparison_df[comparison_df["Model"].isin(
        ["Random Forest", "Gradient Boosting", "XGBoost"]
    )]
    best_tree_model_name = tree_models.iloc[0]["Model"]
    logger.info("Selected %s for hyperparameter tuning", best_tree_model_name)

    # Tune using train+val combined is avoided; tune with CV on train only.
    best_estimator, best_params, best_cv_score = tune_best_tree_model(
        best_tree_model_name, X_train, y_train
    )

    tuned_val_pred = best_estimator.predict(X_val)
    tuned_val_metrics = utils.compute_regression_metrics(y_val, tuned_val_pred)
    logger.info("Tuned %s validation metrics: %s", best_tree_model_name, tuned_val_metrics)

    # Compare tuned vs. the best untuned baseline; keep whichever is better
    # on the validation set (tuning should not make things worse).
    best_baseline_row = comparison_df.iloc[0]
    if tuned_val_metrics["R2"] >= best_baseline_row["Val_R2"]:
        final_model_name = best_tree_model_name + " (Tuned)"
        final_pipeline = best_estimator
        final_val_metrics = tuned_val_metrics
    else:
        final_model_name = best_baseline_row["Model"]
        final_pipeline = fitted_pipelines[best_baseline_row["Model"]]
        final_val_metrics = {
            "MAE": best_baseline_row["Val_MAE"],
            "RMSE": best_baseline_row["Val_RMSE"],
            "R2": best_baseline_row["Val_R2"],
            "MAPE": best_baseline_row["Val_MAPE"],
        }
        best_params = None

    logger.info("Final selected model: %s", final_model_name)

    with open(config.BEST_PARAMS_PATH, "w") as f:
        json.dump(
            {"selected_model": final_model_name, "best_params": best_params},
            f,
            indent=2,
        )

    # --- Step 3: retrain final pipeline on train+val, evaluate once on test ---
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val, y_train_val = _prepare_xy(train_val_df)

    logger.info("Retraining final pipeline on combined train+validation data ...")
    final_pipeline.fit(X_train_val, y_train_val)

    test_pred = final_pipeline.predict(X_test)
    test_metrics = utils.compute_regression_metrics(y_test, test_pred)
    logger.info("FINAL TEST METRICS (%s): %s", final_model_name, test_metrics)
    print(f"\n=== Final Model: {final_model_name} ===")
    print(f"Test metrics: {test_metrics}")

    with open(config.FINAL_METRICS_PATH, "w") as f:
        json.dump(
            {
                "model": final_model_name,
                "validation_metrics": final_val_metrics,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
            default=str,
        )

    # --- Step 4: save model + metadata ---
    joblib.dump(final_pipeline, config.MODEL_PATH)
    logger.info("Saved final trained pipeline to %s", config.MODEL_PATH)

    metadata = {
        "model_name": final_model_name,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_state": config.RANDOM_STATE,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "engineered_features": config.ENGINEERED_NUMERIC_FEATURES,
        "target": config.TARGET_COLUMN,
        "dataset": {
            "name": "Ames Housing Dataset (De Cock, 2011)",
            "n_rows_total": int(len(train_df) + len(val_df) + len(test_df)),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        },
        "metrics": {
            "validation": final_val_metrics,
            "test": test_metrics,
        },
        "version": "1.0.0",
    }
    with open(config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved model metadata to %s", config.MODEL_METADATA_PATH)

    return metadata


if __name__ == "__main__":
    main()
