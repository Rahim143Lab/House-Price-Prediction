"""Shared utility functions: logging setup and metric computation."""

from __future__ import annotations

import logging
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, safe to call multiple times."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, MSE, RMSE, R2, and MAPE for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
        "MAE": round(float(mae), 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(rmse, 2),
        "R2": round(float(r2), 4),
        "MAPE": round(mape, 2),
    }
