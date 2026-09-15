"""
Evaluation utilities for the final trained model.

Generates:
- reports/figures/actual_vs_predicted.png
- reports/figures/residual_distribution.png
- reports/figures/feature_importance.png
- Prints test-set metrics (loaded from the model trained by src/train.py)

Run with: python -m src.evaluate
"""

from __future__ import annotations

import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config, data_loader, feature_engineering, utils

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(config.MODEL_PATH)


def plot_actual_vs_predicted(y_true, y_pred, out_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Sale Price (USD)")
    ax.set_ylabel("Predicted Sale Price (USD)")
    ax.set_title("Actual vs. Predicted Sale Price (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_residual_distribution(y_true, y_pred, out_dir):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, ax=ax, color="#C44E52")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Residual Distribution (Actual - Predicted)")
    ax.set_xlabel("Residual (USD)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_distribution.png", dpi=150)
    plt.close(fig)


def get_feature_importance(pipeline) -> pd.DataFrame:
    """Extract feature importance (tree models) or coefficients (linear models)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    feature_names = [f.split("__", 1)[-1] for f in feature_names]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        kind = "importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        kind = "abs_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.attrs["kind"] = kind
    return df


def plot_feature_importance(importance_df: pd.DataFrame, out_dir, top_n: int = 15):
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"], color="#55A868")
    ax.set_title("Top Feature Importances (Final Model)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    utils.setup_logging()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = load_model()

    _, _, test_df = data_loader.load_and_split(save=False)
    X_test = feature_engineering.build_features(test_df)
    y_test = test_df[config.TARGET_COLUMN].values

    y_pred = pipeline.predict(X_test)
    metrics = utils.compute_regression_metrics(y_test, y_pred)
    logger.info("Test metrics: %s", metrics)
    print("Test metrics:", metrics)

    plot_actual_vs_predicted(y_test, y_pred, config.FIGURES_DIR)
    plot_residual_distribution(y_test, y_pred, config.FIGURES_DIR)

    importance_df = get_feature_importance(pipeline)
    if not importance_df.empty:
        plot_feature_importance(importance_df, config.FIGURES_DIR)
        importance_df.to_csv(config.METRICS_DIR / "feature_importance.csv", index=False)
        logger.info("Top 5 features: %s", importance_df.head(5).to_dict("records"))

    logger.info("Saved evaluation figures to %s", config.FIGURES_DIR)
    return metrics


if __name__ == "__main__":
    main()
