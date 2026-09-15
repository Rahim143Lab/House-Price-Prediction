"""
Exploratory Data Analysis pipeline.

Generates a set of professional, labeled visualizations describing the
Ames Housing dataset and saves them to reports/figures/. Also prints a
text summary (shape, dtypes, missing values, duplicates, skewness) to
stdout/log so it can be captured in notebooks/01_data_exploration.ipynb.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config, data_loader, feature_engineering

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Return a dictionary summary of the dataset's key characteristics."""
    numeric_df = df.select_dtypes(include="number")
    summary = {
        "shape": df.shape,
        "n_duplicates": int(df.duplicated().sum()),
        "missing_values_top10": df.isna().sum().sort_values(ascending=False).head(10).to_dict(),
        "numeric_columns": len(numeric_df.columns),
        "categorical_columns": len(df.select_dtypes(include=["object", "str"]).columns),
        "target_skewness": round(float(df[config.TARGET_COLUMN].skew()), 3),
    }
    return summary


def plot_target_distribution(df: pd.DataFrame, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[config.TARGET_COLUMN], kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of Sale Price")
    ax.set_xlabel("Sale Price (USD)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir):
    numeric_cols = [
        "Lot Area", "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add",
        "Gr Liv Area", "Full Bath", "Half Bath", "Bedroom AbvGr", "TotRms AbvGrd",
        "Fireplaces", "Garage Cars", config.TARGET_COLUMN,
    ]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Key Numeric Features")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_price_vs_living_area(df: pd.DataFrame, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="Gr Liv Area", y=config.TARGET_COLUMN, alpha=0.5, ax=ax)
    ax.set_title("Sale Price vs. Living Area")
    ax.set_xlabel("Above-Grade Living Area (sq ft)")
    ax.set_ylabel("Sale Price (USD)")
    fig.tight_layout()
    fig.savefig(out_dir / "price_vs_living_area.png", dpi=150)
    plt.close(fig)


def plot_price_vs_bedrooms(df: pd.DataFrame, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Bedroom AbvGr", y=config.TARGET_COLUMN, ax=ax, color="#55A868")
    ax.set_title("Sale Price by Number of Bedrooms")
    ax.set_xlabel("Bedrooms (Above Grade)")
    ax.set_ylabel("Sale Price (USD)")
    fig.tight_layout()
    fig.savefig(out_dir / "price_vs_bedrooms.png", dpi=150)
    plt.close(fig)


def plot_price_vs_bathrooms(df: pd.DataFrame, out_dir):
    total_bath = df["Full Bath"].fillna(0) + 0.5 * df["Half Bath"].fillna(0)
    tmp = df.assign(total_bathrooms=total_bath)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=tmp, x="total_bathrooms", y=config.TARGET_COLUMN, ax=ax, color="#C44E52")
    ax.set_title("Sale Price by Total Bathrooms")
    ax.set_xlabel("Total Bathrooms (half counted as 0.5)")
    ax.set_ylabel("Sale Price (USD)")
    fig.tight_layout()
    fig.savefig(out_dir / "price_vs_bathrooms.png", dpi=150)
    plt.close(fig)


def plot_price_vs_property_age(df: pd.DataFrame, out_dir):
    tmp = df.assign(property_age=df["Yr Sold"] - df["Year Built"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=tmp, x="property_age", y=config.TARGET_COLUMN, alpha=0.5, ax=ax)
    ax.set_title("Sale Price vs. Property Age")
    ax.set_xlabel("Property Age at Time of Sale (years)")
    ax.set_ylabel("Sale Price (USD)")
    fig.tight_layout()
    fig.savefig(out_dir / "price_vs_property_age.png", dpi=150)
    plt.close(fig)


def plot_price_by_location(df: pd.DataFrame, out_dir):
    top_neighborhoods = df["Neighborhood"].value_counts().head(10).index
    tmp = df[df["Neighborhood"].isin(top_neighborhoods)]
    order = tmp.groupby("Neighborhood")[config.TARGET_COLUMN].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=tmp, x="Neighborhood", y=config.TARGET_COLUMN, order=order, ax=ax)
    ax.set_title("Sale Price by Neighborhood (Top 10 Most Common)")
    ax.set_xlabel("Neighborhood")
    ax.set_ylabel("Sale Price (USD)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "price_by_location.png", dpi=150)
    plt.close(fig)


def plot_boxplots_numeric(df: pd.DataFrame, out_dir):
    cols = ["Lot Area", "Gr Liv Area", "Overall Qual", "Garage Cars"]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 5))
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], ax=ax, color="#8172B2")
        ax.set_title(col)
    fig.suptitle("Box Plots of Key Numeric Features")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplots_numeric_features.png", dpi=150)
    plt.close(fig)


def run_eda(save_summary: bool = True):
    logging.basicConfig(level=logging.INFO)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = data_loader.load_raw_data()
    df = data_loader.basic_clean(df)

    summary = summarize_dataset(df)
    logger.info("Dataset summary: %s", summary)

    plot_target_distribution(df, config.FIGURES_DIR)
    plot_correlation_heatmap(df, config.FIGURES_DIR)
    plot_price_vs_living_area(df, config.FIGURES_DIR)
    plot_price_vs_bedrooms(df, config.FIGURES_DIR)
    plot_price_vs_bathrooms(df, config.FIGURES_DIR)
    plot_price_vs_property_age(df, config.FIGURES_DIR)
    plot_price_by_location(df, config.FIGURES_DIR)
    plot_boxplots_numeric(df, config.FIGURES_DIR)

    logger.info("Saved EDA figures to %s", config.FIGURES_DIR)
    return summary


if __name__ == "__main__":
    run_eda()
