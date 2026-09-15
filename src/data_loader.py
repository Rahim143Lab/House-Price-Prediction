"""
Data loading utilities.

Responsible for reading the raw Ames Housing dataset from disk and producing
reproducible train/validation/test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Ames Housing CSV file.

    Parameters
    ----------
    path: Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame with the raw dataset.

    Raises
    ------
    FileNotFoundError if the dataset is missing, with a clear message
    pointing the user to data/README.md.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "See data/README.md for instructions on obtaining the dataset."
        )
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and rows missing the target."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[config.TARGET_COLUMN])
    after = len(df)
    if before != after:
        logger.info("Dropped %d duplicate/target-missing rows", before - after)
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    train_size: float = config.TRAIN_SIZE,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
):
    """Split a dataframe into train / validation / test sets.

    The split is performed in two steps (train vs. rest, then val vs. test)
    so that the requested proportions are respected regardless of dataset
    size, using a fixed random_state for reproducibility.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-6, (
        "train_size + val_size + test_size must equal 1.0"
    )

    train_df, remainder_df = train_test_split(
        df, train_size=train_size, random_state=random_state, shuffle=True
    )

    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        remainder_df, train_size=relative_val_size, random_state=random_state, shuffle=True
    )

    logger.info(
        "Split data into train=%d, val=%d, test=%d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_and_split(save: bool = True):
    """Convenience function: load raw data, clean it, and split it.

    If save=True, persists the three splits to data/processed/ so that
    notebooks and scripts can reuse an identical split without recomputing.
    """
    df = load_raw_data()
    df = basic_clean(df)
    train_df, val_df, test_df = split_data(df)

    if save:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
        val_df.to_csv(config.VAL_DATA_PATH, index=False)
        test_df.to_csv(config.TEST_DATA_PATH, index=False)
        logger.info("Saved processed splits to %s", config.PROCESSED_DATA_DIR)

    return train_df, val_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_and_split()
