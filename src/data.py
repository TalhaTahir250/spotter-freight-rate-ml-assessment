from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_train_data(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load labeled development data.
    """
    df = pd.read_csv(path)

    if "posted_rate" not in df.columns:
        raise ValueError(
            "Training data must contain 'posted_rate'."
        )

    return df


def load_validation_data(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load the 12,000-row hidden validation set.
    """
    df = pd.read_csv(path)

    if "posted_rate" in df.columns:
        raise ValueError(
            "Validation data should not contain 'posted_rate'."
        )

    return df


def load_prediction_template(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load Spotter's validation prediction template.
    """
    df = pd.read_csv(path)

    expected_columns = [
        "load_id",
        "predicted_rate",
    ]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Prediction template must contain "
            f"{expected_columns}."
        )

    return df


def load_december_data(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load the fixed December chart input file.
    """
    df = pd.read_csv(path)

    expected_columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"December file must contain "
            f"{expected_columns}."
        )

    return df