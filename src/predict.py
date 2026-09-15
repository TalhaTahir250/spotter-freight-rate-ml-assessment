from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_validation_submission(
    validation_df: pd.DataFrame,
    template_df: pd.DataFrame,
    predictions,
) -> pd.DataFrame:
    """
    Map validation predictions to Spotter's template by load_id.
    """

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if len(validation_df) != len(predictions):
        raise ValueError(
            "Prediction count does not match validation rows."
        )

    prediction_map = pd.Series(
        predictions,
        index=validation_df["load_id"].astype(str),
    )

    submission = template_df[
        ["load_id"]
    ].copy()

    submission["predicted_rate"] = (
        submission["load_id"]
        .astype(str)
        .map(prediction_map)
    )

    validate_validation_submission(
        submission
    )

    return submission


def validate_validation_submission(
    df: pd.DataFrame,
) -> None:
    """
    Validate the 12,000-row prediction output.
    """

    expected_columns = [
        "load_id",
        "predicted_rate",
    ]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Expected columns {expected_columns}."
        )

    if len(df) != 12000:
        raise ValueError(
            f"Expected 12000 rows, got {len(df)}."
        )

    if df["load_id"].duplicated().any():
        raise ValueError(
            "Duplicate load_id values found."
        )

    if df["predicted_rate"].isna().any():
        raise ValueError(
            "Missing predicted_rate values found."
        )

    values = df[
        "predicted_rate"
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite predictions found."
        )

    if (values <= 0).any():
        raise ValueError(
            "Predictions must be positive."
        )


def build_december_submission(
    december_df: pd.DataFrame,
    predictions,
) -> pd.DataFrame:
    """
    Create the seven-column December scorer file.
    """

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if len(december_df) != len(predictions):
        raise ValueError(
            "Prediction count does not match December rows."
        )

    columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]

    submission = december_df[
        columns
    ].copy()

    submission["predicted_rate"] = (
        predictions
    )

    submission["date"] = (
        pd.to_datetime(
            submission["date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    validate_december_submission(
        submission
    )

    return submission


def validate_december_submission(
    df: pd.DataFrame,
) -> None:
    """
    Validate basic December output requirements.
    """

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
            f"Expected columns {expected_columns}."
        )

    if len(df) != 31:
        raise ValueError(
            f"Expected 31 rows, got {len(df)}."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate December dates found."
        )

    if df["predicted_rate"].isna().any():
        raise ValueError(
            "Missing December predictions found."
        )

    values = df[
        "predicted_rate"
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite December predictions found."
        )

    if (values <= 0).any():
        raise ValueError(
            "December predictions must be positive."
        )


def save_submission(
    df: pd.DataFrame,
    path: str | Path,
) -> None:
    """
    Save a validated submission DataFrame.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )