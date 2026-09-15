from __future__ import annotations

import pandas as pd


DATE_ORIGIN = pd.Timestamp("2025-01-01")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic cleaning and calendar feature creation.

    This function does not learn statistics from the data, so it is safe
    to apply independently to training and prediction datasets.
    """
    df = df.copy()

    # Prevent accidental target-derived EDA feature leakage
    df = df.drop(
        columns=["rate_per_mile_eda"],
        errors="ignore",
    )

    # Parse dates
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    # Track missing weight before imputation
    df["weight_missing"] = (
        df["weight"].isna().astype(int)
    )

    # Negative weights appear to be sign errors
    df["weight"] = df["weight"].abs()

    # Calendar features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Continuous time feature
    df["days_since_start"] = (
        df["date"] - DATE_ORIGIN
    ).dt.days

    # Only main train/validation data contains market_index
    if "market_index" in df.columns:
        df["market_index_missing"] = (
            df["market_index"]
            .isna()
            .astype(int)
        )

    return df


def get_imputation_values(
    train_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Learn imputation values from training data only.
    """
    values = {
        "weight": float(
            train_df["weight"].median()
        )
    }

    if "market_index" in train_df.columns:
        values["market_index"] = float(
            train_df["market_index"].median()
        )

    return values


def apply_imputation(
    df: pd.DataFrame,
    imputation_values: dict[str, float],
) -> pd.DataFrame:
    """
    Apply training-derived imputation statistics.
    """
    df = df.copy()

    for column, value in imputation_values.items():
        if column in df.columns:
            df[column] = (
                df[column].fillna(value)
            )

    return df