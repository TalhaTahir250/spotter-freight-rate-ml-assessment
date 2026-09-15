from __future__ import annotations

import pandas as pd


MAIN_NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "days_since_start",
    "weight_missing",
    "market_index_missing",
]

MAIN_CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

MAIN_FEATURES = (
    MAIN_NUMERIC_FEATURES
    + MAIN_CATEGORICAL_FEATURES
)


DECEMBER_NUMERIC_FEATURES = [
    "distance",
    "weight",
    "month",
    "day_of_week",
    "day_of_month",
    "days_since_start",
    "weight_missing",
]

DECEMBER_CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "lane",
]

DECEMBER_FEATURES = (
    DECEMBER_NUMERIC_FEATURES
    + DECEMBER_CATEGORICAL_FEATURES
)


def add_lane_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add origin-destination lane feature.

    Used by the reduced December model.
    """
    df = df.copy()

    df["lane"] = (
        df["pickup"].astype(str)
        + " -> "
        + df["delivery"].astype(str)
    )

    return df