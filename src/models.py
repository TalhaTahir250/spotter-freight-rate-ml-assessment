from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    MAIN_NUMERIC_FEATURES,
    MAIN_CATEGORICAL_FEATURES,
    DECEMBER_NUMERIC_FEATURES,
    DECEMBER_CATEGORICAL_FEATURES,
)


def build_ridge_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    alpha: float = 100.0,
) -> Pipeline:
    """
    Build a Ridge regression pipeline with:
    - standardized numeric features
    - one-hot encoded categorical features
    - safe handling of unseen categories
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                Ridge(alpha=alpha),
            ),
        ]
    )

    return model


def build_main_model(
    alpha: float = 100.0,
) -> Pipeline:
    """
    Main model used for the 12,000-row validation set.
    """
    return build_ridge_pipeline(
        numeric_features=MAIN_NUMERIC_FEATURES,
        categorical_features=MAIN_CATEGORICAL_FEATURES,
        alpha=alpha,
    )


def build_december_model(
    alpha: float = 100.0,
) -> Pipeline:
    """
    Reduced-feature model used for the fixed December scenario.
    """
    return build_ridge_pipeline(
        numeric_features=DECEMBER_NUMERIC_FEATURES,
        categorical_features=DECEMBER_CATEGORICAL_FEATURES,
        alpha=alpha,
    )