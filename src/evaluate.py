from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Calculate standard regression metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def print_regression_metrics(
    name: str,
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Calculate and print regression metrics.
    """

    metrics = regression_metrics(
        y_true,
        y_pred,
    )

    print(name)
    print(
        f"MAE:  ${metrics['MAE']:,.2f}"
    )
    print(
        f"RMSE: ${metrics['RMSE']:,.2f}"
    )
    print(
        f"R²:   {metrics['R2']:.4f}"
    )

    return metrics


def prediction_quality_checks(
    predictions,
) -> dict[str, int | float]:
    """
    Basic safety checks for prediction arrays.
    """

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    return {
        "count": int(len(predictions)),
        "non_finite": int(
            (~np.isfinite(predictions)).sum()
        ),
        "non_positive": int(
            (predictions <= 0).sum()
        ),
        "minimum": float(
            predictions.min()
        ),
        "maximum": float(
            predictions.max()
        ),
        "mean": float(
            predictions.mean()
        ),
    }


def results_table(
    rows: list[dict],
    sort_by: str = "MAE",
) -> pd.DataFrame:
    """
    Create a sorted model-comparison table.
    """

    result = pd.DataFrame(rows)

    if sort_by in result.columns:
        result = result.sort_values(
            sort_by
        ).reset_index(drop=True)

    return result