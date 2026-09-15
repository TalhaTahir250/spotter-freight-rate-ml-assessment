import unittest

import numpy as np
import pandas as pd

from src.preprocessing import (
    apply_imputation,
    basic_cleaning,
    get_imputation_values,
)


class TestPreprocessing(unittest.TestCase):

    def test_basic_cleaning(self):
        df = pd.DataFrame(
            {
                "date": [
                    "2025-01-01",
                    "2025-01-02",
                ],
                "weight": [
                    -32000.0,
                    np.nan,
                ],
                "market_index": [
                    1.10,
                    np.nan,
                ],
            }
        )

        cleaned = basic_cleaning(df)

        # Negative weight should become positive
        self.assertEqual(
            cleaned.loc[0, "weight"],
            32000.0,
        )

        # Missing weight should remain missing
        # until training-derived imputation
        self.assertTrue(
            pd.isna(
                cleaned.loc[1, "weight"]
            )
        )

        # Missing-value indicators
        self.assertEqual(
            cleaned.loc[0, "weight_missing"],
            0,
        )

        self.assertEqual(
            cleaned.loc[1, "weight_missing"],
            1,
        )

        self.assertEqual(
            cleaned.loc[
                1,
                "market_index_missing",
            ],
            1,
        )

        # Calendar features
        self.assertEqual(
            cleaned.loc[0, "month"],
            1,
        )

        self.assertEqual(
            cleaned.loc[0, "day_of_month"],
            1,
        )

        self.assertEqual(
            cleaned.loc[0, "days_since_start"],
            0,
        )

        self.assertEqual(
            cleaned.loc[1, "days_since_start"],
            1,
        )


    def test_training_imputation(self):
        train = pd.DataFrame(
            {
                "weight": [
                    10000.0,
                    20000.0,
                    30000.0,
                ],
                "market_index": [
                    0.9,
                    1.0,
                    1.1,
                ],
            }
        )

        values = get_imputation_values(
            train
        )

        self.assertEqual(
            values["weight"],
            20000.0,
        )

        self.assertEqual(
            values["market_index"],
            1.0,
        )

        prediction_df = pd.DataFrame(
            {
                "weight": np.array(
                    [np.nan],
                    dtype=float,
                ),
                "market_index": np.array(
                    [np.nan],
                    dtype=float,
                ),
            }
        )

        result = apply_imputation(
            prediction_df,
            values,
        )

        self.assertEqual(
            result.loc[0, "weight"],
            20000.0,
        )

        self.assertEqual(
            result.loc[
                0,
                "market_index",
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()