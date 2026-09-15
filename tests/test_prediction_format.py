import unittest

import numpy as np
import pandas as pd

from src.predict import (
    build_december_submission,
    build_validation_submission,
    validate_december_submission,
    validate_validation_submission,
)


class TestPredictionFormat(unittest.TestCase):

    def test_validation_submission_format(self):
        # Validation IDs deliberately reversed to verify
        # that predictions are mapped by load_id, not row order.
        validation_ids = [
            f"TE-{i:06d}"
            for i in range(1, 12001)
        ]

        validation_df = pd.DataFrame(
            {
                "load_id": validation_ids[::-1],
            }
        )

        predictions = np.arange(
            1,
            12001,
            dtype=float,
        )

        template_df = pd.DataFrame(
            {
                "load_id": validation_ids,
                "predicted_rate": np.nan,
            }
        )

        submission = build_validation_submission(
            validation_df=validation_df,
            template_df=template_df,
            predictions=predictions,
        )

        self.assertEqual(
            submission.shape,
            (12000, 2),
        )

        self.assertEqual(
            submission.columns.tolist(),
            [
                "load_id",
                "predicted_rate",
            ],
        )

        self.assertEqual(
            submission["load_id"].nunique(),
            12000,
        )

        self.assertEqual(
            submission["predicted_rate"]
            .isna()
            .sum(),
            0,
        )

        self.assertTrue(
            (
                submission["predicted_rate"]
                > 0
            ).all()
        )

        # TE-000001 is the final row of validation_df,
        # so it should receive prediction 12000.
        self.assertEqual(
            submission.loc[
                0,
                "predicted_rate",
            ],
            12000.0,
        )


    def test_validation_rejects_non_positive_rate(self):
        df = pd.DataFrame(
            {
                "load_id": [
                    f"TE-{i:06d}"
                    for i in range(1, 12001)
                ],
                "predicted_rate": np.ones(
                    12000
                ),
            }
        )

        df.loc[
            0,
            "predicted_rate",
        ] = 0.0

        with self.assertRaises(
            ValueError
        ):
            validate_validation_submission(
                df
            )


    def test_december_submission_format(self):
        dates = pd.date_range(
            "2025-12-01",
            "2025-12-31",
        )

        december_df = pd.DataFrame(
            {
                "pickup": [
                    "Lexington"
                ] * 31,
                "delivery": [
                    "Fort Wayne"
                ] * 31,
                "distance": [
                    360.0
                ] * 31,
                "equipment": [
                    "Dry Van"
                ] * 31,
                "weight": [
                    32000.0
                ] * 31,
                "date": dates,
                "predicted_rate": [
                    np.nan
                ] * 31,
            }
        )

        predictions = np.linspace(
            850.0,
            900.0,
            31,
        )

        submission = build_december_submission(
            december_df=december_df,
            predictions=predictions,
        )

        self.assertEqual(
            submission.shape,
            (31, 7),
        )

        self.assertEqual(
            submission.columns.tolist(),
            [
                "pickup",
                "delivery",
                "distance",
                "equipment",
                "weight",
                "date",
                "predicted_rate",
            ],
        )

        self.assertEqual(
            submission["date"].nunique(),
            31,
        )

        self.assertEqual(
            submission[
                "predicted_rate"
            ]
            .isna()
            .sum(),
            0,
        )

        self.assertTrue(
            (
                submission[
                    "predicted_rate"
                ]
                > 0
            ).all()
        )


    def test_december_rejects_duplicate_date(self):
        dates = pd.date_range(
            "2025-12-01",
            "2025-12-31",
        ).strftime("%Y-%m-%d")

        df = pd.DataFrame(
            {
                "pickup": [
                    "Lexington"
                ] * 31,
                "delivery": [
                    "Fort Wayne"
                ] * 31,
                "distance": [
                    360.0
                ] * 31,
                "equipment": [
                    "Dry Van"
                ] * 31,
                "weight": [
                    32000.0
                ] * 31,
                "date": dates,
                "predicted_rate": [
                    880.0
                ] * 31,
            }
        )

        df.loc[
            1,
            "date",
        ] = df.loc[
            0,
            "date",
        ]

        with self.assertRaises(
            ValueError
        ):
            validate_december_submission(
                df
            )


    def test_december_rejects_non_positive_rate(self):
        dates = pd.date_range(
            "2025-12-01",
            "2025-12-31",
        ).strftime("%Y-%m-%d")

        df = pd.DataFrame(
            {
                "pickup": [
                    "Lexington"
                ] * 31,
                "delivery": [
                    "Fort Wayne"
                ] * 31,
                "distance": [
                    360.0
                ] * 31,
                "equipment": [
                    "Dry Van"
                ] * 31,
                "weight": [
                    32000.0
                ] * 31,
                "date": dates,
                "predicted_rate": [
                    880.0
                ] * 31,
            }
        )

        df.loc[
            0,
            "predicted_rate",
        ] = -1.0

        with self.assertRaises(
            ValueError
        ):
            validate_december_submission(
                df
            )


if __name__ == "__main__":
    unittest.main()