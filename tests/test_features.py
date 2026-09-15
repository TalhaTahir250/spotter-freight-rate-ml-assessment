import unittest

import pandas as pd

from src.features import (
    DECEMBER_FEATURES,
    MAIN_FEATURES,
    add_lane_feature,
)


class TestFeatures(unittest.TestCase):

    def test_main_feature_count(self):
        self.assertEqual(
            len(MAIN_FEATURES),
            18,
        )


    def test_december_feature_count(self):
        self.assertEqual(
            len(DECEMBER_FEATURES),
            11,
        )


    def test_lane_feature(self):
        df = pd.DataFrame(
            {
                "pickup": [
                    "Lexington",
                    "Boston",
                ],
                "delivery": [
                    "Fort Wayne",
                    "Chicago",
                ],
            }
        )

        result = add_lane_feature(df)

        self.assertIn(
            "lane",
            result.columns,
        )

        self.assertEqual(
            result.loc[0, "lane"],
            "Lexington -> Fort Wayne",
        )

        self.assertEqual(
            result.loc[1, "lane"],
            "Boston -> Chicago",
        )


    def test_original_dataframe_not_modified(self):
        df = pd.DataFrame(
            {
                "pickup": ["Lexington"],
                "delivery": ["Fort Wayne"],
            }
        )

        _ = add_lane_feature(df)

        self.assertNotIn(
            "lane",
            df.columns,
        )


    def test_no_target_in_features(self):
        self.assertNotIn(
            "posted_rate",
            MAIN_FEATURES,
        )

        self.assertNotIn(
            "posted_rate",
            DECEMBER_FEATURES,
        )


    def test_no_load_id_in_features(self):
        self.assertNotIn(
            "load_id",
            MAIN_FEATURES,
        )

        self.assertNotIn(
            "load_id",
            DECEMBER_FEATURES,
        )


if __name__ == "__main__":
    unittest.main()