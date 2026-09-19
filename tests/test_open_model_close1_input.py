import unittest

import pandas as pd

from ufc_betting.Models.RiskManagement.open_model_close1_input import (
    _align_model_design,
)


class OpenModelDesignTests(unittest.TestCase):
    def test_maps_scaled_elo_pred_to_encoded_model_name(self):
        scaled = pd.DataFrame(
            {"feature_a": [0.25, -0.5], "elo_pred": [-1.0, 1.0]},
            index=[4, 9],
        )
        raw = pd.DataFrame(
            {"feature_a": [10.0, 20.0], "elo_pred": [0, 1]},
            index=[4, 9],
        )

        design = _align_model_design(
            scaled,
            raw,
            ["const", "feature_a", "elo_pred_1"],
        )

        self.assertEqual(list(design.columns), ["const", "feature_a", "elo_pred_1"])
        self.assertEqual(list(design.index), [4, 9])
        self.assertEqual(design["const"].tolist(), [1.0, 1.0])
        self.assertEqual(design["elo_pred_1"].tolist(), [-1.0, 1.0])

    def test_accepts_model_name_without_underscore(self):
        scaled = pd.DataFrame({"elo_pred": [-1.0, 1.0]})
        design = _align_model_design(scaled, pd.DataFrame(), ["elo_pred1"])
        self.assertEqual(design["elo_pred1"].tolist(), [-1.0, 1.0])

    def test_builds_encoded_elo_column_when_scaler_excludes_category(self):
        scaled = pd.DataFrame({"feature_a": [0.25, -0.5]}, index=[4, 9])
        raw = pd.DataFrame({"elo_pred": [0, 1]}, index=[4, 9])

        design = _align_model_design(
            scaled,
            raw,
            ["const", "feature_a", "elo_pred_1"],
        )

        self.assertEqual(design["elo_pred_1"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
