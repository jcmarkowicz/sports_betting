import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ufc_betting.Models.RiskManagement.odds_sensitivity import (
    american_to_decimal, decimal_to_american, improve_prices, solve_bin, load_paired,
)


def example_group(wins=(True, False), target_prices=(3.0, 3.0)):
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "close_american": [100.0, 100.0], "close_decimal": [2.0, 2.0],
        "close_win": wins, "open_pick_decimal": target_prices,
        "open_win": [True, False], "same_choice_open_decimal": [3.0, 3.0],
        "pred_winner_close": [1, 0], "pred_winner_open": [1, 0],
    })


class OddsSensitivityTests(unittest.TestCase):
    def test_pairing_uses_fight_keys_and_each_models_own_pick(self):
        opening = pd.DataFrame({
            "date": ["2020-01-01"] * 2, "fighter_red": ["a", "c"],
            "fighter_blue": ["b", "d"], "pred_winner": [0, 1],
            "winner": [0, 1], "open_red": [-200, -150], "open_blue": [150, 125],
        })
        closing = opening.drop(columns=["open_red", "open_blue"]).copy()
        closing["pred_winner"] = [1, 1]
        closing["close1_red"] = [-250, -200]
        closing["close1_blue"] = [200, 150]
        with TemporaryDirectory() as directory:
            opening.to_csv(Path(directory) / "test_logit_open.csv", index=False)
            closing.iloc[::-1].to_csv(Path(directory) / "test_logit_close1.csv", index=False)
            paired, excluded = load_paired(directory, "test", "close1")
        first = paired.set_index("fighter_red").loc["a"]
        self.assertEqual(excluded, 0)
        self.assertEqual(first.close_american, -250)
        self.assertEqual(first.open_pick_american, 150)
        self.assertEqual(first.same_choice_open_american, -200)
        self.assertFalse(first.close_win)
        self.assertTrue(first.open_win)

    def test_price_shift_crosses_even_money_without_jump(self):
        np.testing.assert_allclose(improve_prices([-150], 0), [1 + 100 / 150])
        np.testing.assert_allclose(improve_prices([-150], 50), [2])
        np.testing.assert_allclose(improve_prices([-150], 100), [2.5])
        np.testing.assert_allclose(decimal_to_american([2.5]), [150])
        np.testing.assert_allclose(american_to_decimal([-200, 100, 200]), [1.5, 2, 3])

    def test_solve_matches_known_target_and_curve(self):
        # One win/one loss at +100 returns 0%; at +200 returns 50%.
        summary, curve = solve_bin(example_group(), max_shift=200, step=25)
        self.assertEqual(summary["status"], "matched")
        self.assertAlmostEqual(summary["required_adjustment"], 100)
        self.assertAlmostEqual(summary["matched_roi_pct"], 50)
        self.assertAlmostEqual(summary["mean_decimal_increase"], 1)
        self.assertTrue((curve.adjusted_roi_pct.diff().dropna() >= 0).all())
        self.assertAlmostEqual(curve.loc[curve.adjustment.eq(100), "gap_closed_pct"].iloc[0], 100)

    def test_no_improvement_needed_and_unreachable_cases(self):
        summary, _ = solve_bin(example_group(target_prices=(1.5, 1.5)))
        self.assertEqual(summary["required_adjustment"], 0)
        self.assertEqual(summary["status"], "already_at_or_above_open")
        summary, _ = solve_bin(example_group(wins=(False, False)))
        self.assertEqual(summary["status"], "unreachable_no_wins")
        self.assertTrue(np.isnan(summary["required_adjustment"]))
        summary, _ = solve_bin(example_group(), max_shift=50)
        self.assertEqual(summary["status"], "unreachable_within_limit")


if __name__ == "__main__":
    unittest.main()
