import unittest
import numpy as np
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import (
    interpolate_prices, fair_probabilities, expected_bets,
)


class EventSensitivityTests(unittest.TestCase):
    def test_interpolation_and_devig(self):
        close = np.array([[1.5, 3.0]])
        opening = np.array([[1.8, 2.1]])
        np.testing.assert_allclose(interpolate_prices(close, opening, 0), close)
        np.testing.assert_allclose(interpolate_prices(close, opening, 1), opening)
        np.testing.assert_allclose(interpolate_prices(close, opening, .5), [[1.65, 2.55]])
        fair = fair_probabilities(close)
        np.testing.assert_allclose(fair.sum(axis=1), 1)
        np.testing.assert_allclose(fair, [[2/3, 1/3]], atol=1e-10)

    def test_dynamic_stakes_and_expected_return(self):
        a = expected_bets(np.array([.7]), np.array([[2., 2.]]), np.array([[.5, .5]]), .4, 250)
        b = expected_bets(np.array([.7]), np.array([[1.6, 2.8]]), np.array([[.65, .35]]), .4, 250)
        self.assertNotAlmostEqual(a.fstar.iloc[0], b.fstar.iloc[0])
        self.assertAlmostEqual(a.expected_bankroll_return_pct.iloc[0], 100 * a.fstar.iloc[0] * (.7 * 2 - 1))
        no_bet = expected_bets(np.array([.6]), np.array([[1.5, 3.]]), np.array([[.5, .5]]), .4, 250)
        self.assertEqual(no_bet.fstar.iloc[0], 0)

    def test_pick_changes_with_model_probability(self):
        bets = expected_bets(np.array([.4]), np.array([[2., 2.]]), np.array([[.5, .5]]), .4, 250)
        self.assertEqual(bets.pred_winner.iloc[0], 0)
        self.assertAlmostEqual(bets.choice_probability.iloc[0], .6)


if __name__ == "__main__":
    unittest.main()
