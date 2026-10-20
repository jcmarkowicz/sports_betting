import unittest

import numpy as np

from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator
from ufc_betting.BettingStrategy.betting_params import BettingParams


class PerBetRiskTests(unittest.TestCase):
    def test_unrelated_uncertainty_does_not_change_other_stake(self):
        stakes = np.array([.2, .3])
        first = Calculator.adjusted_stakes(stakes, [.2, .1], [.1, .1])
        second = Calculator.adjusted_stakes(stakes, [.2, .1], [.1, 0.])
        np.testing.assert_allclose(first, [.1, .3])
        np.testing.assert_allclose(second, [.1, 0.])

    def test_cap_is_applied_after_independent_adjustments(self):
        np.testing.assert_allclose(
            Calculator.adjusted_stakes([2., 2.], [.2, .2], [.1, .2]),
            [1 / 3, 2 / 3])

    def test_disabled_risk_and_zero_edge_are_finite(self):
        np.testing.assert_allclose(Calculator.adjusted_stakes([.2, .3], [0, .1], [0, 0]), [0, 0])
        np.testing.assert_allclose(
            Calculator.adjusted_stakes([.2, .3], [0, .1], [0, 0], enabled=False), [.2, .3])

    def test_ticket_uses_its_own_legs_only(self):
        selected = np.array([[[.01, .3, 1., .2, .1, .7, 2., 1],
                              [.01, .2, 1., .2, .2, .7, 2., 1],
                              [.01, .1, 1., .2, .2, .7, 2., 1]]])
        params = BettingParams(z=.5)
        first = Calculator.scenario_allocations(selected, [1, 1, 1], params, True)
        selected[0, 2, 4] = 0
        second = Calculator.scenario_allocations(selected, [1, 1, 1], params, True)
        np.testing.assert_allclose(first['moneyline_stakes'][0, :2], second['moneyline_stakes'][0, :2])
        np.testing.assert_allclose(first['parlay_risk_multiplier'], [.75])
        np.testing.assert_allclose(first['parlay_stakes'] * first['multiplier'],
                                   second['parlay_stakes'] * second['multiplier'])
        self.assertEqual(second['moneyline_stakes'][0, 2], 0.)


if __name__ == '__main__':
    unittest.main()
