import unittest

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator
from ufc_betting.BettingStrategy.betting_params import BettingParams


class VoidReturnsTests(unittest.TestCase):
    def setUp(self):
        self.params = BettingParams(z=.5)
        self.policy = self.params.to_policy()
        self.odds = pd.DataFrame([[2., 2.]] * 3)
        self.calculator = Calculator(self.odds, self.odds, ['2026-01-01'] * 3)

    def inputs(self, winners):
        return dict(red_proba=np.array([.8, .7, .6]),
                    risk_red_proba=np.array([.8, .7, .6]), se=np.full(3, .02),
                    winner_fold=np.array(winners), val_idx=np.arange(3), param=self.policy)

    def test_void_refunds_without_changing_allocations(self):
        c = self.calculator
        baseline = self.inputs([1, 1, 1])
        for winners in ([2, 1, 1], [2, 0, 1], [2, 2, 2]):
            with self.subTest(winners=winners):
                args = self.inputs(winners)
                np.testing.assert_allclose(
                    c.calculate_moneyline_stakes(**args, include_parlay=True),
                    c.calculate_moneyline_stakes(**baseline, include_parlay=True))
                ticket = c.calculate_parlay_allocations(**args)
                original = c.calculate_parlay_allocations(**baseline)
                np.testing.assert_allclose(ticket.fstar, original.fstar)
                self.assertEqual(ticket.selected_ids.tolist(), original.selected_ids.tolist())
                self.assertEqual(ticket.pct_return.iloc[0], 0.)
                stakes = c.calculate_moneyline_stakes(**args, include_parlay=True)
                unit = np.where(np.array(winners) == 2, 0, np.where(np.array(winners) == 1, 1, -1))
                events = c.calculate_fight_returns(**args)
                self.assertAlmostEqual(events.event_return.iloc[0], float(stakes @ unit))

    def test_unselected_void_does_not_void_ticket(self):
        c = self.calculator
        ticket = c.calculate_parlay_allocations(**self.inputs([1, 1, 2]))
        self.assertGreater(ticket.pct_return.iloc[0], 0)
        lost = c.calculate_parlay_allocations(**self.inputs([1, 0, 1]))
        self.assertAlmostEqual(lost.pct_return.iloc[0], -lost.fstar.iloc[0])

    def test_scenario_void_matches_fixed_policy(self):
        frame = pd.DataFrame(dict(pred_winner=[1]*3, winner=[2,0,1],
                                  proba_red=[.8,.7,.6], proba_blue=[.2,.3,.4],
                                  prediction_se=[.02]*3))
        for color in ['red','blue']:
            frame[f'dec_close1_{color}'] = 2.
            frame[f'dec_fair_close1_{color}'] = 2.
        cache = Calculator.calculate_scenario_cache(frame, self.params)
        self.assertEqual(cache[0,2], 0.)
        result = Calculator.calculate_scenario_returns(cache[None], frame.winner, self.params, 'combined')
        expected = self.calculator.calculate_fight_returns(**self.inputs([2,0,1]))
        self.assertAlmostEqual(result[1][0], expected.event_return.iloc[0])


if __name__ == '__main__':
    unittest.main()
