import unittest
import numpy as np
import pandas as pd

from ufc_betting.Models.RiskManagement.parlay_switch_evaluation import (
    select_legs, ticket, apply_rule, evaluate,
)


class ParlaySwitchTests(unittest.TestCase):
    def test_side_selection_and_current_ticket_settlement(self):
        frame = pd.DataFrame(dict(fighter_red=['A', 'C', 'E'], fighter_blue=['B', 'D', 'F'],
                                  winner=[1, 0, 1]))
        p = np.array([.8, .2, .6])
        prices = np.array([[2., 1.5], [1.5, 2.], [1.5, 2.]])
        legs = select_legs(frame, p, prices)
        self.assertEqual(legs, [(0, 1), (1, 0)])
        result = ticket(frame, p, prices, legs)
        self.assertEqual(result['fighters'], 'A | D')
        self.assertAlmostEqual(result['ev'], .8 * .8 * 4 - 1)
        self.assertAlmostEqual(result['realized_return'], result['stake_fraction'] * 3)
        repriced = ticket(frame, p, prices / 1.1, legs)
        self.assertNotEqual(result['decimal_odds'], repriced['decimal_odds'])
        self.assertEqual(result['fighters'], repriced['fighters'])

    def test_rule_uses_expected_gain_not_realized_gain(self):
        events = pd.DataFrame(dict(close1_changed=[True, True, False],
            close1_delta_expected_return=[.03, .01, .04],
            close1_selected_realized_return=[-.1, .5, .9],
            close1_keep_open_realized_return=[.2, -.2, .1]))
        switch, returns = apply_rule(events, 'close1', ('expected_return', .02))
        self.assertEqual(switch.tolist(), [True, False, False])
        np.testing.assert_allclose(returns, [-.1, -.2, .1])

    def test_future_outcomes_do_not_change_first_selected_rule(self):
        count = 32
        events = pd.DataFrame(dict(date=pd.date_range('2024-01-01', periods=count, freq='14D'),
            close1_changed=True, close1_delta_ev=np.linspace(-.1, .4, count),
            close1_delta_expected_return=np.linspace(-.02, .04, count),
            close1_delta_expected_log_growth=np.linspace(-.01, .02, count),
            close1_selected_realized_return=.05, close1_keep_open_realized_return=.02))
        _, before, _, _ = evaluate(events, 'close1')
        future = events.date >= pd.Timestamp(before.test_start.iloc[0])
        changed = events.copy()
        changed.loc[future, 'close1_selected_realized_return'] = -.8
        _, after, _, _ = evaluate(changed, 'close1')
        self.assertEqual(before.iloc[0].to_dict(), after.iloc[0].to_dict())


if __name__ == '__main__':
    unittest.main()
