import unittest

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import (
    settle_bankroll_paths, simulate_bankroll_paths,
)


class BankrollSettlementTests(unittest.TestCase):
    def test_bootstrap_retains_crossing_loss_for_every_remaining_event(self):
        events = pd.DataFrame({
            'event_return': [.5, -1.2, 10., -2., -1.],
            'moneyline_return': [.3, -.7, 6., -1., -.6],
            'parlay_return': [.2, -.5, 4., -1., -.4],
        })
        schedules = np.array([
            [0, 1, 2, 3, 1],  # Gain, cross -100%, then repeated scheduled events.
            [0, 4, 2, 3, 0],  # Exact -100% remains frozen too.
            [0, 0, 0, 0, 0],  # Independent solvent path continues compounding.
        ])
        result = simulate_bankroll_paths(events, schedules, initial_bankroll=100)
        np.testing.assert_allclose(result['bankroll'][0], [100, 150, -30, -30, -30, -30])
        np.testing.assert_allclose(result['bankroll'][1], [100, 150, 0, 0, 0, 0])
        np.testing.assert_allclose(result['bankroll'][2], 100 * 1.5 ** np.arange(6))
        np.testing.assert_allclose(result['terminal_returns'], [-1.3, -1., 1.5 ** 5 - 1])
        self.assertEqual(result['bankroll'].shape, (3, schedules.shape[1] + 1))
        combined = result['moneyline_cumulative_returns'] + result['parlay_cumulative_returns']
        np.testing.assert_allclose(combined, result['bankroll'][:, 1:] / 100 - 1)

    def test_batches_share_capital_and_include_partial_batch(self):
        families = {'first': np.array([[.1, .2, .3]]),
                    'second': np.array([[-.05, -.05, -.1]])}
        result = settle_bankroll_paths(
            families['first'] + families['second'], 100,
            family_returns=families, batch_size=2,
        )
        np.testing.assert_allclose(result['bankroll'], [[100, 120, 144]])
        np.testing.assert_allclose(result['first_profits'], [[30, 36]])
        np.testing.assert_allclose(result['second_profits'], [[-10, -12]])

    def test_batch_bankruptcy_preserves_negative_balance(self):
        result = settle_bankroll_paths(
            [[-.6, -.6, 10], [.1, .1, .5]], 100, batch_size=2,
            family_returns={'bets': [[-.6, -.6, 10], [.1, .1, .5]]},
        )
        np.testing.assert_allclose(result['bankroll'], [[100, -20, -20], [100, 120, 180]])
        np.testing.assert_allclose(result['bets_profits'], [[-120, 0], [20, 60]])

    def test_rejects_misaligned_or_inconsistent_families(self):
        for families in ({'bets': [[.1]]}, {'bets': [[.1, .5]]}):
            with self.subTest(families=families), self.assertRaises(ValueError):
                settle_bankroll_paths([[.1, .2]], family_returns=families)


if __name__ == '__main__':
    unittest.main()
