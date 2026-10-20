import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.Models.Search.exponential_decay import MoneylineDecaySearch, decay_weights


class SearchDecayTests(unittest.TestCase):
    def test_return_calculator_settles_blended_allocations(self):
        from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator
        search = self.make_search()
        with patch.object(Calculator, 'calculate_decay_portfolio', wraps=Calculator.calculate_decay_portfolio) as settle:
            result = search.evaluate(k_down=2, k_up=3)
        settle.assert_called_once()
        self.assertTrue((result['events'].exposure <= 1 + 1e-12).all())
        self.assertTrue((result['events'].event_return >= -1 - 1e-12).all())

    def test_allocation_settlement_preserves_uncapped_losses_and_no_bet_events(self):
        search = self.make_search()
        events = search.bets_engine.calculate_allocated_returns([.7, .7, 0, 0], [-1, -1, 2, -1])
        np.testing.assert_allclose(events.event_return, [-1.4, 0])
        np.testing.assert_allclose(events.exposure, [1.4, 0])

    def make_search(self, **kwargs):
        frame = pd.DataFrame(dict(
            date=['2025-01-01'] * 2 + ['2025-01-08'] * 2,
            fighter_red=list('abcd'), fighter_blue=list('efgh'), winner=[1, 0, 1, 1],
            proba_red=[.7] * 4, prediction_se=[.01] * 4,
            dec_close1_red=[1.8, 2.2, 1.9, 2.3], dec_close1_blue=[2.2] * 4,
            dec_fair_close1_red=[2.] * 4, dec_fair_close1_blue=[2.] * 4,
        ))
        reference = frame[['date', 'fighter_red', 'fighter_blue']].copy()
        reference['dec_open_red'] = 2.
        reference['dec_open_blue'] = 2.
        params = dict(mdd=.4, N=250, z=.5, min_moneyline=None)
        return MoneylineDecaySearch(frame, reference, params, {**params, 'mdd': .2}, k=10, **kwargs)

    def test_direction_zero_and_blended_returns(self):
        np.testing.assert_allclose(decay_weights([2, 2, 2], [1.5, 3, 2], 2, 4),
                                   [np.exp(-2 * abs(np.log(.75))), np.exp(-4 * np.log(1.5)), 1])
        search = self.make_search()
        np.testing.assert_allclose(search.event_returns().event_return,
                                   search.event_returns(policy='opening').event_return)
        w = decay_weights(search.opening_prices, search.current_prices, 2, 3)
        manual = np.bincount(search.event_ids, weights=(
            w * search.opening_stakes + (1 - w) * search.closing_stakes) * search.unit_profit)
        np.testing.assert_allclose(search.event_returns(2, 3).event_return, manual)

    def test_saved_paths_and_search_results(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'paths'
            search = self.make_search(bootstrap_mode='save', bootstrap_dir=path)
            replay = self.make_search(bootstrap_mode='load', bootstrap_dir=path)
            np.testing.assert_array_equal(search.path_indices, replay.path_indices)
            before = search.path_indices.copy()
            result = search.run(n_trials=3, output_dir=Path(root) / 'results')
            self.assertEqual(set(result['best_params']), {'k_down', 'k_up'})
            np.testing.assert_array_equal(search.path_indices, before)
            self.assertFalse(hasattr(search, 'simulate_kelly'))
            self.assertTrue((Path(root) / 'results/search_summary.json').is_file())

    def test_final_cap_prevents_overstaking_after_blend(self):
        from ufc_betting.Models.Search.exponential_decay import evaluate_decay_portfolio
        search = self.make_search()
        evaluate_decay_portfolio(search, dict(k_down=0, k_up=0))
        # Raw stake overrides model a highly exposed, all-losing portfolio.
        for cache in search._raw_policy_caches:
            cache[:, 0] = .7
            cache[:, 2] = -1.
        result = search.evaluate()
        np.testing.assert_allclose(result['events'].exposure, 1.)
        np.testing.assert_allclose(result['simulation']['bankroll'], np.tile([500, 0, 0], (10, 1)), atol=1e-10)

    def test_stakes_match_legacy_moneyline_simulation(self):
        from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
        search = self.make_search()
        for name, expected in [('opening', search.opening_stakes), ('closing', search.closing_stakes)]:
            params = search.metadata[f'{name}_params']
            result, _ = simulate_kelly(
                search.frame.assign(open_red=100., open_blue=100.), prob_cols=['proba_blue', 'proba_red'],
                fair_decimal_cols=['dec_fair_close1_blue', 'dec_fair_close1_red'],
                real_decimal_cols=['dec_close1_blue', 'dec_close1_red'],
                pred_winner_col='pred_winner', prediction_se_col='prediction_se',
                max_drawdown=params['mdd'], N=params['N'], z=params['z'],
                adjust_mdd_by_edge=False, calc_parlay=False,
            )
            np.testing.assert_allclose(result.choice_fstar, expected)


if __name__ == '__main__':
    unittest.main()
