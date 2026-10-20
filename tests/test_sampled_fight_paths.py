import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_joint_kelly_decay_search import JointKellyDecayTests
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.Models.Search.frozen_betting import FrozenBettingSearch, JointDecayBettingSearch, terminal_return_score
from ufc_betting.BettingStrategy.MonteCarlo.sampled_paths import load_sampled_paths
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate, simulate_bankroll_paths
from ufc_betting.BettingStrategy.betting_params import BettingParams


class SampledFightPathTests(unittest.TestCase):
    def setUp(self):
        self.frame = JointKellyDecayTests().frame()
        self.paths = []
        for order in ([0, 2, 1, 3, 4, 5], [4, 2, 0, 5, 1, 3]):
            path = self.frame.iloc[order].copy().reset_index(drop=True)
            path['source_date'] = path.date
            path['date'] = self.frame.date.to_numpy()
            path['sampled_event_id'] = np.repeat([0, 1, 2], 2)
            path['bout_position'] = [1, 2] * 3
            self.paths.append(path)

    def test_search_objective_and_simulation_match_direct_saved_cards(self):
        for decay in (False, True):
            market = 'close1' if decay else 'open'
            with self.subTest(decay=decay), tempfile.TemporaryDirectory() as tmp:
                filename = Path(tmp) / 'paths.pkl'
                pd.to_pickle(self.paths, filename)
                extra = dict(use_decay=True, opening_params=BettingParams()) if decay else {}
                with patch('ufc_betting.Models.Search.frozen_betting.sample_schedule',
                           side_effect=AssertionError('Saved paths must not be resampled')):
                    result = optimize_kelly_params(validation_data={market: self.frame},
                        sampled_paths=filename, n_trials=1, parallel=False,
                        params_dir=Path(tmp) / 'params', **extra)
                self.assertTrue((Path(tmp) / 'params' / f'{market}.json').exists())
                self.assertEqual(len(result['sampled_path_results']), 2)
                expected = []
                for path in self.paths:
                    search = (JointDecayBettingSearch(path, market, opening_params=BettingParams(), k=2)
                              if decay else FrozenBettingSearch(path, market, k=2))
                    events = search.event_returns(result['trial_params'])
                    expected.append(simulate_bankroll_paths(events, np.arange(3)[None], 500)['bankroll'][0])
                np.testing.assert_allclose(result['sampled_bankroll_paths'], expected)
                self.assertAlmostEqual(result['best_value'], terminal_return_score(np.array(expected)[:, -1]/500-1))
                with patch('ufc_betting.BettingStrategy.MonteCarlo.mc_engine._event_paths',
                           side_effect=AssertionError('No schedules')), patch('statsmodels.api.load',
                           side_effect=AssertionError('No inference model loading')):
                    actual = simulate(sampled_paths=filename, odds_type=market,
                        kelly_kwargs=result['betting_params'].to_simulation_kwargs(),
                        use_decay=decay, return_details=True)
                np.testing.assert_allclose(actual[2], expected, rtol=1e-10)
                self.assertEqual(len(actual[3]), 2)
                for detail, path in zip(actual[3], self.paths):
                    self.assertEqual(detail['inputs'].fighter_red.tolist(), path.fighter_red.tolist())
                    np.testing.assert_allclose(detail['inputs'].proba_red, path.proba_red)
                    self.assertEqual(len(detail['event_returns']), 3)

    def test_shared_closing_policies_and_partial_batch(self):
        policies = {'close1': BettingParams(z=.3), 'close2': BettingParams(z=.4)}
        a = simulate(sampled_paths=self.paths, closing_params=policies, use_decay=False,
                     return_details=True)
        b = simulate(sampled_paths=self.paths, closing_params=policies, use_decay=False,
                     len_temp=2, return_details=True)
        self.assertEqual(len(b[2][0]), 3)
        for left, right in zip(a[3], b[3]):
            pd.testing.assert_frame_equal(left['event_returns'], right['event_returns'])
        self.assertEqual(self.paths[0].columns.tolist(), self.paths[1].columns.tolist())
        self.assertNotIn('execution_odds_type', self.paths[0])

    def test_market_mapping_and_prediction_specific_results(self):
        altered = [path.assign(proba_red=.51) for path in self.paths]
        with patch('ufc_betting.Models.Search.frozen_betting.sample_schedule',
                   side_effect=AssertionError('No resampling')):
            result = {market: optimize_kelly_params(validation_data={market: self.frame},
                sampled_paths={market: samples}, n_trials=1, parallel=True)
                for market, samples in [('open', self.paths), ('close2', altered)]}
        for market, paths in [('open', self.paths), ('close2', altered)]:
            actual = simulate(sampled_paths=paths, odds_type=market,
                              kelly_kwargs={'betting_params': result[market]['betting_params']}, use_decay=False)
            np.testing.assert_allclose(actual[2], result[market]['sampled_bankroll_paths'])

    def test_saved_intermediate_predictions_do_not_load_kdes(self):
        paths = [path.assign(execution_odds_type='intermediate',
                            recorded_close1_red=path.dec_close1_red,
                            recorded_close1_blue=path.dec_close1_blue,
                            recorded_close2_red=path.dec_close2_red,
                            recorded_close2_blue=path.dec_close2_blue) for path in self.paths]
        policies = dict(close1=BettingParams(), close2=BettingParams())
        with patch('ufc_betting.BettingStrategy.MonteCarlo.price_sampling.load_density_models',
                   side_effect=AssertionError('No new prices')):
            result = simulate(sampled_paths=paths, odds_type='intermediate',
                              closing_params=policies, use_decay=False, return_details=True)
        np.testing.assert_allclose(result[3][0]['inputs'].proba_red, paths[0].proba_red)

    def test_boundary_validation_and_conflicting_inputs(self):
        bad = self.paths[0].copy()
        bad.loc[0, 'sampled_event_id'] = 2
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            load_sampled_paths([bad])
        with self.assertRaisesRegex(ValueError, 'replaces fights'):
            simulate(self.frame, sampled_paths=self.paths)
        with self.assertRaisesRegex(ValueError, 'intermediate'):
            simulate(sampled_paths=self.paths, odds_type='intermediate')
        with self.assertRaisesRegex(ValueError, 'replaces bootstrap'):
            optimize_kelly_params(validation_data={'open': self.frame}, sampled_paths=self.paths,
                                  bootstrap_mode='load', bootstrap_dir='unused')


if __name__ == '__main__':
    unittest.main()
