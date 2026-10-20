import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import test_joint_kelly_decay_search as fixtures

from Notebooks.Search.run_search import run_search
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import settle_bankroll_paths
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.Models.Search.frozen_betting import path_mean_event_returns


class EventMeanStdTests(unittest.TestCase):
    def test_average_events_not_terminal_return_or_bet_sharpe(self):
        returns = np.array([[.1, -.2, .3], [.01, .02, .03]])
        for initial in (100., 500.):
            sim = settle_bankroll_paths(returns, initial)
            np.testing.assert_allclose(path_mean_event_returns(sim['bankroll']), returns.mean(axis=1))
        self.assertFalse(np.allclose(returns.mean(axis=1), sim['terminal_returns']))

    def test_all_scheduled_events_count_including_ruin_and_no_bets(self):
        sim = settle_bankroll_paths([[0., 0., 0.], [-1., .5, .5], [.1, 0., .2]])
        np.testing.assert_allclose(path_mean_event_returns(sim['bankroll']), [0., -1./3, .1])
        np.testing.assert_allclose(path_mean_event_returns([[100., 110.]]), [.1])
        for invalid in ([[100., np.nan]], [[0., 1.]], [[100.]], []):
            with self.assertRaises(ValueError):
                path_mean_event_returns(invalid)

    def test_bootstrap_objective_uses_sample_sd_of_path_means(self):
        frame = fixtures.JointKellyDecayTests().frame()
        with tempfile.TemporaryDirectory() as tmp:
            for market in ('open', 'close1'):
                extra = {} if market == 'open' else dict(use_decay=True, opening_params=BettingParams())
                root = Path(tmp) / market
                result = optimize_kelly_params(validation_data={market: frame}, k=7, n_trials=2,
                    risk_penalty=.7, score_type='event_mean_std', bootstrap_mode='save',
                    bootstrap_dir=root, **extra)
                indices = np.load(root / market / 'paths.npy')
                means = result['event_results'].event_return.to_numpy()[indices].mean(axis=1)
                np.testing.assert_allclose(result['path_mean_event_returns'], means)
                expected = means.mean() - .7 * means.std(ddof=1)
                self.assertAlmostEqual(result['best_value'], expected)
                study = result['studies']['full_validation']
                self.assertEqual(study.user_attrs['score_type'], 'event_mean_std')
                self.assertAlmostEqual(study.best_trial.user_attrs['std_path_mean_event_return'], means.std(ddof=1))
                self.assertAlmostEqual(result['objective_breakdown'].objective_pct.iloc[0], 100 * expected)
                self.assertNotIn('mean_event_sharpe', result['objective_breakdown'])

    def test_saved_paths_zero_spread_and_pickle(self):
        frame = fixtures.JointKellyDecayTests().frame().assign(sampled_event_id=np.repeat([0, 1, 2], 2))
        frame['source_date'] = frame.date
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame.to_csv(root / 'predictions.csv', index=False)
            pd.to_pickle([frame, frame], root / 'paths.pkl')
            result = run_search('open', paths_path=root / 'paths.pkl',
                predictions_path=root / 'predictions.csv', params_dir=root / 'params',
                n_paths=2, n_trials=2, score_type='event_mean_std', risk_penalty=100.)
            # Identical paths have zero SD: even a large penalty leaves their mean.
            means = result['sampled_path_results'].mean_event_return.to_numpy()
            self.assertAlmostEqual(result['best_value'], means.mean())
            self.assertEqual(means.std(ddof=1), 0.)
            saved = pd.read_pickle(root / 'params/search_results/open_search_results.pkl')
            self.assertEqual(saved['score_type'], 'event_mean_std')
            np.testing.assert_allclose(saved['path_mean_event_returns'], means)


if __name__ == '__main__':
    unittest.main()
