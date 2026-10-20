import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import test_joint_kelly_decay_search as fixtures

from Notebooks.Search.run_search import run_search
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.Models.Search.frozen_betting import (
    path_mean_sharpe_score, event_roi_statistics, validate_score_settings,
)
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import settle_bankroll_paths


class PathMeanSharpeTests(unittest.TestCase):
    def test_ratio_of_path_means_not_average_of_ratios(self):
        returns = np.array([[.1, -.2, .3], [.01, .02, .03], [-.1, .02, .05]])
        means = returns.mean(axis=1)
        sim = settle_bankroll_paths(returns)
        actual = path_mean_sharpe_score(event_roi_statistics(returns, np.ones_like(returns), sim['bankroll'])['path_mean_event_rois'])
        self.assertAlmostEqual(actual, means.mean() / means.std(ddof=1))
        self.assertNotAlmostEqual(actual, np.mean(returns.mean(axis=1) / returns.std(axis=1, ddof=1)))
        self.assertNotAlmostEqual(actual, means.mean() / (means.std(ddof=1) / np.sqrt(len(means))))
        self.assertAlmostEqual(path_mean_sharpe_score(10 * means), actual)

    def test_roi_equal_weights_excludes_no_stakes_and_after_ruin(self):
        returns = np.array([[.02, .10, 0.], [-.5, -.5, .1], [0., 0., 0.]])
        stakes = np.array([[.1, .2, 0.], [.5, .5, .1], [0., 0., 0.]])
        bankroll = np.array([[100., 102., 112.2, 112.2], [100., 0., 0., 0.], [100., 100., 100., 100.]])
        stats = event_roi_statistics(returns, stakes, bankroll)
        np.testing.assert_allclose(stats['path_mean_event_rois'], [.35, -1., np.nan], equal_nan=True)
        np.testing.assert_array_equal(stats['betting_event_counts'], [2, 1, 0])
        self.assertNotAlmostEqual(stats['path_mean_event_rois'][0], .12/.3)

    def test_no_cap_and_zero_variance_policy(self):
        self.assertGreater(path_mean_sharpe_score([.100, .101, .102]), 5.)
        self.assertEqual(path_mean_sharpe_score([0., 0.]), 0.)
        for values in ([.1, .1], [-.1, -.1], [.1], [np.nan, 0.]):
            with self.assertRaises(ValueError):
                path_mean_sharpe_score(values)
        with self.assertRaises(ValueError):
            validate_score_settings('event_sharpe')

    def test_open_and_closing_bootstrap_scores(self):
        frame = fixtures.JointKellyDecayTests().frame()
        for market in ('open', 'close1'):
            extra = {} if market == 'open' else dict(use_decay=True, opening_params=BettingParams())
            with tempfile.TemporaryDirectory() as tmp:
                result = optimize_kelly_params(validation_data={market: frame}, k=7, n_trials=2,
                    score_type='path_mean_sharpe', risk_penalty=100.,
                    bootstrap_mode='save', bootstrap_dir=tmp, **extra)
                indices = np.load(Path(tmp) / market / 'paths.npy')
                means = np.nanmean(result['event_rois'], axis=1)
                np.testing.assert_allclose(result['path_mean_event_rois'], means)
                if market == 'close1':
                    events = result['event_results']
                    roi = events.event_return.to_numpy() / events.exposure.to_numpy()
                    np.testing.assert_allclose(means, roi[indices].mean(axis=1))
                self.assertAlmostEqual(result['best_value'], means.mean() / means.std(ddof=1))
                self.assertAlmostEqual(result['objective_breakdown'].path_mean_sharpe.iloc[0], result['best_value'])
                self.assertNotIn('objective_pct', result['objective_breakdown'])

    def test_saved_paths_and_pickle(self):
        frame = fixtures.JointKellyDecayTests().frame().assign(sampled_event_id=np.repeat([0, 1, 2], 2))
        frame['source_date'] = frame.date
        # A different event mix produces different path-average returns.
        second = pd.concat([frame.iloc[:2], frame.iloc[:2], frame.iloc[4:]], ignore_index=True)
        second['date'] = np.repeat(frame.date.unique(), 2)
        second['sampled_event_id'] = np.repeat([0, 1, 2], 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame.to_csv(root / 'predictions.csv', index=False)
            pd.to_pickle([frame, second], root / 'paths.pkl')
            result = run_search('open', paths_path=root / 'paths.pkl',
                predictions_path=root / 'predictions.csv', params_dir=root / 'params',
                n_paths=2, n_trials=2, score_type='path_mean_sharpe')
            means = np.nanmean(result['event_rois'], axis=1)
            self.assertAlmostEqual(result['best_value'], means.mean() / means.std(ddof=1))
            saved = pd.read_pickle(root / 'params/search_results/open_search_results.pkl')
            self.assertEqual(saved['score_type'], 'path_mean_sharpe')
            np.testing.assert_allclose(saved['sampled_path_results'].mean_event_roi, means)
            self.assertNotIn('event_sharpe_cap', saved)


if __name__ == '__main__':
    unittest.main()
