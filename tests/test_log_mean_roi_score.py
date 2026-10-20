import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import test_joint_kelly_decay_search as fixtures

from Notebooks.Search.run_search import run_search
from ufc_betting.Models.Search.frozen_betting import log_mean_roi_score
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.BettingStrategy.betting_params import BettingParams


class LogMeanROITests(unittest.TestCase):
    def test_log_after_averaging_and_negative_penalty(self):
        self.assertAlmostEqual(log_mean_roi_score([-.1, .3]), np.log(.1))
        self.assertAlmostEqual(log_mean_roi_score([.1, .9]), np.log(.5))
        self.assertNotAlmostEqual(log_mean_roi_score([.1, .9]), np.log([.1, .9]).mean())
        self.assertEqual(log_mean_roi_score([-.3, .1], 42.), -42.)
        self.assertEqual(log_mean_roi_score([-.1, .1]), -np.inf)
        for penalty in (-1., 0., np.inf, np.nan, True):
            with self.assertRaises(ValueError):
                log_mean_roi_score([.1, .2], penalty)

    def test_open_and_decay_objectives(self):
        frame = fixtures.JointKellyDecayTests().frame()
        for market in ('open', 'close1'):
            extra = {} if market == 'open' else dict(use_decay=True, opening_params=BettingParams())
            result = optimize_kelly_params(validation_data={market: frame}, k=7, n_trials=2,
                score_type='log_mean_roi', negative_mean_penalty=42., **extra)
            self.assertAlmostEqual(result['best_value'], log_mean_roi_score(result['path_mean_event_rois'], 42.))
            self.assertEqual(result['negative_mean_penalty'], 42.)
            self.assertEqual(result['studies']['full_validation'].user_attrs['negative_mean_penalty'], 42.)
            self.assertIn('log_mean_roi', result['objective_breakdown'])

    def test_run_search_and_saved_metadata(self):
        frame = fixtures.JointKellyDecayTests().frame().assign(sampled_event_id=np.repeat([0, 1, 2], 2))
        frame['source_date'] = frame.date
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame.to_csv(root / 'predictions.csv', index=False)
            pd.to_pickle([frame, frame], root / 'paths.pkl')
            result = run_search('open', paths_path=root / 'paths.pkl', predictions_path=root / 'predictions.csv',
                params_dir=root / 'params', n_paths=2, n_trials=2, score_type='log_mean_roi', negative_mean_penalty=77.)
            self.assertAlmostEqual(result['best_value'], log_mean_roi_score(result['path_mean_event_rois'], 77.))
            saved = pd.read_pickle(root / 'params/search_results/open_search_results.pkl')
            self.assertEqual(saved['negative_mean_penalty'], 77.)
            self.assertEqual(saved['score_type'], 'log_mean_roi')
