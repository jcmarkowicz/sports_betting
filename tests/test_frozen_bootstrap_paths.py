import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_betting_returns import BettingReturnCalculatorTests
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.Models.Search.frozen_betting import FrozenBettingSearch
from ufc_betting.Models.Search.prediction_input import read_frozen_predictions


class FrozenBootstrapPathTests(unittest.TestCase):
    def test_numbered_schedule_metadata_rejects_changed_types(self):
        frame = self.predictions()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'numbered'
            kwargs = dict(k=3, sample_with_replacement=False, bootstrap_dir=directory)
            saved = FrozenBettingSearch(frame, bootstrap_mode='save', **kwargs)
            loaded = FrozenBettingSearch(frame, bootstrap_mode='load', **kwargs)
            np.testing.assert_array_equal(saved.path_indices, loaded.path_indices)
            frame['event_type'] = 'numbered'
            with self.assertRaisesRegex(ValueError, 'match'):
                FrozenBettingSearch(frame, bootstrap_mode='load', **kwargs)

    def predictions(self):
        frame = BettingReturnCalculatorTests().predictions()
        frame['fighter_red'] = [f'red_{i}' for i in frame.index]
        frame['fighter_blue'] = [f'blue_{i}' for i in frame.index]
        frame['event_type'] = 'fight_night'
        return frame

    def test_save_load_and_metadata(self):
        frame = self.predictions()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'saved'
            saved = FrozenBettingSearch(frame, k=5, bootstrap_mode='save', bootstrap_dir=directory)
            with patch('ufc_betting.Models.Search.frozen_betting.sample_schedule',
                       side_effect=AssertionError('Loading must not resample')):
                loaded = FrozenBettingSearch(frame, k=3, seed=999,
                                             bootstrap_mode='load', bootstrap_dir=directory)
            np.testing.assert_array_equal(loaded.path_indices, saved.path_indices[:3])
            metadata = json.loads((directory / 'metadata.json').read_text())
            self.assertEqual(metadata['n_samples'], 5)
            self.assertEqual(metadata['n_fights'], len(frame))
            self.assertEqual(metadata['n_events'], frame.date.nunique())
            self.assertEqual(len(metadata['fights']), len(frame))
            self.assertTrue(metadata['replace'])
            with self.assertRaises(FileExistsError):
                FrozenBettingSearch(frame, k=2, bootstrap_mode='save', bootstrap_dir=directory)
            with self.assertRaisesRegex(ValueError, 'requested'):
                FrozenBettingSearch(frame, k=6, bootstrap_mode='load', bootstrap_dir=directory)
            for column, value in [('fighter_red', 'different'), ('proba_red', .123),
                                  ('prediction_se', .123), ('dec_open_red', 5.)]:
                changed = frame.copy()
                changed.loc[0, column] = value
                with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'match'):
                    FrozenBettingSearch(changed, k=3, bootstrap_mode='load', bootstrap_dir=directory)
            with self.assertRaisesRegex(ValueError, 'match'):
                FrozenBettingSearch(frame, k=3, sample_with_replacement=False,
                                    bootstrap_mode='load', bootstrap_dir=directory)
            with (directory / 'paths.npy').open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                FrozenBettingSearch(frame, k=3, bootstrap_mode='load', bootstrap_dir=directory)

    def test_choice_probabilities_and_standard_errors(self):
        data = pd.DataFrame({'choice_proba': [.8, .7], 'choice_fighter_bool': [0, 1],
                             'se': [.02, .03]})
        result = read_frozen_predictions(data)
        np.testing.assert_allclose(result.proba_red, [.2, .7])
        np.testing.assert_allclose(result.risk_red_proba, [.2, .7])
        np.testing.assert_allclose(result.prediction_se, data.se)
        self.assertNotIn('proba_red', data)
        with self.assertRaisesRegex(ValueError, 'Choice probabilities'):
            read_frozen_predictions(data.drop(columns='choice_fighter_bool'))

    def test_all_markets_use_supplied_data_and_replay_paths(self):
        frame = self.predictions()
        with tempfile.TemporaryDirectory() as root:
            csv = Path(root) / 'validation.csv'
            frame.to_csv(csv, index=False)
            inputs = {'open': frame, 'close1': csv, 'close2': frame}
            kwargs = dict(n_trials=1, k=3,
                          bootstrap_dir=Path(root) / 'paths',
                          risk_penalty={'open': .25, 'close1': .4, 'close2': .4})
            with patch('ufc_betting.Models.Search.logit_search.load_validation_predictions',
                       side_effect=AssertionError('Must use supplied predictions')), patch(
                       'ufc_betting.Models.Search.logit_search.discover_alpha_history',
                       side_effect=AssertionError('Must not load models')):
                saved = {market: optimize_kelly_params(validation_data={market: data}, **kwargs, bootstrap_mode='save')
                         for market, data in inputs.items()}
                loaded = {market: optimize_kelly_params(validation_data={market: data}, **kwargs, bootstrap_mode='load', parallel=False)
                          for market, data in inputs.items()}
            for market in inputs:
                self.assertTrue((Path(saved[market]['bootstrap_dir']) / 'paths.npy').is_file())
                self.assertEqual(saved[market]['best_params'], loaded[market]['best_params'])
                self.assertAlmostEqual(saved[market]['best_value'], loaded[market]['best_value'])
                self.assertEqual(len(saved[market]['betting_data']), len(frame))
                np.testing.assert_allclose(saved[market]['betting_data'].proba_red, frame.proba_red)

    def test_supplied_choice_probability_is_not_replaced_by_risk_probability(self):
        data = pd.DataFrame({'choice_proba': [.8, .7], 'choice_fighter_bool': [0, 1],
                             'proba_red': [.5, .5], 'risk_red_proba': [.3, .6],
                             'prediction_se': [.02, .03]})
        result = read_frozen_predictions(data)
        np.testing.assert_allclose(result.proba_red, [.2, .7])
        np.testing.assert_allclose(result.risk_red_proba, [.3, .6])
        canonical = read_frozen_predictions(data.drop(columns=['choice_proba', 'choice_fighter_bool']))
        np.testing.assert_allclose(canonical.proba_red, [.5, .5])


if __name__ == '__main__':
    unittest.main()
