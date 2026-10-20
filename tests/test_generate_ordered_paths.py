import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_joint_kelly_decay_search import JointKellyDecayTests
from ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling import generate_ordered_paths
from ufc_betting.BettingStrategy.MonteCarlo.sampled_paths import load_sampled_paths, predict_intermediate_paths
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.Models.LogisticRegression.odds_input import replace_opening_odds
from ufc_betting.BettingStrategy.betting_params import BettingParams


class GeneratedPathsTests(unittest.TestCase):
    def frame(self):
        return JointKellyDecayTests().frame().assign(bout_position=[1, 2] * 3, event_name='UFC 300')

    def test_recorded_generation_roundtrip_and_reproducibility(self):
        source = self.frame()
        original = source.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'nested' / 'paths.pkl'
            paths = generate_ordered_paths(source, n_paths=3, output_path=file, progress_every=0)
            loaded = load_sampled_paths(file)
            repeated = generate_ordered_paths(source, n_paths=3, progress_every=0)
            for a, b, c in zip(paths, loaded, repeated):
                pd.testing.assert_frame_equal(a, b)
                pd.testing.assert_frame_equal(a, c)
                donor = source.iloc[a.source_row_position]
                np.testing.assert_array_equal(a.proba_red, donor.proba_red)
                np.testing.assert_array_equal(a.dec_close1_red, donor.dec_close1_red)
                self.assertEqual(a.sampled_event_id.nunique(), 3)
                self.assertEqual(a.attrs['generation']['odds_type'], 'close1')
        pd.testing.assert_frame_equal(source, original)

    def test_sampled_prices_refresh_predictions_and_feed_both_consumers(self):
        source = self.frame().assign(risk_red_proba=.01, proba_se=.99)
        models = {5: {stage: {'model': object()} for stage in ('close1', 'close2')}}
        def sample(model, opening, closing, count, random_state):
            return [float(closing) + .05 * random_state.random()]
        def predict(frame, prices, predictor, model_type):
            output = replace_opening_odds(frame, 'close1', prices=prices, features=[])
            return output.assign(proba_red=.65, proba_blue=.35, proba_se=.02, pred_winner=1)
        for market in ('close1', 'close2'):
            with tempfile.TemporaryDirectory() as tmp, \
                 patch('ufc_betting.BettingStrategy.MonteCarlo.market_policy.load_predictor', return_value=object()), \
                 patch('ufc_betting.BettingStrategy.MonteCarlo.market_policy.predict_sampled_prices', side_effect=predict) as inference, \
                 patch('ufc_betting.BettingStrategy.MonteCarlo.price_sampling.load_density_models', return_value=models), \
                 patch('ufc_betting.BettingStrategy.MonteCarlo.price_sampling.sample_prices', side_effect=sample):
                file = Path(tmp) / 'paths.pkl'
                paths = generate_ordered_paths(source, n_paths=2, output_path=file, odds_type=market,
                    sample_prices=True, prediction_model=object(), days_out=5, progress_every=0)
                self.assertEqual(inference.call_count, 2)
                for path in paths:
                    self.assertNotIn('prediction_se', path)
                    self.assertNotIn('risk_red_proba', path)
                    self.assertTrue(path.execution_odds_type.eq('intermediate').all())
                    np.testing.assert_allclose(path[f'dec_{market}_red'], path.dec_close1_red)
                    np.testing.assert_allclose(path.recorded_close1_red, source.iloc[path.source_row_position].dec_close1_red)
                for decay in (False, True):
                    searched = optimize_kelly_params(validation_data={market: paths[0]}, sampled_paths=file,
                        n_trials=1, parallel=False, params_dir=Path(tmp)/'params',
                        use_decay=decay, opening_params=BettingParams() if decay else None)
                    result = simulate(sampled_paths=file, odds_type='intermediate', use_decay=decay,
                        kelly_kwargs={'betting_params': searched['betting_params']})
                    np.testing.assert_allclose(result[2], searched['sampled_bankroll_paths'])

    def test_invalid_options_do_not_create_output(self):
        for options in [dict(n_paths=0), dict(n_paths=True),
                        dict(sample_prices=True, odds_type='open', prediction_model=object()),
                        dict(sample_prices=True, odds_type=None, prediction_model=object())]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                file = Path(tmp) / 'paths.pkl'
                with self.assertRaises(ValueError):
                    generate_ordered_paths(self.frame(), output_path=file, **options)
                self.assertFalse(file.exists())

    def test_intermediate_generation_without_model_then_predict_without_resampling(self):
        source = self.frame()
        models = {5: {stage: {'model': object()} for stage in ('close1', 'close2')}}
        with tempfile.TemporaryDirectory() as directory, \
             patch('ufc_betting.BettingStrategy.MonteCarlo.price_sampling.load_density_models', return_value=models), \
             patch('ufc_betting.BettingStrategy.MonteCarlo.price_sampling.sample_prices', return_value=[2.25]), \
             patch('ufc_betting.BettingStrategy.MonteCarlo.market_policy.load_predictor', side_effect=AssertionError('No model at generation')):
            file = Path(directory) / 'intermediate.pkl'
            paths = generate_ordered_paths(source, n_paths=2, output_path=file, odds_type=None,
                                           sample_prices=True, days_out=(5,), progress_every=0)
            saved = load_sampled_paths(file)
            for path in saved:
                self.assertNotIn('proba_red', path)
                self.assertNotIn('proba_se', path)
                self.assertIsNone(path.attrs['generation']['odds_type'])
                np.testing.assert_array_equal(path.dec_intermediate_red, 2.25)
                donor = source.iloc[path.source_row_position]
                for stage in ('open', 'close1', 'close2'):
                    for side in ('red', 'blue'):
                        np.testing.assert_array_equal(path[f'dec_{stage}_{side}'], donor[f'dec_{stage}_{side}'])

        def predict(frame, prices, predictor, model_type):
            np.testing.assert_array_equal(prices, frame[['dec_intermediate_red', 'dec_intermediate_blue']])
            output = replace_opening_odds(frame, 'close1', prices=prices, features=[])
            return output.assign(proba_red=.65, proba_blue=.35, proba_se=.02, pred_winner=1)

        for market in ('close1', 'close2'):
            with patch('ufc_betting.BettingStrategy.MonteCarlo.market_policy.load_predictor', return_value=object()), \
                 patch('ufc_betting.BettingStrategy.MonteCarlo.market_policy.predict_sampled_prices', side_effect=predict), \
                 patch('ufc_betting.BettingStrategy.MonteCarlo.price_sampling.ConditionalPriceSampler.sample', side_effect=AssertionError('No resampling')):
                evaluated = predict_intermediate_paths(paths, prediction_model=object(),
                    prediction_model_type='open', prediction_scalers={}, prediction_encoders={}, search_market=market)
            for raw, result, original in zip(paths, evaluated, saved):
                pd.testing.assert_frame_equal(raw, original)
                np.testing.assert_array_equal(raw.date, result.date)
                np.testing.assert_array_equal(raw.source_row_position, result.source_row_position)
                np.testing.assert_array_equal(raw.dec_intermediate_red, result[f'dec_{market}_red'])
            searched = optimize_kelly_params(validation_data={market: evaluated[0]}, sampled_paths=evaluated,
                                             n_trials=1, parallel=False)
            simulated = simulate(sampled_paths=evaluated, odds_type='intermediate',
                                 kelly_kwargs={'betting_params': searched['betting_params']})
            np.testing.assert_allclose(simulated[2], searched['sampled_bankroll_paths'])


if __name__ == '__main__':
    unittest.main()
