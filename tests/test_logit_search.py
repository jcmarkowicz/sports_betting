import unittest
from unittest.mock import patch

import pandas as pd

from ufc_betting.Models.Search import logit_search


class LogitSearchTests(unittest.TestCase):
    def test_prepares_all_markets_then_calls_frozen_optimizer(self):
        history = pd.DataFrame({'alpha': [.5]})
        prepared = {market: pd.DataFrame({'market': [market]})
                    for market in ('open', 'close1', 'close2')}
        for parallel in (False, True):
            with self.subTest(parallel=parallel), patch.object(
                logit_search, 'discover_alpha_history', return_value=history
            ) as discover, patch.object(
                logit_search, 'load_validation_predictions',
                side_effect=lambda path, market, **kwargs: prepared[market],
            ) as loader, patch.object(
                logit_search, 'optimize_kelly_params', return_value='result'
            ) as optimize:
                result = logit_search.optimize_kelly_params_with_logit(
                    input_paths={'open': 'validation.csv'}, training_input='training.csv',
                    search_dir='history', model_refit_months=2, parallel=parallel,
                    n_trials=3, risk_penalty=.4, bootstrap_mode='load', bootstrap_dir='saved',
                )
                self.assertEqual(result, 'result')
                discover.assert_called_once_with('history')
                self.assertEqual(loader.call_count, 3)
                for call in loader.call_args_list:
                    self.assertIs(call.kwargs['alpha_history'], history)
                    self.assertEqual(call.kwargs['training_input'], 'training.csv')
                    self.assertEqual(call.kwargs['model_refit_months'], 2)
                optimize.assert_called_once_with(
                    validation_data=prepared, parallel=parallel, n_trials=3,
                    risk_penalty=.4, bootstrap_mode='load', bootstrap_dir='saved',
                )

    def test_saved_model_mode_skips_alpha_discovery(self):
        with patch.object(logit_search, 'discover_alpha_history') as discover, patch.object(
            logit_search, 'load_validation_predictions', return_value=pd.DataFrame()
        ) as loader, patch.object(logit_search, 'optimize_kelly_params'):
            logit_search.optimize_kelly_params_with_logit(use_monthly_models=False, parallel=False)
            discover.assert_not_called()
            self.assertEqual(loader.call_count, 3)
            self.assertTrue(all(not call.kwargs['use_monthly_models'] for call in loader.call_args_list))


if __name__ == '__main__':
    unittest.main()
