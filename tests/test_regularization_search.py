import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.Models.Search import regularization_search as search


class RegularizationSearchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.dates = pd.Series(np.repeat(pd.date_range('2020-01-01', periods=24, freq='7D'), 4))
        self.X = pd.DataFrame({'x': rng.normal(size=96)})
        self.y = pd.Series(np.tile([0, 1, 1, 0], 24))

    def test_event_folds_are_chronological_even_with_shuffled_rows(self):
        dates = self.dates.sample(frac=1, random_state=3).reset_index(drop=True)
        for train, valid in search.chronological_folds(dates, n_splits=3, gap=1):
            self.assertLess(dates.iloc[train].max(), dates.iloc[valid].min())
            self.assertFalse(set(dates.iloc[train]) & set(dates.iloc[valid]))
            self.assertEqual(len(train) % 4, 0)
            self.assertEqual(len(valid) % 4, 0)
            self.assertGreater((dates.iloc[valid].min() - dates.iloc[train].max()).days, 7)

    def test_csv_runner_exports_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.X.assign(winner=self.y, date=self.dates).to_csv(root / 'input.csv', index=False)
            result = search.run_from_csv(
                input_path=root / 'input.csv', output_dir=root / 'results',
                features=['x'], n_trials=1, n_splits=2,
                logit_alpha_bounds=(.5, .5), glm_alpha_bounds=(.1, .1),
            )
            self.assertEqual(set(result['summary'].model_type), {'glm', 'logit'})
            self.assertEqual(
                {p.name for p in (root / 'results').iterdir()},
                {'logit_best_params.json', 'glm_best_params.json'},
            )
            for name, bundle in result['bundles'].items():
                saved = json.loads((root / f'results/{name}_best_params.json').read_text())
                self.assertEqual(saved, bundle['study'].best_params)
                self.assertEqual(saved, result['best_params'][name])
                self.assertEqual(len(bundle['model'].model.endog), len(self.X))
                self.assertEqual(len(bundle['folds']), 2)

    def test_both_models_minimize_fold_brier_and_preprocess_training_only(self):
        for model_type in ('logit', 'glm'):
            with self.subTest(model_type=model_type), patch.object(
                search, '_preprocessor', wraps=search._preprocessor
            ) as preprocess:
                result = search.search_regularization(
                    self.X, self.y, self.dates, model_type=model_type,
                    n_trials=2, n_splits=3, alpha_bounds=(0.1, 0.2))
                study = result['study']
                self.assertEqual(study.direction.name, 'MINIMIZE')
                self.assertEqual(set(study.best_params),
                                 {'alpha'} if model_type == 'logit' else {'alpha', 'L1_wt'})
                self.assertAlmostEqual(study.best_value,
                                       np.mean(study.best_trial.user_attrs['fold_brier']))
                self.assertEqual([len(c.args[0]) for c in preprocess.call_args_list], [24, 48, 72, 96])

    def test_monthly_holdouts_do_not_enter_their_own_search(self):
        frame = self.X.assign(winner=self.y, date=self.dates)
        builder = SimpleNamespace(all_processed_df=frame, selected_feats=['x'],
                                  target_col='winner', date_col='date', start_date=pd.Timestamp('2020-05-01'))
        with patch.object(search, 'search_regularization', wraps=search.search_regularization) as tune:
            result = search.walk_forward_regularization(
                builder, n_trials=1, n_splits=2,
                logit_alpha_bounds=(0.5, 0.5), glm_alpha_bounds=(0.1, 0.1))
        for call, row in zip(tune.call_args_list, result['monthly_results'].itertuples()):
            self.assertTrue((call.args[2] < row.training_cutoff).all())
        counts = result['predictions'].groupby(['model_type', 'source_index']).size()
        self.assertTrue((counts == 1).all())
        self.assertEqual(len(result['predictions']), 2 * (self.dates >= builder.start_date).sum())
        self.assertTrue(result['summary'].holdout_brier.between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
