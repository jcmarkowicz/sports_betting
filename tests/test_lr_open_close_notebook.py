"""Regression checks for the notebook's shared preprocessing/inference path."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ufc_betting.Models.LogisticRegression.inference import predict_logit
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.Models.LogisticRegression.utils import save_results


class NotebookInferenceTests(unittest.TestCase):
    def setUp(self):
        self.feats = ['proba_fair_open_diff', 'reach_diff', 'elo_pred']
        n = 40
        self.frame = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=n),
            'winner': np.arange(n) % 2,
            'proba_fair_open_diff': np.linspace(-.8, .8, n),
            'proba_fair_close1_diff': np.linspace(.7, -.7, n),
            'proba_fair_close2_diff': np.linspace(-.4, .4, n),
            'reach_diff': np.sin(np.arange(n)),
            'elo_pred': np.arange(n) % 3,
            'math_red': 0, 'math_blue': 1, 'womens_fight': 0,
            'fighter_red': [f'red_{i}' for i in range(n)],
            'fighter_blue': 'blue',
            'num_wins_red': 2, 'num_wins_blue': 2,
            'num_losses_red': 1, 'num_losses_blue': 1,
        })
        for market in ('open', 'close1', 'close2'):
            for side in ('red', 'blue'):
                for prefix in ('', 'dec_', 'dec_fair_'):
                    self.frame[f'{prefix}{market}_{side}'] = 2.0
        builder = TrainTestBuilder(self.frame, self.feats, 'winner', 'date',
                                   'open', 2010, 2, 26)
        pkt = builder.prepare_train_test(.85)
        self.scaler, self.encoder = pkt['scaler'], pkt['encoder']
        self.names = ['const'] + list(pkt['X_train'].columns)
        self.model = SimpleNamespace(
            model=SimpleNamespace(exog_names=self.names),
            predict=lambda design: 1 / (1 + np.exp(-design['proba_fair_open_diff'])),
        )
        self.model_se = SimpleNamespace(
            model=SimpleNamespace(exog_names=['const', 'reach_diff']),
            get_prediction=lambda design: SimpleNamespace(
                summary_frame=lambda: pd.DataFrame({'se': np.full(len(design), .05)})
            ),
        )
        artifacts = dict(model=self.model, model_se=self.model_se,
                         scaler=self.scaler, encoder=self.encoder)
        self.env = dict(
            Path=Path, TrainTestBuilder=TrainTestBuilder, predict_logit=predict_logit,
            save_results=save_results, winner_col='winner', date_col='date',
            df_model=self.frame, YEAR=2010, MONTH=2, DAY=26, TRAIN_SIZE=.85,
            VAL_START_DATE=pd.Timestamp('2025-01-21'), TEST_START_DATE=pd.Timestamp('2025-02-01'),
            config=SimpleNamespace(model_open_path='model', model_open_se_path='model_se',
                                   scaler_open_path='scaler', encoder_open_path='encoder'),
            sm=SimpleNamespace(load=artifacts.__getitem__),
            joblib=SimpleNamespace(load=artifacts.__getitem__),
        )
        notebook = Path(__file__).resolve().parents[1] / 'Notebooks/Models/training/lr_open_close.ipynb'
        cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
        for cell in cells:
            if cell['cell_type'] == 'code':
                tree = ast.parse(''.join(cell['source']))
                tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                exec(compile(tree, str(notebook), 'exec'), self.env)

    def infer(self, df, odds_type, *, model_type='open', save_path=None, **kwargs):
        return self.env['run_inference'](
            df, model_path='model', model_se_path='model_se',
            scaler_path='scaler', encoder_path='encoder',
            model_type=model_type, odds_type=odds_type, save_path=save_path, **kwargs,
        )

    def test_optional_metrics_exclude_nonbinary_outcomes_and_align_plot_inputs(self):
        rows = self.frame.iloc[[7, 8, 19, 20, 31]].copy()
        rows['winner'] = [1, 0, 2, 3, np.nan]
        rows.index = [90, 90, 12, 13, 14]
        plot = self.env['plot_model_metrics'] = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plots' / 'metrics.png'
            result = self.infer(rows, 'close2', report_metrics=True,
                                show_plots=True, plot_path=path, plot_title='Validation')
            self.assertTrue(path.parent.exists())
            self.assertEqual(plot.call_args.kwargs, dict(suptitle='Validation', path=path))
        evaluated = result.iloc[:2]
        metrics = result.attrs['metrics']
        self.assertAlmostEqual(metrics['accuracy'],
                               (evaluated.winner == evaluated.pred_winner).mean())
        self.assertAlmostEqual(metrics['brier_score'],
                               ((evaluated.winner - evaluated.proba_red) ** 2).mean())
        self.assertEqual((metrics['n_evaluated'], metrics['n_excluded']), (2, 3))
        self.assertEqual(len(result), len(rows))
        self.assertEqual(result.index.tolist(), rows.index.tolist())
        for values in plot.call_args.args:
            self.assertEqual(values.index.tolist(), [0, 1])

    def test_evaluation_requires_labels_and_skips_single_class_plots(self):
        self.env['plot_model_metrics'] = Mock()
        with self.assertRaisesRegex(ValueError, 'column'):
            self.infer(self.frame.drop(columns='winner'), 'open', report_metrics=True)
        with self.assertRaisesRegex(ValueError, 'at least one'):
            self.infer(self.frame.assign(winner=2), 'open', show_plots=True)
        result = self.infer(self.frame.assign(winner=1), 'open', show_plots=True)
        self.assertEqual(result.attrs['metrics']['n_evaluated'], len(self.frame))
        self.env['plot_model_metrics'].assert_not_called()

    def test_default_inference_does_not_evaluate(self):
        self.env['plot_model_metrics'] = Mock()
        result = self.infer(self.frame, 'open')
        self.assertNotIn('metrics', result.attrs)
        self.env['plot_model_metrics'].assert_not_called()

    def test_workflow_trains_on_train_evaluates_val_and_exports_three_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            self.env['results_dir'] = Path(directory) / 'results'
            self.env['plots_dir'] = Path(directory) / 'plots'
            trainer = Mock(return_value={'model': self.model, 'model_se': self.model_se})
            self.env['train_logit_model'] = trainer
            self.env['plot_model_metrics'] = Mock()
            outputs = self.env['workflow'](self.feats, 'open')
            self.assertEqual([len(frame) for frame in outputs], [20, 11, 9])
            self.assertEqual([len(arg) for arg in trainer.call_args.args], [20, 20, 11, 11])
            for split, expected in zip(('train', 'val', 'test'), outputs):
                saved = pd.read_csv(self.env['results_dir'] / f'{split}_logit_open.csv')
                self.assertEqual(len(saved), len(expected))
                self.assertFalse(saved[['proba_red', 'proba_se']].isna().any().any())

    def test_workflow_combines_train_val_without_fitting_on_test(self):
        with tempfile.TemporaryDirectory() as directory:
            self.env['results_dir'] = Path(directory) / 'results'
            self.env['plots_dir'] = Path(directory) / 'plots'
            trainer = Mock(return_value={'model': self.model, 'model_se': self.model_se})
            self.env['train_logit_model'] = trainer
            plot = self.env['plot_model_metrics'] = Mock()
            with patch.object(TrainTestBuilder, 'scale_encode',
                              wraps=TrainTestBuilder.scale_encode) as preprocessing:
                outputs = self.env['workflow'](self.feats, 'open', train_on_train_val=True)
            self.assertEqual([len(frame) for frame in outputs], [20, 11, 9])
            self.assertEqual([len(arg) for arg in trainer.call_args.args], [31, 31, 9, 9])
            self.assertEqual(trainer.call_args.kwargs['evaluation_label'], 'TEST')
            preprocessing.assert_called_once()
            raw_fit, raw_test = preprocessing.call_args.args[:2]
            self.assertEqual(raw_fit.index.tolist(), list(range(31)))
            self.assertEqual(raw_test.index.tolist(), list(range(31, 40)))
            fitted_X = trainer.call_args.args[0]
            np.testing.assert_allclose(fitted_X[['proba_fair_open_diff', 'reach_diff']].mean(), 0, atol=1e-12)
            self.assertIn('(in-sample)', plot.call_args_list[1].kwargs['suptitle'])
            self.assertIn('(held-out)', plot.call_args_list[2].kwargs['suptitle'])

    def test_combined_workflow_preserves_validation_prediction_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.env['results_dir'] = Path(directory) / 'results'
            self.env['plots_dir'] = Path(directory) / 'plots'
            self.env['train_logit_model'] = Mock(return_value={'model': self.model, 'model_se': self.model_se})
            self.env['plot_model_metrics'] = Mock()
            kwargs = dict(pred_output_filename='val_logit_open_with_open.csv')
            self.env['workflow'](self.feats, 'open', **kwargs)
            originals = {p: p.read_bytes() for p in self.env['results_dir'].glob('*.csv')}
            self.env['workflow'](self.feats, 'open', train_on_train_val=True, **kwargs)
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(len(list(self.env['results_dir'].glob('trainval_fit_*.csv'))), 3)
            self.assertTrue((self.env['results_dir'] / 'trainval_fit_val_logit_open_with_open.csv').exists())

    def test_closing_models_do_not_switch_opening_features(self):
        rows = self.frame.iloc[[7, 19, 31]]
        expected = self.infer(rows, 'open')['proba_red']
        for model_type in ('close1', 'close2'):
            for odds_type in ('open', 'close1', 'close2'):
                result = self.infer(rows, odds_type, model_type=model_type)
                np.testing.assert_allclose(result['proba_red'], expected)
        with self.assertRaisesRegex(ValueError, 'model_type'):
            self.infer(rows, 'open', model_type='invalid')

    def test_custom_prediction_filename_is_only_used_for_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.env['results_dir'] = Path(directory) / 'results'
            self.env['plots_dir'] = Path(directory) / 'plots'
            self.env['train_logit_model'] = Mock(return_value={'model': self.model, 'model_se': self.model_se})
            self.env['plot_model_metrics'] = Mock()
            self.env['workflow'](self.feats, 'open', pred_output_filename='val_logit_open_with_open.csv')
            files = [('train_logit_open.csv', 20), ('val_logit_open_with_open.csv', 11),
                     ('test_logit_open.csv', 9)]
            for filename, count in files:
                self.assertEqual(len(pd.read_csv(self.env['results_dir'] / filename)), count)
            with self.assertRaisesRegex(ValueError, 'must be different'):
                self.env['workflow'](self.feats, 'open', pred_output_filename='test_logit_open.csv')

    def test_market_substitution_matches_manual_transform_with_multiclass_encoding(self):
        rows = self.frame.iloc[[7, 19, 31]].copy()
        original = rows.copy(deep=True)
        for market in ('open', 'close1', 'close2'):
            result = self.infer(rows, odds_type=market)
            features = rows[self.feats].copy()
            features['proba_fair_open_diff'] = rows[f'proba_fair_{market}_diff']
            scaled = self.scaler.transform(features[self.scaler.feature_names_in_])
            expected = 1 / (1 + np.exp(-scaled[:, 0]))
            np.testing.assert_allclose(result['proba_red'], expected)
            np.testing.assert_allclose(result['proba_se'], .05)
            self.assertEqual(result.index.tolist(), rows.index.tolist())
        pd.testing.assert_frame_equal(rows, original)

    def test_odds_switch_defaults_off_and_requires_a_closing_market_when_enabled(self):
        rows = self.frame.iloc[[7, 19, 31]]
        expected = TrainTestBuilder.transform_features(rows, self.scaler, self.encoder)
        for market in (None, 'open', 'close1', 'close2'):
            actual = TrainTestBuilder.prepare_inference_data(
                rows, self.scaler, self.encoder, odds_type=market,
            )
            pd.testing.assert_frame_equal(actual, expected)
        for market in (None, 'open', 'invalid'):
            with self.assertRaisesRegex(ValueError, 'requires odds_type'):
                TrainTestBuilder.prepare_inference_data(
                    rows, self.scaler, self.encoder,
                    switch_open_odds=True, odds_type=market,
                )

    def test_inference_accepts_unlabelled_rows_without_splitting_filtering_or_fitting(self):
        supplied = self.frame.iloc[[31, 7, 19]][
            ['proba_fair_close1_diff', 'reach_diff', 'elo_pred']
        ].copy()
        supplied.index = [100, 100, 42]
        original = supplied.copy(deep=True)
        self.scaler.fit = Mock(side_effect=AssertionError('must not refit'))
        self.encoder.fit = Mock(side_effect=AssertionError('must not refit'))
        with patch.object(TrainTestBuilder, '__init__', side_effect=AssertionError('no builder or split')):
            result = self.infer(supplied, 'close1')
        self.assertEqual(len(result), len(supplied))
        self.assertEqual(result.index.tolist(), supplied.index.tolist())
        self.assertFalse(result[['proba_red', 'proba_se']].isna().any().any())
        pd.testing.assert_frame_equal(supplied, original)
        self.scaler.fit.assert_not_called()
        self.encoder.fit.assert_not_called()

    def test_preparation_and_inference_are_separate_and_preserve_model_column_order(self):
        rows = self.frame.iloc[[31, 7, 19]].copy()
        inputs = TrainTestBuilder.prepare_inference_data(
            rows, self.scaler, self.encoder, switch_open_odds=True, odds_type='close2',
        )
        prepared = inputs.copy(deep=True)
        seen = {}

        def predict(design):
            seen['prediction'] = design.copy()
            return np.full(len(design), .7)

        def prediction_se(design):
            seen['se'] = design.copy()
            return SimpleNamespace(summary_frame=lambda: pd.DataFrame({'se': [.1, .2, .3]}))

        self.model.predict = predict
        self.model_se.get_prediction = prediction_se
        result = predict_logit(inputs, self.model, self.model_se)
        self.assertEqual(seen['prediction'].columns.tolist(), self.names)
        self.assertEqual(seen['se'].columns.tolist(), self.model_se.model.exog_names)
        self.assertTrue(seen['prediction']['const'].eq(1).all())
        self.assertEqual(result.index.tolist(), rows.index.tolist())
        np.testing.assert_allclose(result['proba_se'], [.1, .2, .3])
        pd.testing.assert_frame_equal(inputs, prepared)

    def test_optional_export_contains_all_supplied_rows(self):
        saver = self.env['save_results'] = Mock(wraps=save_results)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions' / 'close2.csv'
            result = self.infer(self.frame, 'close2', save_path=path)
            saved = pd.read_csv(path)
            self.assertEqual(len(saved), len(self.frame))
            np.testing.assert_allclose(saved['proba_red'], result['proba_red'])
            np.testing.assert_allclose(saved['proba_se'], result['proba_se'])
            saver.assert_called_once()
            self.assertEqual(saver.call_args.args[2], path)
            self.assertTrue({'correct_pred', 'prob_winner', 'pred_winner'} <= set(saved))
            np.testing.assert_array_equal(saved['correct_pred'],
                                          (saved.pred_winner == saved.winner).astype(int))

    def test_save_results_supports_custom_and_unavailable_outcomes(self):
        rows = self.frame.iloc[:4].rename(columns={'winner': 'outcome'}).copy()
        rows['outcome'] = [0, 1, 2, np.nan]
        result = self.infer(rows, 'open', target_col='outcome')
        self.assertTrue(result.correct_pred.iloc[:2].notna().all())
        self.assertTrue(result.correct_pred.iloc[2:].isna().all())
        unlabelled = self.infer(rows.drop(columns=['outcome', 'date']), 'open')
        self.assertTrue(unlabelled.correct_pred.isna().all())

    def test_invalid_inputs_raise_instead_of_silently_dropping_rows(self):
        supplied = self.frame.copy()
        supplied.loc[5, 'proba_fair_close1_diff'] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.infer(supplied, 'close1')
        with self.assertRaisesRegex(ValueError, 'odds_type'):
            self.infer(self.frame, 'invalid')


if __name__ == '__main__':
    unittest.main()
