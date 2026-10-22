import unittest
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.Models.LogisticRegression.utils import save_results
from ufc_betting.Models.LogisticRegression.statistical_functions import train_logit_model
from ufc_betting.Models.LogisticRegression.visual_functions import plot_model_metrics


class VoidEvaluationTests(unittest.TestCase):
    def test_notebook_filters_metrics_before_calling_plotter(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
            'Notebooks/Models/training/lr_open_close.ipynb').read_text(encoding='utf-8'))
        tree = ast.parse(''.join(notebook['cells'][2]['source']))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'workflow')
        pkt = {}
        for split, labels in [('train', [0, 1]), ('val', [0, 1, 2, 3]), ('test', [0, 1, 2, 3])]:
            pkt[f'y_{split}'] = pd.Series(labels)
            pkt[f'X_{split}'] = pd.DataFrame({'x': range(len(labels))})
            pkt[f'{split}_df'] = pd.DataFrame({'winner': labels})
        scored_counts = []
        def plot(y, pred, probs, **kwargs):
            self.assertTrue(np.isin(y, [0, 1]).all())
            scored_counts.append(len(y))
        def predict(x, *args):
            return pd.DataFrame({'proba_red': [.6] * len(x),
                                 'pred_winner': [1] * len(x), 'proba_se': [.01] * len(x)})
        ns = dict(pd=pd, Path=Path, VAL_START_DATE='2025-01-01', TEST_START_DATE='2026-01-01',
                  df_model=None, results_dir=Path('unused'), plots_dir=Path('unused'),
                  make_builder=lambda *args: SimpleNamespace(prepare_train_test=lambda **kwargs: pkt),
                  train_logit_model=lambda *args, **kwargs: dict(model=None, model_se=None),
                  predict_logit=predict, plot_model_metrics=plot,
                  save_results=lambda rows, *args: rows)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'workflow', 'exec'), ns)
        with patch('pathlib.Path.mkdir'):
            outputs = ns['workflow']([], 'open')
        self.assertEqual(scored_counts, [2, 2, 2])
        self.assertEqual([len(frame) for frame in outputs], [2, 4, 4])

    def test_training_excludes_voids_while_evaluation_and_exports_retain_them(self):
        rows = pd.DataFrame(dict(date=pd.date_range('2025-01-01', periods=12),
                                 feature=np.arange(12.), winner=[0, 1, 2, 3] * 3))
        builder = TrainTestBuilder(rows, ['feature'], 'winner', 'date', 'open',
                                  2024, 1, 1, required_features_only=True,
                                  retain_void_evaluation=True)
        with patch.object(pd.DataFrame, 'to_csv') as saved, patch('pathlib.Path.mkdir'):
            pkt = builder.prepare_train_test(val_start_date='2025-01-05',
                test_start_date='2025-01-09', val_split_path='val.csv', test_split_path='test.csv')
        self.assertEqual(saved.call_count, 2)
        self.assertEqual(pkt['y_train'].tolist(), [0, 1])
        self.assertAlmostEqual(pkt['scaler'].mean_[0], .5)
        for split in ('val', 'test'):
            self.assertEqual(pkt[f'y_{split}'].tolist(), [0, 1, 2, 3])
            result = save_results(pkt[f'{split}_df'], np.array([.2, .8, .7, .3]))
            self.assertEqual(len(result), 4)
            self.assertEqual(result.correct_pred.dropna().tolist(), [1, 1])
            self.assertTrue(result.proba_red.notna().all())

    def test_fit_rejects_void_training_targets(self):
        x = pd.DataFrame({'x': [1., 2., 3.]})
        with self.assertRaisesRegex(ValueError, 'binary outcomes'):
            train_logit_model(x, pd.Series([0, 1, 2]), x, pd.Series([0, 1, 3]))

    def test_metric_plots_ignore_void_predictions(self):
        with patch('matplotlib.pyplot.show'):
            plot_model_metrics([0, 1, 2, 3], [0, 1, 1, 0], [.1, .9, .9, .1])
            actual = [line.get_ydata().copy() for ax in plt.gcf().axes for line in ax.lines]
            plt.close('all')
            plot_model_metrics([0, 1], [0, 1], [.1, .9])
            expected = [line.get_ydata().copy() for ax in plt.gcf().axes for line in ax.lines]
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b)
            plt.close('all')
