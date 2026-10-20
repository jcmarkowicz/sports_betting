import ast
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from ufc_betting.Models.LogisticRegression.inference import predict_logit
from ufc_betting.Models.LogisticRegression.utils import save_results


class ProbabilityModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(date=['2026-01-01'] * 2,
            winner=[1, 1], feature=[.1, .2], dec_open_blue=[2., 2.], dec_open_red=[2., 2.]))
        regularized = SimpleNamespace(model=SimpleNamespace(exog_names=['const', 'feature']),
                                      predict=lambda x: np.full(len(x), .4))
        inference = SimpleNamespace(model=SimpleNamespace(exog_names=['const']),
            predict=lambda x: np.full(len(x), .8),
            get_prediction=lambda x: SimpleNamespace(summary_frame=lambda:
                pd.DataFrame({'se': np.full(len(x), .03)})))
        self.loader = Mock(side_effect={'regularized': regularized, 'inference': inference}.__getitem__)
        self.env = dict(Path=Path, predict_logit=predict_logit, save_results=save_results,
            sm=SimpleNamespace(load=self.loader), joblib=SimpleNamespace(load=lambda p: None),
            TrainTestBuilder=SimpleNamespace(prepare_inference_data=lambda frame, **kw: frame[['feature']]),
            plot_model_metrics=Mock())
        file = Path(__file__).resolve().parents[1] / 'Notebooks/Models/training/lr_open_close.ipynb'
        cells = json.loads(file.read_text(encoding='utf-8'))['cells']
        source = next(''.join(c['source']) for c in cells if 'def run_inference(' in ''.join(c['source']))
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('run_inference', 'parlay_accuracy_metrics')]
        exec(compile(tree, str(file), 'exec'), self.env)

    def run_prediction(self, **kwargs):
        return self.env['run_inference'](self.frame, model_path='regularized', model_se_path='inference',
            scaler_path='scaler', encoder_path='encoder', model_type='open', odds_type='open', **kwargs)

    def test_default_and_inference_probabilities_metrics_and_saved_csv(self):
        for mode, probability, accuracy in [('regularized', .4, 0.), ('inference', .8, 1.)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'predictions.csv'
                output = io.StringIO()
                with redirect_stdout(output):
                    result = self.run_prediction(**({} if mode == 'regularized' else {'probability_model': mode}),
                                                 report_metrics=True, pred_save_path=path)
                np.testing.assert_allclose(result.proba_red, probability)
                np.testing.assert_allclose(result.proba_se, .03)
                self.assertEqual(result.attrs['metrics']['accuracy'], accuracy)
                self.assertEqual(result.attrs['metrics']['parlay_accuracy'], accuracy)
                saved = pd.read_csv(path)
                np.testing.assert_allclose(saved.proba_red, probability)
                self.assertTrue(saved.probability_model.eq(mode).all())
                self.assertIn(f'Probability model: {mode}', output.getvalue())

    def test_inference_does_not_load_regularized_artifact(self):
        self.run_prediction(probability_model='inference')
        self.loader.assert_called_once_with('inference')

    def test_invalid_option_fails_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'probability_model'):
            self.run_prediction(probability_model='unknown')
        self.loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
