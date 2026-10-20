import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ufc_betting.Models.LogisticRegression.glm_workflow import fit_glm_bundle, predict_glm_bundle
from ufc_betting.Models.Search.prediction_input import read_frozen_predictions


class GLMWorkflowTests(unittest.TestCase):
    def test_saved_bundle_exports_simulator_probabilities_and_bootstrap_sd(self):
        rng = np.random.default_rng(12)
        X = pd.DataFrame({'proba_fair_open_diff': rng.normal(size=100),
                          'reach_diff': rng.normal(size=100)})
        dates = np.repeat(pd.date_range('2020-01-01', periods=25), 4)
        y = np.tile([0, 1, 0, 1], 25)
        bundle = fit_glm_bundle(X, y, dates, alpha=.01, n_bootstrap=3)
        future = X.iloc[:8].copy()
        future['proba_fair_close1_diff'] = -future.proba_fair_open_diff
        future['date'] = pd.Timestamp('2025-01-01')
        future['winner'] = y[:8]
        future['dec_open_red'] = 2.1
        future['dec_open_blue'] = 1.9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            joblib.dump(bundle, path / 'bundle.joblib')
            loaded = joblib.load(path / 'bundle.joblib')
            result = predict_glm_bundle(future, loaded, odds_type='close1', fp=path / 'pred.csv')
            frozen = read_frozen_predictions(path / 'pred.csv')
        self.assertTrue(frozen.prediction_se.ge(0).all())
        self.assertTrue(frozen.proba_red.between(0, 1).all())
        np.testing.assert_allclose(frozen.proba_red, frozen.risk_red_proba)
        pd.testing.assert_series_equal(result.dec_open_red, future.dec_open_red)
        # Substitution occurs before the saved training transform.
        replaced = future.copy()
        replaced['proba_fair_open_diff'] = future.proba_fair_close1_diff
        expected = predict_glm_bundle(replaced, loaded)
        np.testing.assert_allclose(result.proba_red, expected.proba_red)
        np.testing.assert_allclose(result.prediction_se, expected.prediction_se)
        self.assertEqual(loaded['training_last_date'], pd.Timestamp('2020-01-25'))


if __name__ == '__main__':
    unittest.main()
