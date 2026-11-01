"""Comparison cohorts, probability sources, and metric regressions."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Notebooks/Models/training'))
import lr_workflow as workflow

class TestNotebookCohort(unittest.TestCase):
    def test_metrics_include_probability_one_and_exclude_voids(self):
        frame = pd.DataFrame(dict(winner=[0, 1, 0, 2], proba_red=[0., 1., 1., .4]))
        score = workflow.score_predictions(frame)
        self.assertAlmostEqual(score['ece'], 1/3)
        self.assertAlmostEqual(score['brier'], 1/3)
        self.assertAlmostEqual(score['mean_calibration_error'], .25)
        self.assertAlmostEqual(score['maximum_calibration_error'], .5)
        self.assertEqual(score['n_scored'], 3)
        self.assertEqual(score['n_void_or_missing'], 1)

    def test_comparison_uses_new_test_and_preserves_all_probability_sources(self):
        dates = pd.date_range('2025-01-01', periods=12, freq='MS')
        frame = pd.DataFrame(dict(date=dates, winner=np.arange(12)%2,
            fighter_red=[f'red{i}' for i in range(12)], fighter_blue='blue',
            feature=np.arange(12,dtype=float)))
        frame['source'] = 'old'
        for market in ('open', 'close1', 'close2'):
            for side in ('red', 'blue'):
                frame[f'dec_{market}_{side}'] = 2.1
        fresh = frame.iloc[8:].copy()
        fresh['source'] = 'new'
        fresh.loc[fresh.index[-1], 'winner'] = 2
        class Builder:
            valid_cols = ['date','winner','fighter_red','fighter_blue','feature']
        fits, predictions = [], []
        def fit(*args, **kw): fits.append(kw)
        def infer(rows, **kw):
            predictions.append((rows.copy(), kw))
            result = rows.copy()
            result['proba_red'] = .6 if kw['probability_model']=='regularized' else .8
            result['pred_winner'] = 1
            return result
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(workflow,'make_builder',return_value=Builder()), \
             patch.object(workflow,'workflow',side_effect=fit), \
             patch.object(workflow,'run_inference',side_effect=infer):
            metrics = workflow.run_comparison(frame,fresh,directory,
                feature_sets={m:['feature'] for m in ('open','close1','close2')},
                val_start_date='2025-05-01',test_start_date='2025-09-01')
            self.assertEqual(len(fits),6)
            self.assertEqual(len(metrics),60)
            self.assertEqual(metrics.prediction_path.nunique(),60)
            self.assertTrue(metrics.loc[metrics.split=='test','held_out'].all())
            self.assertFalse(metrics.loc[(metrics.fit_mode=='trainval') & (metrics.split=='val'),'held_out'].any())
            self.assertEqual(set(metrics.loc[metrics.split=='test','n_scored']),{3})
            for rows, kwargs in predictions:
                if rows.date.min() >= pd.Timestamp('2025-09-01'):
                    self.assertTrue(rows.source.eq('new').all())
            for fit in fits:
                self.assertTrue(fit['df_model'].loc[lambda x:x.date >= '2025-09-01','source'].eq('new').all())
            paths={str(fit['model_path']) for fit in fits}
            self.assertEqual(len(paths),6)

if __name__ == '__main__':
    unittest.main()
