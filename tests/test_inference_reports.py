import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from ufc_betting.Results.inference_reports import REPORTS, load_report, report_paths


class InferenceReportTests(unittest.TestCase):
    def test_test_split_is_excluded(self):
        for mode in ('train', 'trainval'):
            with self.assertRaises(ValueError):
                report_paths('.', mode, 'test')
        self.assertEqual(set(REPORTS.values()), {('train', 'train'), ('train', 'val'), ('trainval', 'train')})

    def test_report_preserves_metrics_and_matches_notebook_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = pd.DataFrame([dict(model='close1', odds='close1', accuracy=.7,
                f1=.6, brier=.2, expected_calibration_error=.1, mean_calibration_error=.15)])
            metrics.to_csv(root / 'train_val_inference_metrics.csv', index=False)
            path = root / 'report_predictions/train/close1/inference'
            path.mkdir(parents=True)
            frame = pd.DataFrame({'winner': [0, 1, 1, 2], 'proba_red': [.1, .6, .9, .7],
                                  'probability_model': ['inference'] * 4})
            frame.to_csv(path / 'val_with_close1.csv', index=False)
            table, curves = load_report(root, 'train', 'val')
            pd.testing.assert_frame_equal(table, metrics.set_index(['model', 'odds']))
            actual, predicted = calibration_curve([0, 1, 1], [.1, .6, .9], n_bins=10)
            np.testing.assert_allclose(curves[0][1], predicted)
            np.testing.assert_allclose(curves[0][2], actual)
            self.assertEqual(curves[0][3], 3)


if __name__ == '__main__':
    unittest.main()
