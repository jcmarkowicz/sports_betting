import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ufc_betting.Results.live_calibration import calibration_rows, plot_live_calibration


class LiveCalibrationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            'winner_bool': [1, 0, 2, 3, 1, 0],
            'pred_winner_open': [1, 0, 1, 0, 1, 0],
            'choice_proba_open': [.7, .8, .6, .7, np.nan, .6],
            'open_red': [-150, 200, -110, -110, 100, 0],
            'open_blue': [130, -220, -110, -110, 100, 100],
            'net_stake_open': [.1, 0, .1, .1, .1, .1],
        })

    def test_orientation_no_vig_selection_and_exclusions(self):
        rows = calibration_rows(self.frame(), 'open')
        np.testing.assert_allclose(rows.model, [.7, .2])
        np.testing.assert_allclose(rows.vegas, [.6 / (.6 + 100/230), (1/3) / (1/3 + 220/320)])
        self.assertEqual(rows.selected.tolist(), [True, False])
        self.assertEqual(rows.outcome.tolist(), [1, 0])

    def test_plot_counts_and_empty_history(self):
        for frame in (self.frame(), self.frame().iloc[:0]):
            fig = plot_live_calibration(frame, 'open')
            self.assertEqual(len(fig.axes), 2)
            expected = '(n=1)' if len(frame) else '(n=0)'
            self.assertIn(expected, fig.axes[0].get_title())
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
