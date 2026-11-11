import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ufc_betting.Results.comparison_calibration import plot_comparison_calibration, rolling_edge_summary, brier_decomposition, rolling_brier_summary


class ComparisonCalibrationTests(unittest.TestCase):
    def test_orientation_selection_and_empty(self):
        frame = pd.DataFrame({
            'date': ['2026-01-01'] * 4,
            'winner': [0, 1, 2, 1], 'pred_winner': [0, 1, 1, 1],
            'choice_proba': [.8, .6, .7, .9],
            'choice_fstar': [.1, 0, .1, .1],
            'real_odds_red': [3., 2., 2., 2.], 'real_odds_blue': [1.5, 2., 2., 2.],
            'bankroll_postevent': [100, 100, 100, 0],
            'event_payout_money_line': [0, 0, 0, 0], 'parlay_net': [0, 0, 0, 0],
        })
        fig, axes = plot_comparison_calibration(frame, odds_type='open', show=False)
        self.assertEqual(len(fig.axes), 5)
        np.testing.assert_allclose(axes[0].lines[1].get_xdata(), [2/3])
        np.testing.assert_allclose(axes[0].lines[1].get_ydata(), [1])
        np.testing.assert_allclose(axes[1].lines[2].get_ydata(), [100/3])
        self.assertTrue(all('(n=1)' in ax.get_title() for ax in axes[:3]))
        plt.close(fig)
        fig, axes = plot_comparison_calibration(frame.iloc[:0], odds_type='open', show=False)
        self.assertTrue(all('(n=0)' in ax.get_title() for ax in axes[:3]))
        plt.close(fig)

    def test_timeline_sorts_dates_and_weights_bets_not_events(self):
        frame = pd.DataFrame({
            'date': ['2026-02-01', '2026-01-01', '2026-02-01'],
            'winner': [0, 1, 1], 'pred_winner': [1, 1, 1],
            'proba_red': [.7, .6, .8], 'proba_blue': [.3, .4, .2],
            'choice_fstar': [.1, .2, .3],
            'dec_close1_red': [2., 2., 2.], 'dec_close1_blue': [2., 2., 2.],
        })
        fig, axes = plot_comparison_calibration(frame, odds_type='close1', show=False)
        np.testing.assert_allclose(axes[0].lines[1].get_xdata(), [.5])
        np.testing.assert_allclose(axes[0].lines[1].get_ydata(), [2/3])
        np.testing.assert_allclose(axes[1].lines[2].get_ydata(), [50, 100/6])
        self.assertEqual(pd.Timestamp(axes[1].lines[1].get_xdata()[0]), pd.Timestamp('2026-01-01'))
        plt.close(fig)

    def test_rolling_windows_bootstrap_and_gap(self):
        rows = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-01', '2026-01-01', '2026-02-01', '2026-03-01']),
            'model': [.8, .8, .6, .7], 'vegas': [.5] * 4, 'outcome': [1, 0, 1, 0]})
        result = rolling_edge_summary(rows, window_bets=2, bootstrap_samples=200)
        self.assertEqual(result['count'].tolist(), [2, 3, 2])
        np.testing.assert_allclose(result.predicted, [30, 70/3, 15])
        np.testing.assert_allclose(result.realized, [0, 100/6, 0])
        np.testing.assert_allclose(result.gap, result.realized - result.predicted)
        self.assertTrue(np.isnan(result.realized_low.iloc[0]))
        self.assertAlmostEqual(result.realized_low.iloc[-1], -50)
        self.assertAlmostEqual(result.realized_high.iloc[-1], 50)
        pd.testing.assert_frame_equal(result, rolling_edge_summary(rows, window_bets=2, bootstrap_samples=200))

    def test_brier_decomposition_and_rolling(self):
        result = brier_decomposition([0, 1], [.2, .8])
        self.assertAlmostEqual(result['brier'], .04)
        self.assertAlmostEqual(result['reliability'], .04)
        self.assertAlmostEqual(result['resolution'], .25)
        self.assertAlmostEqual(result['uncertainty'], .25)
        result = brier_decomposition([0, 1], [.51, .59])
        self.assertAlmostEqual(result['brier'], result['reliability'] - result['resolution'] +
                               result['uncertainty'] + result['residual'])
        self.assertNotAlmostEqual(result['residual'], 0)
        rows = pd.DataFrame({'date': pd.to_datetime(['2026-01-01', '2026-02-01', '2026-02-01']),
                             'model': [.8, .6, .7], 'vegas': [.5, .5, .5], 'outcome': [1, 0, 1]})
        rolling = rolling_brier_summary(rows, window_bets=2)
        np.testing.assert_allclose(rolling.model, [.04, (.36 + .09)/2])
        np.testing.assert_allclose(rolling.vegas, [.25, .25])
        self.assertEqual(rolling['count'].tolist(), [1, 2])
