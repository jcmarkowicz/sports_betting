import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ufc_betting.BettingStrategy.Backtest.analysis_functions import pick_accuracy, bin_parlay_prob
from ufc_betting.BettingStrategy.Backtest.backtest_plots import summary_stats, parlay_analysis
from ufc_betting.Results.vegas_accuracy import score_predictions


class AccuracyVoidTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_fight_accuracy_excludes_voids_but_pick_count_keeps_them(self):
        frame = pd.DataFrame(dict(date=['2026-01-01']*4, winner=[1, 0, 2, 3],
            pred_winner=[1]*4, choice_fstar=[.1]*4, choice_ev=[.2]*4,
            bankroll_pct_change=[.1]*4))
        with patch('seaborn.histplot') as hist, patch('matplotlib.pyplot.show'):
            pick_accuracy(frame)
            self.assertEqual(hist.call_args_list[0].args[0].iloc[0], .5)
            self.assertEqual(hist.call_args_list[1].args[0].iloc[0], 4)
        # An all-void event must not appear as an accuracy of zero.
        extra = frame.iloc[2:].assign(date='2026-02-01')
        with patch('seaborn.histplot') as hist, patch('seaborn.regplot'), patch('matplotlib.pyplot.show'):
            summary_stats(pd.concat([frame, extra]))
            self.assertEqual(hist.call_args_list[0].args[0].tolist(), [.5])

    def test_parlay_bins_exclude_refunds_from_accuracy(self):
        frame = pd.DataFrame(dict(parlay_prob=[.7]*4, parlay_ev=[.2]*4,
                                   parlay_net_odds=[3., -1., 0., np.nan]))
        with patch('matplotlib.pyplot.show'):
            result = bin_parlay_prob(frame)
            self.assertEqual(result.accuracy.dropna().tolist(), [.5])
            self.assertEqual(result['count'].sum(), 2)
            with patch('seaborn.barplot') as bars:
                parlay_analysis(frame.copy())
                accuracy = bars.call_args_list[-1].kwargs['data']
                self.assertEqual(accuracy.accuracy.dropna().tolist(), [.5])

    def test_scores_ignore_voids_and_all_void_is_nan(self):
        frame = pd.DataFrame(dict(winner=[0,1,2,3], model_prediction=[0,1,1,0],
                                  model_probability=[.1,.9,.8,.2]))
        self.assertEqual(score_predictions(frame, 'model')['accuracy'], 1.)
        self.assertTrue(np.isnan(score_predictions(frame.iloc[2:], 'model')['accuracy']))
