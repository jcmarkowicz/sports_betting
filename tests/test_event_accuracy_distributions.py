import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.Backtest.backtest_plots import plot_event_accuracy_distributions


class AccuracyDistributionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(date=['2025-01-01']*4+['2025-02-01'],
            fighter_red=list('abcde'), fighter_blue=list('vwxyz'),
            winner=[1, 0, 1, 2, 0], pred_winner=[1, 1, 1, 1, 1],
            choice_fstar=[.1, 0., .1, .1, 0.], choice_ev=[.1]*5,
            dec_close2_red=[2.5, 1.5, 2., 1.5, 1.5],
            dec_close2_blue=[1.5, 2.5, 2., 2.5, 2.5]))

    def plot(self, frame, parlays=None, **kwargs):
        with patch('matplotlib.pyplot.show'):
            fig, axes, events = plot_event_accuracy_distributions(frame, parlays, **kwargs)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        return axes, events

    def test_actual_stakes_ties_voids_and_no_bet_event(self):
        frame = self.frame()
        original = frame.copy(deep=True)
        axes, events = self.plot(frame, odds_type='close2')
        self.assertEqual(len(axes), 4)
        np.testing.assert_allclose(events.bankroll_wagered_pct, [30., 0.])
        first = events.iloc[0]
        self.assertEqual(first.your_picks, 1.)
        self.assertEqual(first.your_picks_n, 2)
        self.assertEqual(first.vegas_all, 0.)
        self.assertEqual(first.vegas_all_n, 2)  # Void and pick'em excluded.
        self.assertEqual(first.vegas_wagered_n, 1)
        self.assertTrue(np.isnan(events.iloc[1].your_picks))
        self.assertEqual(events.iloc[1].vegas_all, 0.)
        self.assertEqual(sum(p.get_height() for p in axes[0].patches), 1)
        pd.testing.assert_frame_equal(frame, original)

    def test_parlay_only_legs_and_deduplication(self):
        frame = self.frame()
        for use_index in (True, False):
            tickets = pd.DataFrame(dict(date=['2025-01-01']*2, fstar_parlay=[.1, .1],
                                        choice_fighter_name=['b', 'b']))
            if use_index:
                tickets['fight_index'] = 1
            _, events = self.plot(frame, tickets)
            self.assertAlmostEqual(events.iloc[0].your_picks, 2/3)
            self.assertEqual(events.iloc[0].your_picks_n, 3)
            self.assertEqual(events.iloc[0].vegas_wagered_n, 2)
            self.assertAlmostEqual(events.iloc[0].bankroll_wagered_pct, 40.)
        duplicated = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        _, events = self.plot(duplicated)
        self.assertEqual(events.iloc[0].your_picks_n, 2)

    def test_execution_price_columns_and_market_validation(self):
        frame = self.frame().rename(columns={'dec_close2_red':'real_odds_red', 'dec_close2_blue':'real_odds_blue'})
        self.plot(frame, odds_type='close2')
        frame = self.frame().assign(dec_open_red=1.2, dec_open_blue=5.)
        with self.assertRaisesRegex(ValueError, 'Specify odds_type'):
            self.plot(frame)

    def test_post_ruin_stakes_are_not_wagers(self):
        frame = self.frame().assign(bankroll_postevent=0., event_payout_money_line=0., parlay_net=0.)
        tickets = pd.DataFrame(dict(date=['2025-01-01'], fstar_parlay=[.1], fight_index=[1]))
        _, events = self.plot(frame, tickets)
        self.assertTrue(events.your_picks.isna().all())
        self.assertEqual(events.vegas_all_n.sum(), 3)
        np.testing.assert_allclose(events.bankroll_wagered_pct, 0.)

    def test_event_total_includes_tickets_only_once(self):
        frame = self.frame()
        frame['event_total_fstar'] = [.55] * 4 + [0.]
        _, events = self.plot(frame)
        np.testing.assert_allclose(events.bankroll_wagered_pct, [55., 0.])
