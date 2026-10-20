import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.Backtest.backtest_plots import plot_bankroll_distributions


class BacktestDistributionTests(unittest.TestCase):
    def test_actual_parlay_loss_and_additive_contributions(self):
        for scale in (1., 100.):
            # Shared bankroll: 500 -> 1000 -> 940. Parlays lose $10 overall,
            # even though standalone compounding would incorrectly show +3.4%.
            frame = pd.DataFrame(dict(date=['2025-01-01']*2+['2025-02-01'],
                bankroll_postevent=[1000., 1000., 940.],
                event_payout_money_line=[450., 450., 0.],
                parlay_net=[50., 50., -60.],
                bankroll_pct_change=np.array([1., 1., -.06])*scale))
            original = frame.copy(deep=True)
            with patch('matplotlib.pyplot.show'):
                fig, axes = plot_bankroll_distributions(frame, pd.DataFrame())
            try:
                ml, pl, total = [ax.lines[0].get_ydata() for ax in axes[1]]
                np.testing.assert_allclose(ml, [90., 90.])
                np.testing.assert_allclose(pl, [10., -2.])
                np.testing.assert_allclose(total, ml+pl)
                self.assertEqual(len(fig.axes), 7)  # Only combined has a growth axis.
                np.testing.assert_allclose(fig.axes[-1].lines[0].get_ydata(),
                                           100*(np.array([2., 1.88])**[1., .5]-1))
                pd.testing.assert_frame_equal(frame, original)
                fig.canvas.draw()
            finally:
                plt.close(fig)
