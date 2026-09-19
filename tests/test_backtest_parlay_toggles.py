import io
import itertools
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.BettingStrategy.Backtest.backtest_plots import (
    plot_backtest, parlay_analysis,
)


class ParlayToggleTests(unittest.TestCase):
    def test_toggle_combinations(self):
        for regular, heavy, count in itertools.product((False, True), (False, True), (1, 4)):
            with self.subTest(regular=regular, heavy=heavy, count=count):
                data = pd.DataFrame([
                    dict(date='2026-01-01', fighter_red=str(i), fighter_blue=str(i+5),
                         winner=1, pred_winner=1, blue=.05, red=.95,
                         fair_blue=20., fair_red=1.1, odds_blue=5., odds_red=1.2,
                         open_red=-500, open_blue=400, se=.01)
                    for i in range(count)
                ])
                results, parlays = simulate_kelly(
                    data, prob_cols=['blue', 'red'],
                    fair_decimal_cols=['fair_blue', 'fair_red'],
                    real_decimal_cols=['odds_blue', 'odds_red'],
                    pred_winner_col='pred_winner', z=1,
                    calc_parlay=regular, calc_heavy_favorite_parlay=heavy,
                )
                if not regular:
                    self.assertTrue(results.regular_parlay_net.eq(0).all())
                if not heavy:
                    self.assertTrue(results.heavy_parlay_net.eq(0).all())
                self.assertAlmostEqual(
                    results.bankroll_postevent.iloc[0],
                    1000 + results.event_payout_money_line.iloc[0]
                    + results.parlay_net.iloc[0],
                )
                if not regular and not heavy:
                    self.assertTrue(parlays.empty)
                    self.assertTrue(results.parlay_net_odds.eq(0).all())
                    parlay_analysis(parlays)
                elif heavy and not regular and count == 4:
                    self.assertGreater(results.parlay_net_odds.iloc[0], 0)
                pd.read_csv(io.StringIO(parlays.to_csv(index=False)))
                with patch.object(plt, 'show'):
                    plot_backtest(results, 1000)
                    plot_backtest(results.drop(columns=['parlay_net', 'parlay_net_odds']), 1000)
                plt.close('all')


if __name__ == '__main__':
    unittest.main()
