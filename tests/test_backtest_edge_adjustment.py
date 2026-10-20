import unittest

import pandas as pd

from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly


class EdgeAdjustmentTests(unittest.TestCase):
    def test_legacy_flag_never_changes_supplied_mdd_sizing(self):
        data = pd.DataFrame([dict(
            date='2026-01-01', fighter_red='Red', fighter_blue='Blue',
            winner=1, pred_winner=1, red=.8, blue=.2,
            fair_red=2., fair_blue=2., odds_red=1.9, odds_blue=1.9,
            open_red=-111, open_blue=-111,
        )])
        kwargs = dict(
            prob_cols=['blue', 'red'],
            fair_decimal_cols=['fair_blue', 'fair_red'],
            real_decimal_cols=['odds_blue', 'odds_red'],
            pred_winner_col='pred_winner', max_drawdown=.3, N=250,
        )
        default, _ = simulate_kelly(data, **kwargs)
        enabled, _ = simulate_kelly(data, **kwargs, adjust_mdd_by_edge=True)
        disabled, _ = simulate_kelly(data, **kwargs, adjust_mdd_by_edge=False)
        pd.testing.assert_frame_equal(default, enabled)
        self.assertGreater(disabled.choice_fstar.iloc[0], 0)
        pd.testing.assert_frame_equal(disabled, enabled)


if __name__ == '__main__':
    unittest.main()
