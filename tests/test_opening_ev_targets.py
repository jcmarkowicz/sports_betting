import unittest
import numpy as np
import pandas as pd
from ufc_betting.Models.RiskManagement.opening_ev_targets import expected_events, required_targets
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly


class OpeningTargetTests(unittest.TestCase):
    def test_first_crossing_including_nonmonotone(self):
        data = pd.DataFrame({"date":["2020-01-01"]*4,"movement_pct":[0,20,50,100],
            "expected_return_pct":[3,9,4,8], "mean_absolute_decimal_change":[0,.2,.5,1.]})
        result = required_targets(data, [25, 50, 100]).set_index("target_pct_opening_ev")
        self.assertEqual(result.loc[25,"required_movement_pct"],0)
        self.assertEqual(result.loc[100,"required_movement_pct"],20)

    def test_matches_notebook_event_stakes(self):
        frame = pd.DataFrame({"date":["2020-01-01"]*2,"fighter_red":["a","c"],"fighter_blue":["b","d"],
            "pred_winner":[1,0],"winner":[1,0],"proba_red":[.7,.3],"proba_blue":[.3,.7],
            "proba_se":[.03,.05],"dec_close1_red":[2.,2.],"dec_close1_blue":[2.,2.],
            "dec_fair_close1_red":[2.,2.],"dec_fair_close1_blue":[2.,2.],"open_red":[100,100],"open_blue":[100,100]})
        direct = expected_events(frame).expected_return_pct.iloc[0]
        bets,_ = simulate_kelly(frame,prob_cols=['proba_blue','proba_red'],
            fair_decimal_cols=['dec_fair_close1_blue','dec_fair_close1_red'],
            real_decimal_cols=['dec_close1_blue','dec_close1_red'],pred_winner_col='pred_winner',
            init_bankroll=500,bankroll_floor=100,max_drawdown=.4,parlay_mdd=.4,N=250,N_parlay=1490,
            calc_parlay=False,calc_heavy_favorite_parlay=False,z=.5,prediction_se_col='proba_se')
        self.assertAlmostEqual(direct,100*(bets.choice_fstar*bets.choice_ev).sum(),places=9)

    def test_zero_benchmark_undefined(self):
        data=pd.DataFrame({"date":["2020-01-01"],"movement_pct":[100],"expected_return_pct":[0],"mean_absolute_decimal_change":[1]})
        self.assertTrue(required_targets(data).required_movement_pct.isna().all())
