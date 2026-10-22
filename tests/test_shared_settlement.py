import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.UpcomingPicks.betting_pipeline import betting_pipeline
from ufc_betting.DataPipeline.dataframes.moneylines import MoneylineDataFrame
from ufc_betting.DataPipeline.dataframes.parlays import ParlayDataFrame


class SharedSettlementTests(unittest.TestCase):
    def test_net_payout_and_void_priority(self):
        np.testing.assert_allclose(Calculator.moneyline_unit_profit(3., 1, [1, 0, 2, 3]), [2, -1, 0, 0])
        for outcomes, expected in [([1, 1], 5.), ([1, 0], -1.), ([2, 0], 0.), ([3, 1], 0.)]:
            self.assertEqual(float(Calculator.parlay_unit_profit(6., [1, 1], outcomes)), expected)

    def test_live_and_backtest_keep_same_legs_and_settle_identically(self):
        baseline_stakes = {}
        for market in ('open', 'close1', 'close2'):
            for outcome in (1, 0, 2, 3):
                with self.subTest(market=market, outcome=outcome):
                    frame = pd.DataFrame(dict(
                        date=['2026-01-01']*3, fighter_red=['red0','red1','red2'],
                        fighter_blue=['blue0','blue1','blue2'], winner=[outcome,1,1],
                        winner_name=['red0','red1','red2'], signal=[.85,.8,.7],
                        pred_winner=[1]*3, proba_red=[.85,.8,.7], proba_blue=[.15,.2,.3],
                        prediction_se=[.01]*3,
                    ))
                    for stage in ('open','close1','close2'):
                        for side in ('red','blue'):
                            frame[f'{stage}_{side}'] = 100.
                            frame[f'dec_{stage}_{side}'] = 2.
                            frame[f'dec_fair_{stage}_{side}'] = 2.
                    opening = BettingParams(z=None, mdd=.3, mdd_parlay=.5, N=250, N_parlay=500)
                    params = opening if market == 'open' else BettingParams.from_value({
                        **opening.to_dict(), 'opening_params': opening.to_dict(),
                        'decay_rates': dict(k_down=1.,k_up=1.,k_parlay_down=1.,k_parlay_up=1.),
                    })
                    predictions = (frame.pred_winner, frame.proba_red, frame.proba_blue, frame.prediction_se)
                    with patch('ufc_betting.UpcomingPicks.betting_pipeline.logit_predict', return_value=predictions):
                        ml, pl = betting_pipeline(
                            frame, [['signal']], [None], [None], [None], [market],
                            [[f'dec_fair_{market}_blue', f'dec_fair_{market}_red']],
                            [[f'dec_{market}_blue', f'dec_{market}_red']], 500,
                            betting_params={market:params},
                        )
                    inp = KellySimInput.from_predictions(frame, params, odds_type=market,
                                                         initial_bankroll=500, calc_parlay=True,
                                                         opening_data=frame)
                    data, kwargs = inp.to_simulation_args()
                    results, tickets = simulate_kelly(data, **kwargs)
                    # Populate other-market schema fields without changing this market's values.
                    for dat in (ml,pl):
                        for col in list(dat):
                            if col.endswith('_'+market):
                                for stage in ('open','close1','close2'):
                                    dat[col[:-len(market)]+stage] = dat[col]
                    settled_ml = MoneylineDataFrame(ml).with_results(frame).frame
                    settled_pl = ParlayDataFrame(pl.assign(date='2026-01-01')).with_results(frame).frame
                    self.assertEqual(len(settled_ml), 3)
                    live_profit = (settled_ml[f'net_stake_{market}'].abs()*settled_ml[f'net_odds_{market}']).sum()
                    live_profit += abs(settled_pl[f'net_stake_{market}'].iloc[0])*settled_pl[f'net_odds_{market}'].iloc[0]
                    self.assertAlmostEqual(500*(1+live_profit), float(results.bankroll_postevent.iloc[0]), places=8)
                    self.assertEqual(set(tickets.choice_fighter_name), {'red0','red1'})
                    stakes = ml[f'fstar_{market}'].to_numpy()
                    if outcome == 1: baseline_stakes[market] = stakes
                    else: np.testing.assert_allclose(stakes, baseline_stakes[market])
                    if outcome in (2,3):
                        self.assertEqual(settled_ml[f'net_odds_{market}'].iloc[0], 0.)
                        self.assertTrue(settled_pl[f'net_odds_{market}'].eq(0).all())

if __name__ == '__main__':
    unittest.main()
