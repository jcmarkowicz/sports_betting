import unittest
import numpy as np
import pandas as pd
from ufc_betting.Models.RiskManagement.validation_bayesian_search_logit import portable_kelly_input
from ufc_betting.Models.RiskManagement.delayed_walk_forward_logit import kelly_simulation_input
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly


class PortableKellyTests(unittest.TestCase):
    def test_all_nine_combinations_match_explicit_parameter_transfer(self):
        markets = ['open','close1','close2']
        params, original, portable = {}, {}, {}
        for i, market in enumerate(markets):
            params[market] = dict(mdd=.2+i*.1,N=250+i*100,mdd_parlay=.2+i*.05,
                N_parlay=800+i*100,z=.2+i*.2,moneyline_min_american=-300)
            data = pd.DataFrame(dict(date=['2025-01-01']*3,fighter_red=['a','b','c'],
                fighter_blue=['d','e','f'],winner=[1,0,1],pred_winner=[1]*3,
                proba_red=[.8,.75,.7],proba_blue=[.2,.25,.3],
                risk_red_proba=[.78,.73,.68],prediction_se=[.02]*3,
                open_red=[-150]*3,open_blue=[130]*3))
            for side in ('blue','red'):
                data[f'dec_{market}_{side}'] = 1.9+i*.1
                data[f'dec_fair_{market}_{side}'] = 2.+i*.1
            for k,v in params[market].items():data[f'best_{k}']=v
            original[market] = kelly_simulation_input(data,market)
            portable[market] = portable_kelly_input(*original[market],params[market])
            self.assertNotIn('bet_real_decimal_red',original[market][0])
        for target in markets:
            for source in markets:
                with self.subTest(target=target,source=source):
                    actual, actual_parlay = simulate_kelly(portable[target][0],**portable[source][1])
                    expected_data = original[target][0].copy()
                    for k,v in params[source].items():expected_data[f'best_{k}']=v
                    expected, expected_parlay = simulate_kelly(expected_data,**original[target][1])
                    np.testing.assert_allclose(actual.bankroll_postevent,expected.bankroll_postevent)
                    np.testing.assert_allclose(actual_parlay.fstar_parlay.to_numpy(float),expected_parlay.fstar_parlay.to_numpy(float))
                    np.testing.assert_allclose(actual.real_odds_red,1.9+markets.index(target)*.1)


if __name__ == '__main__':unittest.main()
