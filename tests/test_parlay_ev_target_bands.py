import unittest
import numpy as np
import pandas as pd
from ufc_betting.Models.RiskManagement.parlay_ev_target_bands import evaluate_parlays
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly, run_per_bet_scaling
from ufc_betting.BettingStrategy.kelly_scaling import kelly_edge


class ParlayBandsTests(unittest.TestCase):
    def test_matches_full_backtest_including_shared_risk_adjustment(self):
        rng = np.random.default_rng(812)
        for won in (True, False):
            for count in (2, 8, 14):
                p = rng.uniform(.55, .95, count)
                fair = 1 / (p - .12)
                odds = fair * .96
                se = np.full(count, .04)
                winners = np.ones(count) if won else np.zeros(count)
                data = pd.DataFrame(dict(date=['2026-01-01']*count,
                    fighter_red=[str(i) for i in range(count)], fighter_blue=[str(i+50) for i in range(count)],
                    winner=winners, pred_winner=np.ones(count), red=p, blue=1-p,
                    fair_red=fair, fair_blue=1/(1-1/fair), odds_red=odds,
                    odds_blue=np.full(count, 2.), se=se, open_red=-150, open_blue=130))
                inputs = pd.DataFrame(dict(f_star_unscaled=[kelly_edge(a,b) for a,b in zip(p,fair)],
                    choice_proba=p, choice_fair_odds=fair, choice_real_odds=odds,
                    choice_ev=p*odds-1, choice_idx=np.ones(count), winner=winners))
                sized, _ = run_per_bet_scaling(inputs,.4,500,250)
                cache = np.column_stack([sized.f_star_scaled,p*odds-1,
                    np.where(winners==1,odds-1,-1),p-1/fair,
                    np.maximum(p-.5*se-1/fair,0),p,odds,np.ones(count)])[None,:,:]
                expected, realized, stake = evaluate_parlays(cache,data)
                results, tickets = simulate_kelly(data,
                    prob_cols=['blue','red'], fair_decimal_cols=['fair_blue','fair_red'],
                    real_decimal_cols=['odds_blue','odds_red'], pred_winner_col='pred_winner',
                    init_bankroll=500, max_drawdown=.4, N=250, parlay_mdd=.4,
                    N_parlay=1490, calc_parlay=True, z=.5, prediction_se_col='se')
                np.testing.assert_allclose(realized[0], results.regular_parlay_net.iloc[0]/500, atol=1e-12)
                np.testing.assert_allclose(stake[0], tickets.fstar_parlay.iloc[0], atol=1e-12)
                np.testing.assert_allclose(expected[0], tickets.fstar_parlay.iloc[0]*tickets.parlay_ev.iloc[0], atol=1e-12)

    def test_ties_and_nonpositive_ev_match_top_ev(self):
        group = pd.DataFrame(dict(winner=[1,0,1,1], fighter_red=list('abcd'),
            fighter_blue=list('efgh'), date=['2026-01-01']*4))
        cache = np.zeros((2,4,8))
        cache[:,:,0] = .1
        cache[:,:,3:5] = .1
        cache[:,:,5] = .6
        cache[0,:,6] = 2.
        cache[1,:,6] = 1.1
        cache[:,:,1] = cache[:,:,5]*cache[:,:,6]-1
        cache[:,:,7] = 1
        expected, realized, stake = evaluate_parlays(cache,group)
        self.assertEqual(stake[1],0)
        self.assertEqual(realized[1],0)
        self.assertEqual(expected[1],0)


if __name__ == '__main__':
    unittest.main()
