import unittest
import numpy as np
import pandas as pd
from ufc_betting.Models.RiskManagement.exponential_decay_search import MoneylineDecaySearch,decay_weights


class DecayTests(unittest.TestCase):
    def test_direction_and_zero(self):
        w=decay_weights([2,2,2],[1.5,3,2],2,4)
        np.testing.assert_allclose(w,[np.exp(-2*abs(np.log(.75))),np.exp(-4*np.log(1.5)),1])
        np.testing.assert_array_equal(decay_weights([2,2],[1.5,3],0,0),[1,1])

    def test_search_reuses_paths_and_preserves_blend(self):
        frame=pd.DataFrame(dict(date=['2025-01-01']*2+['2025-01-08']*2,
            fighter_red=list('abcd'),fighter_blue=list('efgh'),winner=[1,0,1,1],pred_winner=[1]*4,
            proba_red=[.7]*4,proba_blue=[.3]*4,se=[.01]*4,
            dec_close1_red=[1.8,2.2,1.9,2.3],dec_close1_blue=[2.2]*4,
            dec_fair_close1_red=[2.]*4,dec_fair_close1_blue=[2.]*4))
        reference=frame[['date','fighter_red','fighter_blue']].copy()
        reference['dec_open_red']=2.;reference['dec_open_blue']=2.
        p=dict(mdd=.4,N=250,z=.5,min_moneyline=None)
        search=MoneylineDecaySearch(frame,reference,p,{**p,'mdd':.2},k=10)
        before=search.path_indices.copy()
        np.testing.assert_allclose(search.event_returns().event_return,search.event_returns(policy='opening').event_return)
        w=decay_weights(search.opening_prices,search.current_prices,2,3)
        stakes=w*search.opening_stakes+(1-w)*search.closing_stakes
        manual=np.bincount(search.event_ids,weights=stakes*search.unit_profit)
        np.testing.assert_allclose(search.event_returns(2,3).event_return,manual)
        result=search.run(n_trials=3)
        np.testing.assert_array_equal(search.path_indices,before)
        self.assertEqual(set(result['best_params']),{'k_down','k_up'})
        backtest,parlays=search.simulate_kelly(result)
        self.assertTrue(parlays.empty)
        event_returns=search.event_returns(**result['best_params']).event_return.to_numpy()
        np.testing.assert_allclose(backtest.groupby('date').bankroll_postevent.first(),
                                   search.initial_bankroll*np.cumprod(1+event_returns))
        np.testing.assert_allclose(backtest.groupby('date').fight_payout.sum(),
                                   backtest.groupby('date').event_payout_money_line.first())
        from unittest.mock import patch
        import matplotlib
        matplotlib.use('Agg')
        from ufc_betting.BettingStrategy.Backtest.backtest_plots import plot_backtest
        import matplotlib.pyplot as plt
        with patch.object(plt,'show'):
            plot_backtest(backtest,search.initial_bankroll)
        plt.close('all')
        from ufc_betting.Models.RiskManagement.parlay_decay_search import ParlayDecaySearch
        search.metadata['opening_params'].update(mdd_parlay=.3,N_parlay=800)
        search.metadata['closing_params'].update(mdd_parlay=.2,N_parlay=900)
        combined=ParlayDecaySearch(search)
        with self.assertRaises(ValueError):
            combined.simulate_kelly({'k_parlay_down':1.,'k_parlay_up':1.})
        np.testing.assert_array_equal(combined.path_indices,search.path_indices)
        combined_rates=combined.run(n_trials=3)
        events=combined_rates['event_results']
        rates=combined_rates['best_params']
        self.assertEqual(set(rates),{'k_down','k_up','k_parlay_down','k_parlay_up'})
        np.testing.assert_allclose(events.moneyline_return,search.event_returns(rates['k_down'],rates['k_up']).event_return)
        a=combined.evaluate(k_down=0,k_up=0)
        b=combined.evaluate(k_down=30,k_up=20)
        self.assertFalse(np.allclose(a['events'].moneyline_return,b['events'].moneyline_return))
        self.assertNotEqual(a['score'],b['score'])
        np.testing.assert_allclose(a['events'].parlay_return,b['events'].parlay_return)
        c=combined.evaluate(k_parlay_down=30,k_parlay_up=20)
        np.testing.assert_allclose(a['events'].moneyline_return,c['events'].moneyline_return)
        self.assertFalse(np.allclose(a['events'].parlay_return,c['events'].parlay_return))
        np.testing.assert_allclose(events.event_return,events.moneyline_return+events.parlay_return)
        plotted,tickets=combined.simulate_kelly(combined_rates)
        from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
        via_function, via_tickets = simulate_kelly(
            frame, prob_cols=['proba_blue','proba_red'],
            fair_decimal_cols=['dec_fair_close1_blue','dec_fair_close1_red'],
            real_decimal_cols=['dec_close1_blue','dec_close1_red'],
            pred_winner_col='pred_winner', prediction_se_col='se',
            init_bankroll=500, max_drawdown=.2, N=250, z=.5,
            parlay_mdd=.2, N_parlay=900, calc_parlay=True,
            opening_data=reference, opening_params=search.metadata['opening_params'],
            decay_rates=combined_rates)
        np.testing.assert_allclose(via_function.bankroll_postevent, plotted.bankroll_postevent)
        np.testing.assert_allclose(via_function.choice_fstar, plotted.choice_fstar)
        np.testing.assert_allclose(via_tickets.fstar_parlay, tickets.fstar_parlay)
        np.testing.assert_allclose(plotted.groupby('date').bankroll_postevent.first(),
            search.initial_bankroll*np.cumprod(1+events.event_return.to_numpy()))
        with patch.object(plt,'show'):plot_backtest(plotted,search.initial_bankroll)
        plt.close('all')
        self.assertFalse(tickets.empty)


if __name__=='__main__':unittest.main()
