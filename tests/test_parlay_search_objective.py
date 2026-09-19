import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.Models.RiskManagement.bayesian_search import BayesianSearchLR
from ufc_betting.BettingStrategy.Backtest.backtest_functions import parlay_heavy_favorites


class ParlaySearchTests(unittest.TestCase):
    def test_heavy_mdd_changes_only_heavy_sizing(self):
        search = BayesianSearchLR.__new__(BayesianSearchLR)
        data = pd.DataFrame(dict(
            date=['2026-01-01'] * 2, id=[0, 1],
            choice_fighter_name=['a', 'b'], choice_fighter_bool=[1, 1],
            choice_proba=[.8, .95], choice_real_odds=[1.5, 1.4],
            choice_ev=[.2, .33], choice_american_odds=[-200, -250],
        ))
        winners = pd.DataFrame(dict(id=[0, 1], pred_win=[True, True]))
        params = dict(mdd_parlay=.25, N_parlay=1000,
                      heavy_favorite_cutoff=-200, heavy_parlay_max_legs=2)
        stakes = {}
        for strategy in ('top_ev', 'heavy_favorite'):
            stakes[strategy] = [search.parlay_returns(
                data, winners, {**params, 'heavy_parlay_mdd': mdd}, strategy,
            ).fstar.iloc[0] for mdd in (.15, .7)]
        self.assertEqual(*stakes['top_ev'])
        self.assertGreater(stakes['heavy_favorite'][1], stakes['heavy_favorite'][0])

    def test_average_return_counts_no_bet_events(self):
        search = BayesianSearchLR.__new__(BayesianSearchLR)
        search.score_type = 'average_return'
        self.assertAlmostEqual(search.pct_returns_score([.10, -.04, np.nan]), .02)
        self.assertEqual(search.pct_returns_score([]), 0.)

    def test_win_rate_pools_tickets_and_handles_no_bets(self):
        search = BayesianSearchLR.__new__(BayesianSearchLR)
        fold = dict(red_proba=[], risk_red_proba=[], se=[], winner_fold=[], val_idx=[])
        with patch.object(search, 'calculate_fight_returns', side_effect=[[True], [False, False, True]]):
            self.assertEqual(search.score_cached_parlay_win_rate([fold, fold], {}), .5)
        self.assertEqual(search.score_cached_parlay_win_rate([], {}), 0.)

    def test_heavy_selection_ignores_moneyline_bounds(self):
        search = BayesianSearchLR.__new__(BayesianSearchLR)
        data = pd.DataFrame(dict(
            date=['2026-01-01'] * 3, id=[0, 1, 2],
            choice_fighter_name=['a', 'b', 'c'], choice_fighter_bool=[1]*3,
            choice_proba=[.8, .95, .4], choice_real_odds=[1.5, 1.4, 2.],
            choice_ev=[.2, .33, -.2], choice_american_odds=[-200, -250, 100],
        ))
        winners = pd.DataFrame(dict(id=[0, 1, 2], pred_win=[True, True, False]))
        for cutoff in (-500, -200):
            result = search.parlay_returns(data, winners, dict(
                heavy_parlay_max_legs=2, moneyline_min_american=cutoff,
                heavy_favorite_cutoff=-200,
                mdd_parlay=None, N_parlay=200,
            ), strategy='heavy_favorite')
            self.assertEqual(result.selected_ids.iloc[0], [1, 0])
            self.assertGreater(result.pct_return.iloc[0], 0)
        backtest_data = data.assign(
            pred_winner=1, winner=1, fighter_red=['a', 'b', 'c'], fighter_blue='other',
        )
        profit, _, legs = parlay_heavy_favorites(
            backtest_data, 500, max_legs=2, favorite_cutoff=-200, parlay_mdd=None,
        )
        self.assertEqual(legs.index.tolist(), [1, 0])
        self.assertGreater(profit, 0)
        profit, _, legs = parlay_heavy_favorites(
            backtest_data, 500, max_legs=2, favorite_cutoff=-300, parlay_mdd=None,
        )
        self.assertTrue(legs.empty)
        self.assertEqual(profit, 0.)
        result = search.parlay_returns(data, winners, dict(
            heavy_parlay_max_legs=2, heavy_favorite_cutoff=-300,
            mdd_parlay=None, N_parlay=200,
        ), strategy='heavy_favorite')
        self.assertEqual(result.selected_ids.iloc[0], [])

    def test_zero_stake_parlays_excluded_from_outcomes(self):
        search = BayesianSearchLR.__new__(BayesianSearchLR)
        search.fair_odds = pd.DataFrame([[5., 1.5]] * 2)
        search.real_odds = pd.DataFrame([[5., 1.5]] * 2)
        search.dates = ['2026-01-01'] * 2
        args = dict(red_proba=np.array([.8, .8]), risk_red_proba=np.array([.8, .8]),
                    se=np.array([.1, .1]), winner_fold=[1, 1], val_idx=[0, 1],
                    return_parlay_outcomes=True)
        params = dict(mdd=.3, N=200, mdd_parlay=.3, N_parlay=200,
                      heavy_parlay_max_legs=2, moneyline_min_american=-500,
                      heavy_favorite_cutoff=-200,
                      moneyline_max_american=500, z=0)
        self.assertEqual(search.calculate_fight_returns(**args, param=params), [True, True])
        self.assertEqual(search.calculate_fight_returns(**args, param={**params, 'z': 10}), [])
