import unittest

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator


class PostDecayAdjustmentTests(unittest.TestCase):
    def caches(self):
        # stake, EV, realized profit, raw/conservative edge, probability, odds, side
        opening = np.array([[[.9, .4, -1, .25, .25, .7, 2., 1],
                             [.1, .4, -1, .25, .25, .7, 4., 1]]])
        closing = opening.copy()
        closing[0, :, 0] = [.1, .9]
        closing[0, :, 4] = .2
        return opening, closing

    def test_blended_exposure_is_capped_after_z(self):
        opening, closing = self.caches()
        result = Calculator.decay_scenario_allocations(
            opening, closing, [[2., 2.], [2., 2.]], [0, 0],
            BettingParams(z=None), BettingParams(z=.5), dict(k_down=0, k_up=1000))
        # Weights ~[1,0] select [.9,.9], then z multiplies by .8, then cap
        # reduces total 1.44 to 1. Pre-blend caps would incorrectly leave 1.44.
        np.testing.assert_allclose(result['moneyline_stakes'], [[.5, .5]])

    def test_z_is_applied_once_after_blend_using_closing_policy(self):
        opening, closing = self.caches()
        opening[:, :, 0] *= .1
        closing[:, :, 0] *= .1
        opening[:, :, 4] = 0.  # Opening z must not suppress raw opening stakes.
        result = Calculator.decay_scenario_allocations(
            opening, closing, [[2., 2.], [2., 2.]], [0, 0],
            BettingParams(z=10), BettingParams(z=.5), dict(k_down=0, k_up=1))
        # Weights [1,.5], raw stakes [.09,.05], closing correction = .2/.25.
        np.testing.assert_allclose(result['moneyline_stakes'], [[.072, .04]])

    def test_joint_moneyline_and_parlay_share_final_cap(self):
        opening, closing = self.caches()
        result = Calculator.decay_scenario_allocations(
            opening, closing, [[2., 2.], [2., 2.]], [0, 0],
            BettingParams(z=None), BettingParams(z=.5),
            dict(k_down=0, k_up=1000, k_parlay_down=0, k_parlay_up=0), True)
        ticket = result['parlay_stakes'] * result['multiplier']
        exposure = result['moneyline_stakes'].sum(axis=1) + ticket
        np.testing.assert_allclose(exposure, 1.)
        self.assertGreater(ticket[0], 0.)
        losses = (result['moneyline_stakes'] * closing[:, :, 2]).sum(axis=1) + ticket * result['parlay_profit']
        np.testing.assert_allclose(losses, -1.)

    def test_search_backtest_and_monte_carlo_use_same_decay_order(self):
        from test_search_exponential_decay import SearchDecayTests
        from ufc_betting.Models.Search.exponential_decay import evaluate_decay_portfolio
        from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
        search = SearchDecayTests().make_search()
        search.metadata['opening_params']['z'] = .1
        search.metadata['closing_params']['z'] = 2.
        rates = dict(k_down=2., k_up=3., k_parlay_down=1., k_parlay_up=4.)
        expected = evaluate_decay_portfolio(search, rates, include_parlay=True)
        opening = BettingParams.from_value(search.metadata['opening_params'])
        closing = BettingParams.from_value(search.metadata['closing_params']).with_decay(rates, opening)
        data = search.frame.copy()
        data = data.drop(columns=[c for c in data if c.startswith('reference_')])
        data['proba_se'] = data.prediction_se
        for side in ('red', 'blue'):
            data[f'dec_open_{side}'] = 2.
            for stage in ('close1', 'close2'):
                data[f'recorded_{stage}_{side}'] = data[f'dec_close1_{side}']
        mc_stakes, mc_returns = [], []
        for _, group in data.groupby('date', sort=True):
            bets, _ = Calculator.calculate_market_event_returns(group, dict(close1=closing, close2=closing))
            mc_stakes.extend(bets.choice_fstar)
            mc_returns.append(bets.event_payout_money_line.iloc[0] + bets.parlay_net.iloc[0])
        np.testing.assert_allclose(mc_stakes, expected['stakes'])
        np.testing.assert_allclose(mc_returns, expected['events'].event_return)
        backtest, _ = simulate_kelly(data, prob_cols=['proba_blue', 'proba_red'],
            fair_decimal_cols=['dec_fair_close1_blue', 'dec_fair_close1_red'],
            real_decimal_cols=['dec_close1_blue', 'dec_close1_red'],
            pred_winner_col='pred_winner', prediction_se_col='prediction_se',
            init_bankroll=500, calc_parlay=True, bankroll_floor=None,
            opening_data=data, **closing.to_simulation_kwargs())
        np.testing.assert_allclose(backtest.choice_fstar, expected['stakes'])
        np.testing.assert_allclose(backtest.groupby('date').bankroll_pct_change.first() / 100,
                                   expected['events'].event_return)


if __name__ == '__main__':
    unittest.main()
