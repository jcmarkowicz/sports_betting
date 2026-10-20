import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.MonteCarlo.market_policy import (
    load_predictor, predict_sampled_prices, settle_market_event, nearest_market,
)
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator


class MarketPolicyTests(unittest.TestCase):
    def test_vectorized_cache_matches_existing_scalar_sizing(self):
        rng = np.random.default_rng(42)
        count = 150
        sides = rng.integers(0, 2, count)
        red = rng.uniform(.05, .95, count)
        data = pd.DataFrame(dict(pred_winner=sides, winner=rng.integers(0, 2, count),
            proba_red=red, proba_blue=1-red, proba_se=.02,
            dec_fair_close1_red=rng.uniform(1.05, 8, count),
            dec_fair_close1_blue=rng.uniform(1.05, 8, count),
            dec_close1_red=rng.uniform(1.05, 8, count),
            dec_close1_blue=rng.uniform(1.05, 8, count)))
        from ufc_betting.BettingStrategy.kelly_scaling import decimal_to_american, expected_value
        p = np.where(sides == 1, red, 1-red)
        fair = data[['dec_fair_close1_blue', 'dec_fair_close1_red']].to_numpy()[np.arange(count), sides]
        offered = data[['dec_close1_blue', 'dec_close1_red']].to_numpy()[np.arange(count), sides]
        bets = pd.DataFrame(dict(choice_proba=p, fair_odds=fair, choice_real_odds=offered,
            choice_ev=expected_value(p, offered),
            choice_american_odds=[round(decimal_to_american(x)) for x in offered],
            pred_win=sides == data.winner))
        for params in (BettingParams(), BettingParams(N=40, mdd=.8, z=None)):
            expected = Calculator._moneyline_returns(bets, params.to_policy())
            actual = Calculator.calculate_scenario_cache(data, params)
            np.testing.assert_allclose(actual[:, 0], expected.fstar, rtol=0, atol=0)
            np.testing.assert_allclose(actual[:, 0] * actual[:, 2], expected.pct_return.fillna(0))

    def frame(self):
        return pd.DataFrame(dict(
            date=['2025-01-01'] * 2, winner=[1, 1], pred_winner=[1, 1],
            proba_red=[.75, .7], proba_blue=[.25, .3], proba_se=[.01, .02],
            dec_open_red=[2., 2.], dec_open_blue=[2., 2.],
            dec_close1_red=[2.1, 3.8], dec_close1_blue=[2., 2.],
            dec_fair_close1_red=[2.2, 4.], dec_fair_close1_blue=[2., 2.],
            recorded_close1_red=[2., 2.], recorded_close1_blue=[2., 2.],
            recorded_close2_red=[4., 4.], recorded_close2_blue=[4., 4.],
        ))

    def test_ticket_uses_product_distance_not_individual_leg_policy(self):
        # Singles select close1/close2, but 2.1*3.8 is nearer 2*2 than 4*4.
        policies = {'close1': BettingParams(z=None),
                    'close2': BettingParams(z=None, N_parlay=50, mdd_parlay=.8)}
        bets, tickets = settle_market_event(self.frame(), policies, use_decay=False)
        self.assertEqual(bets.betting_policy.tolist(), ['close1', 'close2'])
        self.assertEqual(tickets.betting_policy.tolist(), ['close1'])
        self.assertAlmostEqual(tickets.sampled_combined_odds.iloc[0], 7.98)
        changed = self.frame()
        changed['dec_close1_red'] = [3., 3.8]
        _, tickets2 = settle_market_event(changed, policies, use_decay=False)
        self.assertEqual(tickets2.betting_policy.tolist(), ['close2'])
        self.assertEqual(str(nearest_market(10., 4., 16.)), 'close1')

    def test_identical_policies_match_shared_calculator(self):
        data = self.frame()
        params = BettingParams(z=.5)
        bets, tickets = settle_market_event(data, dict(close1=params, close2=params), use_decay=False)
        cache = Calculator.calculate_scenario_cache(data, params)[None]
        expected = Calculator.scenario_allocations(cache, data.winner.to_numpy(), params, True)
        np.testing.assert_allclose(bets.choice_fstar, expected['moneyline_stakes'][0])
        self.assertAlmostEqual(tickets.fstar_parlay.iloc[0],
                               expected['parlay_stakes'][0] * expected['multiplier'][0])
        self.assertLessEqual(bets.choice_fstar.sum() + tickets.fstar_parlay.sum(), 1.)

    def test_ticket_uncertainty_uses_its_selected_policy(self):
        policies = {'close1': BettingParams(z=0), 'close2': BettingParams(z=100)}
        data = self.frame()
        bets, tickets = settle_market_event(data, policies, use_decay=False)
        cache = Calculator.calculate_scenario_cache(data, policies['close1'])
        allocations = Calculator.scenario_allocations(cache[None], data.winner.to_numpy(), policies['close1'], True)
        ticket = allocations['parlay_stakes'][0]
        # close2 suppresses only the second single's edge. Both parlay legs
        # use close1 (z=0), because combined sampled odds are nearer close1.
        # Per-bet scaling leaves the close1 single/ticket unchanged and
        # eliminates only the close2 single, before the shared exposure cap.
        cap = 1 / max(cache[0, 0] + ticket, 1.)
        self.assertAlmostEqual(tickets.fstar_parlay.iloc[0], ticket * cap)
        np.testing.assert_allclose(bets.choice_fstar, [cache[0, 0] * cap, 0.])

    def test_disabled_policy_stays_disabled_in_mixed_card(self):
        selected = np.array([[[.2, .1, 1., 0., 0., .7, 2., 1.],
                              [.3, .1, -1., .2, .1, .7, 2., 1.]]])
        raw = dict(moneyline_stakes=np.array([[.2, .3]]), parlay_stakes=np.array([.1]),
                   legs=np.array([[0, 1]]))
        result = Calculator.finalize_scenario_allocations(
            selected, raw, True, single_enabled=[[False, True]], ticket_enabled=False)
        np.testing.assert_allclose(result['moneyline_stakes'], [[.2, .15]])
        np.testing.assert_allclose(result['parlay_stakes'] * result['multiplier'], [.1])

    def test_blue_choices_compare_blue_prices(self):
        data = self.frame().assign(pred_winner=0, proba_red=.2, proba_blue=.8)
        data['dec_close1_blue'] = [3.8, 3.9]
        params = BettingParams(z=None)
        bets, tickets = settle_market_event(data, dict(close1=params, close2=params), use_decay=False)
        self.assertEqual(bets.betting_policy.tolist(), ['close2', 'close2'])
        self.assertEqual(tickets.betting_policy.tolist(), ['close2'])
        self.assertAlmostEqual(tickets.sampled_combined_odds.iloc[0], 3.8 * 3.9)

    def test_zero_decay_uses_opening_stakes_then_closing_z(self):
        opening = BettingParams(N=500, N_parlay=100, z=.2)
        closing = BettingParams(N=50, z=.5, opening_params=opening,
                                decay_rates=dict(k_down=0, k_up=0, k_parlay_down=0, k_parlay_up=0))
        actual, tickets = settle_market_event(self.frame(), dict(close1=closing, close2=closing))
        post_blend_policy = replace(opening, z=closing.z)
        expected, expected_ticket = settle_market_event(self.frame(), dict(close1=post_blend_policy, close2=post_blend_policy), use_decay=False)
        np.testing.assert_allclose(actual.choice_fstar, expected.choice_fstar)
        np.testing.assert_allclose(tickets.fstar_parlay, expected_ticket.fstar_parlay)

    def test_prediction_uses_selected_preprocessing_and_fresh_odds(self):
        class Scaler:
            feature_names_in_ = ['proba_fair_close2_diff']
            def transform(self, frame):
                self.observed = frame.copy()
                return frame.to_numpy() * 2
        class Encoder:
            feature_names_in_ = ['category']
        scaler, encoder = Scaler(), Encoder()
        model = SimpleNamespace(model=SimpleNamespace(exog_names=['const', 'proba_fair_close2_diff', 'category']))
        predictor = load_predictor(model, 'close2', {'close2': scaler}, {'close2': encoder}, object())
        data = self.frame().assign(category=1, proba_fair_close2_diff=999.)
        prices = np.array([[1.8, 2.5], [2.4, 1.9]])
        def prediction(inputs, *args):
            self.assertTrue((abs(scaler.observed.to_numpy()) <= 1).all())
            np.testing.assert_allclose(inputs.proba_fair_close2_diff, scaler.observed.iloc[:, 0] * 2)
            self.assertTrue(inputs.category.eq(7).all())
            return pd.DataFrame(dict(proba_red=[.6,.7], pred_winner=[1,1], proba_se=[.01,.01]))
        with patch('ufc_betting.Models.LogisticRegression.train_test_builder.TrainTestBuilder.encode_categorical',
                   return_value=pd.DataFrame({'category': [7,7]})) as encode, patch(
                   'ufc_betting.BettingStrategy.MonteCarlo.market_policy.predict_logit', side_effect=prediction):
            result = predict_sampled_prices(data, prices, predictor, 'close2')
        self.assertIs(encode.call_args.args[1], encoder)
        np.testing.assert_allclose(result.dec_close1_red, prices[:,0])
        np.testing.assert_allclose(result.dec_open_red, data.dec_open_red)
        self.assertTrue(data.proba_fair_close2_diff.eq(999).all())


if __name__ == '__main__':
    unittest.main()
