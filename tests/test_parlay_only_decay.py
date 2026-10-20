import unittest
import numpy as np
from dataclasses import replace
import test_joint_kelly_decay_search as fixtures
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.kelly_scaling import scale_kelly_for_mdd
from ufc_betting.Models.Search.exponential_decay import MoneylineDecaySearch
from ufc_betting.Models.Search.parlay_decay import ParlayDecaySearch


class ParlayOnlyDecayTests(unittest.TestCase):
    def test_returns_match_independent_ticket_calculation(self):
        frame = fixtures.JointKellyDecayTests().frame()
        opening = BettingParams(mdd_parlay=.2, N_parlay=800)
        closing = BettingParams(mdd_parlay=.4, N_parlay=250, z=.4)
        engine = ParlayDecaySearch(MoneylineDecaySearch(
            frame, frame, opening, closing, odds_type='close1', k=5))
        for decay in (False, True):
            result = engine.evaluate_parlay_only(k_parlay_down=2., k_parlay_up=3.,
                                                policy='decay' if decay else 'closing')
            expected = []
            for _, group in engine.frame.groupby('date', sort=True):
                side = (group.proba_red.to_numpy() >= .5).astype(int)
                rows = np.arange(len(group))
                p = np.where(side == 1, group.proba_red, 1 - group.proba_red)
                odds = group[['dec_close1_blue', 'dec_close1_red']].to_numpy()[rows, side]
                fair = group[['dec_fair_close1_blue', 'dec_fair_close1_red']].to_numpy()[rows, side]
                old = group[['dec_open_blue', 'dec_open_red']].to_numpy()[rows, side]
                ids = np.argsort(-(p * odds - 1), kind='stable')[:2]
                ticket_p, ticket_odds = p[ids].prod(), odds[ids].prod()
                full = max((ticket_p * ticket_odds - 1) / (ticket_odds - 1), 0.)
                stake = scale_kelly_for_mdd(ticket_p, ticket_odds, full, closing.N_parlay, closing.mdd_parlay)
                if decay:
                    other = scale_kelly_for_mdd(ticket_p, ticket_odds, full, opening.N_parlay, opening.mdd_parlay)
                    move = np.log(odds[ids] / old[ids])
                    weight = np.exp(-2 * np.maximum(-move, 0).sum() - 3 * np.maximum(move, 0).sum())
                    stake = weight * other + (1 - weight) * stake
                raw = np.maximum(p[ids] - 1 / fair[ids], 0).sum()
                conservative = np.maximum(p[ids] - closing.z * group.prediction_se.to_numpy()[ids] - 1 / fair[ids], 0).sum()
                uncertainty = min(1., conservative / raw) if raw > 0 else 0.
                stake = min(stake * uncertainty, 1.)
                won = (side[ids] == group.winner.to_numpy()[ids]).all()
                expected.append(stake * (ticket_odds - 1 if won else -1.))
            np.testing.assert_allclose(result['events'].parlay_return, expected)
            np.testing.assert_array_equal(result['events'].moneyline_return, 0.)
            np.testing.assert_allclose(result['events'].event_return, expected)

    def test_moneyline_controls_do_not_affect_parlay_only(self):
        frame = fixtures.JointKellyDecayTests().frame()
        baseline = BettingParams()
        outputs = []
        for controls in (baseline, replace(baseline, mdd=.8, N=2000)):
            engine = ParlayDecaySearch(MoneylineDecaySearch(frame, frame, controls, controls, k=4, odds_type='close1'))
            outputs.append(engine.evaluate_parlay_only(k_parlay_down=1., k_parlay_up=2.)['simulation']['bankroll'])
        np.testing.assert_allclose(*outputs)


if __name__ == '__main__':
    unittest.main()
