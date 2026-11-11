import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from ufc_betting.Models.Search.rolling_brier import rolling_brier_shortfall
from ufc_betting.Models.Search.frozen_betting import FrozenBettingSearch, JointDecayBettingSearch
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.BettingStrategy.betting_params import BettingParams
import test_joint_kelly_decay_search as fixtures


class RollingBrierPenaltyTests(unittest.TestCase):
    def test_monotonic_signed_gap(self):
        dates = pd.date_range('2025-01-01', periods=4)
        values = [rolling_brier_shortfall(dates, [gap]*4, window_bets=2)['brier_mean_shortfall']
                  for gap in [.02, .01, .005, 0, -.01]]
        np.testing.assert_allclose(values, [0, 0, .005, .01, .02], atol=1e-15)

    def test_whole_events_and_minimum(self):
        dates = pd.to_datetime(['2025-01-01']*2 + ['2025-02-01', '2025-03-01'])
        result = rolling_brier_shortfall(dates, [.1, .1, -.1, -.1], window_bets=2)
        self.assertEqual(result['brier_window_count'], 3)
        self.assertAlmostEqual(result['brier_mean_gap'], (.1 + .1/3 - .1)/3)
        self.assertAlmostEqual(result['brier_mean_shortfall'], .11/3)
        with self.assertRaisesRegex(ValueError, 'at least'):
            rolling_brier_shortfall(dates, [0]*4, window_bets=5)

    def test_search_penalty_applied_and_reported(self):
        frame = fixtures.JointKellyDecayTests().frame()
        params = dict(mdd=.2, N=100, mdd_parlay=.2, N_parlay=100, z=0., moneyline_min_american=-600)
        for decay in (False, True):
            extra = dict(use_decay=True, opening_params=BettingParams.from_value(params)) if decay else {}
            market = 'close1' if decay else 'open'
            suggestions = dict(params)
            if decay:
                suggestions.update({'log1p_' + name: 0. for name in JointDecayBettingSearch.rate_names})
            with patch.object(JointDecayBettingSearch if decay else FrozenBettingSearch, 'suggest',
                              side_effect=lambda trial: {key: (trial.suggest_int(key, value, value) if isinstance(value, int) else trial.suggest_float(key, value, value))
                                                         for key, value in suggestions.items()}):
                # Exercise objective directly: mocked suggestions do not populate best_params.
                cls = JointDecayBettingSearch if decay else FrozenBettingSearch
                kwargs = dict(opening_params=extra['opening_params']) if decay else {}
                search = cls(frame, market, k=2, brier_penalty=2., brier_target_gap=.5, brier_window_bets=2, **kwargs)
                study = search.run(n_trials=1)
                attrs = study.best_trial.user_attrs
                self.assertGreater(attrs['brier_penalty_amount'], 0)
                self.assertAlmostEqual(study.best_value, attrs['base_objective_score'] - attrs['brier_penalty_amount'])
                self.assertGreaterEqual(attrs['brier_selected_count'], 2)
        result = optimize_kelly_params(validation_data={'open': frame}, n_trials=1, k=2,
                                      brier_penalty=1., brier_target_gap=.5, brier_window_bets=2)
        self.assertIn('brier_penalty_amount', result['objective_breakdown'])
