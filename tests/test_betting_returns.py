import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.Models.Search.frozen_betting import FrozenBettingSearch
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate_bankroll_paths
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator
from ufc_betting.BettingStrategy.MonteCarlo.schedule_sampling import sample_schedule
from ufc_betting.Models.RiskManagement.delayed_bayesian_search import (
    FrozenBettingSearch as LegacyFrozenBettingSearch,
)


class BettingReturnCalculatorTests(unittest.TestCase):
    def test_exhausted_paths_freeze_and_families_stop_betting(self):
        events = pd.DataFrame({
            'event_return': [-1.2, .5, -1., -.2],
            'moneyline_return': [-.8, .3, -.6, -.1],
            'parlay_return': [-.4, .2, -.4, -.1],
        })
        result = simulate_bankroll_paths(events, np.array([
            [0, 1, 0], [1, 0, 1], [2, 1, 0], [3, 1, 1],
        ]), initial_bankroll=500)
        np.testing.assert_allclose(result['bankroll'], [
            [500, -100, -100, -100],
            [500, 750, -150, -150],
            [500, 0, 0, 0],
            [500, 400, 600, 900],
        ])
        np.testing.assert_allclose(result['terminal_returns'], [-1.2, -1.3, -1., .8])
        for family in ('moneyline', 'parlay'):
            cumulative = result[f'{family}_cumulative_returns']
            np.testing.assert_allclose(cumulative[:3, -1], cumulative[:3, -2])
        np.testing.assert_allclose(
            result['moneyline_cumulative_returns'] + result['parlay_cumulative_returns'],
            result['bankroll'][:, 1:] / 500 - 1,
        )

    def test_nonfinite_event_returns_are_still_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite'):
                simulate_bankroll_paths(pd.DataFrame({'event_return': [value]}),
                                        np.array([[0]]))

    def predictions(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=8).repeat(3),
            'proba_red': rng.uniform(.2, .9, 24),
            'risk_red_proba': rng.uniform(.2, .9, 24),
            'prediction_se': rng.uniform(.01, .1, 24),
            'winner': rng.integers(0, 2, 24),
        })
        for kind in ('open', 'close1', 'close2'):
            for side in ('blue', 'red'):
                frame[f'dec_{kind}_{side}'] = rng.uniform(1.2, 4, 24)
                frame[f'dec_fair_{kind}_{side}'] = 2.
        return frame

    def test_event_returns_and_paths_match_frozen_legacy(self):
        frames = [self.predictions(), self.predictions().iloc[::3].copy()]
        frames.append(self.predictions().assign(proba_red=.5, risk_red_proba=.5))
        for frame in frames:
            for odds_type in ('open', 'close1', 'close2'):
                for z in (0., .65, 100.):
                    with self.subTest(rows=len(frame), odds_type=odds_type, z=z):
                        params = dict(mdd=.3, N=250, mdd_parlay=.3, N_parlay=500,
                                      z=z, moneyline_min_american=-300)
                        search = FrozenBettingSearch(frame, odds_type, k=20)
                        legacy = LegacyFrozenBettingSearch(frame, odds_type, k=20)
                        self.assertIsInstance(search.bets_engine, BettingReturnCalculator)
                        # Compare settlement on identical schedules; the new sampler
                        # deliberately produces different draws from the old bootstrap.
                        legacy.path_indices = search.path_indices
                        pd.testing.assert_frame_equal(search.event_returns(params),
                                                      legacy.event_returns(params))
                        for key, values in legacy.simulate(params).items():
                            np.testing.assert_allclose(search.simulate(params)[key], values)

    def test_run_uses_only_frozen_betting_parameters(self):
        search = FrozenBettingSearch(self.predictions(), k=20)
        study = search.run(n_trials=2)
        self.assertEqual(set(study.best_params), {
            'mdd', 'N', 'mdd_parlay', 'N_parlay', 'z', 'moneyline_min_american'
        })
        self.assertTrue(np.isfinite(study.best_value))
        events = search.event_returns(study.best_params)
        np.testing.assert_allclose(events.event_return,
                                   events.moneyline_return + events.parlay_return)
        self.assertFalse(hasattr(search.bets_engine, 'fit_logistic_model'))
        self.assertFalse(hasattr(search.bets_engine, 'run_two_stage'))

    def test_replacement_schedules_are_reproducible_and_reused(self):
        frame = self.predictions()
        target = 'ufc_betting.Models.Search.frozen_betting.sample_schedule'
        with patch(target, wraps=sample_schedule) as sampler:
            search = FrozenBettingSearch(frame, k=4, seed=42)
            self.assertEqual(sampler.call_count, 4)
            for call in sampler.call_args_list:
                self.assertTrue(call.kwargs['replace'])
            dates = pd.DatetimeIndex(sorted(frame.date.unique()))
            for path, call in zip(search.path_indices, sampler.call_args_list):
                sampled = sample_schedule(*call.args, **call.kwargs)
                np.testing.assert_array_equal(dates[path], sampled.source_date)
            original = search.path_indices.copy()
            search.run(n_trials=2)
            self.assertEqual(sampler.call_count, 4)
            np.testing.assert_array_equal(search.path_indices, original)
        repeated = FrozenBettingSearch(frame, k=4, seed=42)
        np.testing.assert_array_equal(search.path_indices, repeated.path_indices)
        self.assertTrue(any(len(set(path)) < len(path) for path in original))

    def test_without_replacement_keeps_each_event_once(self):
        frame = self.predictions().assign(title_fight=0, event_type='fight_night')
        frame.loc[0, 'title_fight'] = 1
        search = FrozenBettingSearch(frame, k=4, sample_with_replacement=False)
        for path in search.path_indices:
            np.testing.assert_array_equal(np.sort(path), np.arange(frame.date.nunique()))


if __name__ == '__main__':
    unittest.main()
