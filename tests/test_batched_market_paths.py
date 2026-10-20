import unittest

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.MonteCarlo.sampled_paths import simulate_sampled_predictions


def make_paths(n=4):
    rng = np.random.default_rng(29)
    paths = []
    for path_id in range(n):
        frames = []
        for event_id, size in enumerate((1, 2, 5, 2)):
            red = rng.uniform(.1, .9, size)
            frame = pd.DataFrame(dict(proba_red=red, risk_red_proba=red * .9,
                prediction_se=rng.uniform(0, .08, size), winner=rng.integers(0, 2, size),
                date=pd.Timestamp('2025-01-01') + pd.Timedelta(days=event_id),
                sampled_event_id=event_id, odds_type='intermediate', execution_odds_type='intermediate'))
            for prefix in ('dec_open', 'dec_close1', 'dec_fair_close1', 'recorded_close1', 'recorded_close2'):
                for side in ('blue', 'red'):
                    frame[f'{prefix}_{side}'] = rng.uniform(1.2, 4, size)
            frames.append(frame)
        paths.append(pd.concat(frames, ignore_index=True))
    return paths


class BatchedMarketPathTests(unittest.TestCase):
    def test_intermediate_execution_must_match_saved_sampled_prices(self):
        paths = make_paths(1)
        for side in ('red', 'blue'):
            paths[0][f'dec_intermediate_{side}'] = paths[0][f'dec_close1_{side}']
        paths[0].loc[0, 'dec_close1_red'] += .2
        with self.assertRaisesRegex(ValueError, 'Execution odds differ'):
            simulate_sampled_predictions(paths, 'intermediate', controls={'calc_parlay': True},
                policies=dict(close1=BettingParams(), close2=BettingParams()), use_decay=False,
                initial_bankroll=500, batch_size=1, progress_every=0, return_details=False)

    def test_batch_boundary_and_tied_prices_preserve_outputs(self):
        path = make_paths(1)[0]
        path = path.loc[path.sampled_event_id == 1].copy()
        path['proba_red'] = .8
        for side in ('red', 'blue'):
            for prefix in ('dec_close1', 'dec_fair_close1', 'recorded_close1', 'recorded_close2'):
                path[f'{prefix}_{side}'] = 2.
        policies = dict(close1=BettingParams(z=None), close2=BettingParams(z=10))
        options = dict(controls={'calc_parlay': True}, policies=policies,
            use_decay=True, initial_bankroll=500, batch_size=1, progress_every=0)
        expected = simulate_sampled_predictions([path], 'intermediate', return_details=True, **options)
        actual = simulate_sampled_predictions([path] * 1025, 'intermediate', return_details=False, **options)
        for a, b in zip(actual, expected[:3]):
            np.testing.assert_allclose(a, np.repeat(b, 1025, axis=0), rtol=1e-12, atol=1e-10)

    def test_matches_reference_for_mixed_policies_and_decay(self):
        opening = BettingParams(N=200, N_parlay=100, z=.3)
        rates = dict(k_down=.5, k_up=2., k_parlay_down=1., k_parlay_up=3.)
        paths = make_paths()
        paths.append(paths[0].copy())
        paths.append(paths[1].loc[paths[1].sampled_event_id < 2].copy())
        for decay in (False, True):
            for parlay in (False, True):
                for batch_size in (1, 3):
                    with self.subTest(decay=decay, parlay=parlay, batch_size=batch_size):
                        policies = dict(close1=BettingParams(z=None, opening_params=opening, decay_rates=rates),
                                        close2=BettingParams(N=30, N_parlay=50, z=.8, opening_params=opening, decay_rates=rates))
                        options = dict(controls={'calc_parlay': parlay}, policies=policies,
                            use_decay=decay, initial_bankroll=500, batch_size=batch_size, progress_every=0)
                        expected = simulate_sampled_predictions(paths, 'intermediate', return_details=True, **options)
                        actual = simulate_sampled_predictions(paths, 'intermediate', return_details=False, **options)
                        for actual_family, expected_family in zip(actual, expected[:3]):
                            for a, b in zip(actual_family, expected_family):
                                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
