import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly


class BettingParamsTests(unittest.TestCase):
    def test_combined_json_round_trip_recreates_simulation(self):
        opening = BettingParams(mdd=.35, N=400)
        rates = dict(k_down=1., k_up=2., k_parlay_down=3., k_parlay_up=4.)
        params = BettingParams(mdd=.2, N=600, min_moneyline=-300).with_decay(rates, opening)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'combined.json'
            params.save(path)
            contents = json.loads(path.read_text())
            self.assertEqual(contents['decay_rates'], rates)
            self.assertEqual(contents['opening_params']['mdd'], .35)
            self.assertNotIn('Infinity', path.read_text())
            restored = BettingParams.load(path)
        self.assertEqual(restored, params)
        self.assertEqual(restored.to_simulation_kwargs(), params.to_simulation_kwargs())
        rates['k_down'] = 99
        self.assertEqual(params.decay_rates['k_down'], 1.)

    def test_invalid_controls_and_decay_are_rejected(self):
        for values in (dict(decay_rates={'k_down': 1}),
                       dict(decay_rates={'k_down': -1, 'k_up': 0}),
                       dict(decay_rates={'k_down': float('nan'), 'k_up': 0}),
                       dict(min_moneyline=100, moneyline_max_american=-100),
                       dict(N=1), dict(N=10.5)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BettingParams(**values)

    def test_upper_only_bound_is_preserved(self):
        params = BettingParams.from_value(dict(moneyline_min_american=None, moneyline_max_american=200))
        self.assertEqual(params.to_policy()['moneyline_max_american'], 200)
        self.assertEqual(params.to_simulation_kwargs()['moneyline_american_bounds'], (-np.inf, 200))

    def test_direct_object_matches_simulation_kwargs(self):
        data = pd.DataFrame(dict(
            date=['2026-01-01'] * 2, fighter_red=['a', 'b'], fighter_blue=['c', 'd'],
            winner=[1, 0], pred_winner=[1, 1], proba_red=[.8, .7], proba_blue=[.2, .3],
            open_red=[-110, -110], open_blue=[110, 110],
            dec_open_red=[1.9, 1.9], dec_open_blue=[2.1, 2.1],
            dec_fair_open_red=[2., 2.], dec_fair_open_blue=[2., 2.],
        ))
        controls = BettingParams(z=None, min_moneyline=-300, mdd=.2, N=500)
        prepared = KellySimInput(data, betting_params=controls, calc_parlay=True)
        actual, actual_parlays = simulate_kelly(data, **prepared.kwargs)
        kwargs = {**KellySimInput.COLUMN_DEFAULTS, 'calc_parlay': True, 'betting_params': controls}
        expected, expected_parlays = simulate_kelly(data, **kwargs)
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(actual_parlays, expected_parlays)
        self.assertEqual(prepared.betting_params, controls)


if __name__ == '__main__':
    unittest.main()
