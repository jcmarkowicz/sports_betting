import ast
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput
from ufc_betting.BettingStrategy.kelly_utils import kelly_simulation_input, portable_kelly_input


class KellySimInputTests(unittest.TestCase):
    def test_moneyline_decay_backtest_is_prepared_by_generic_input(self):
        opening = self.predictions()
        rates = {'k_down': 2., 'k_up': 3.}
        inputs = KellySimInput.from_predictions(
            self.predictions(), self.params(), decay_rates=rates,
            opening_data=opening, opening_params=self.params(), calc_parlay=False,
        )
        self.assertEqual(inputs.kwargs['decay_rates'], rates)
        self.assertEqual(inputs.kwargs['opening_params'], self.params())
        pd.testing.assert_frame_equal(inputs.kwargs['opening_data'], opening)
        self.assertFalse(inputs.kwargs['calc_parlay'])
        self.assertFalse(inputs.kwargs['use_walk_forward_params'])
        self.assertIsNone(inputs.kwargs['bankroll_floor'])

    def test_saved_parameters_recreate_simulation_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'open.json'
            path.write_text(json.dumps(self.params()), encoding='utf-8')
            loaded = KellySimInput.from_params_file(self.predictions(), path)
            expected = KellySimInput.from_predictions(self.predictions(), self.params())
            pd.testing.assert_frame_equal(loaded.data, expected.data)
            self.assertEqual(loaded.kwargs, expected.kwargs)
            path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Parameter file'):
                KellySimInput.from_params_file(self.predictions(), path)

    def predictions(self):
        return pd.DataFrame(dict(
            date=['2026-01-01'], fighter_red=['red'], fighter_blue=['blue'],
            winner=[1], proba_red=[.8], proba_blue=[.2], pred_winner=[1],
            risk_red_proba=[.75], prediction_se=[.03],
            dec_open_red=[1.8], dec_open_blue=[2.2],
            dec_fair_open_red=[2.], dec_fair_open_blue=[2.],
        ))

    def params(self):
        return dict(mdd=.3, N=250, mdd_parlay=.25, N_parlay=500,
                    z=.2, moneyline_min_american=-300)

    def test_search_adapter_matches_previous_inputs(self):
        predictions, params = self.predictions(), self.params()
        prepared = KellySimInput.from_predictions(predictions, params)
        legacy = predictions.assign(**{f'best_{k}': v for k, v in params.items()})
        data, kwargs = portable_kelly_input(*kelly_simulation_input(legacy), params)
        pd.testing.assert_frame_equal(prepared.data, data)
        for key, value in kwargs.items():
            self.assertEqual(prepared.kwargs[key], value)
        self.assertNotIn('best_mdd', predictions)

    def test_all_simulator_arguments_are_supported(self):
        # Inspect the signature without importing unrelated backtest dependencies.
        path = Path(__file__).parents[1] / 'src/ufc_betting/BettingStrategy/Backtest/backtest_functions.py'
        module = ast.parse(path.read_text())
        function = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'simulate_kelly')
        expected = {arg.arg for arg in function.args.args[1:]}
        self.assertEqual(expected, set(KellySimInput.DEFAULTS) | set(KellySimInput.COLUMN_DEFAULTS))

    def test_optional_controls_and_copies(self):
        opening = self.predictions()
        prepared = KellySimInput.from_predictions(
            self.predictions(), self.params(), calc_heavy_favorite_parlay=True,
            heavy_parlay_mdd=.15, heavy_favorite_cutoff=-250, heavy_parlay_max_legs=3,
            decay_rates={'k_down': .2, 'k_up': .4}, opening_data=opening,
            opening_params=self.params(),
        )
        self.assertTrue(prepared.kwargs['calc_heavy_favorite_parlay'])
        self.assertEqual(prepared.kwargs['heavy_parlay_mdd'], .15)
        data, kwargs = prepared.to_simulation_args()
        data.loc[0, 'proba_red'] = .1
        kwargs['decay_rates']['k_down'] = 9
        kwargs['opening_data'].loc[0, 'proba_red'] = .1
        self.assertEqual(prepared.data.loc[0, 'proba_red'], .8)
        self.assertEqual(prepared.kwargs['decay_rates']['k_down'], .2)
        self.assertEqual(opening.loc[0, 'proba_red'], .8)

    def test_invalid_arguments_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'Unknown Kelly'):
            KellySimInput(self.predictions(), typo_param=.3)
        with self.assertRaisesRegex(ValueError, 'Missing Kelly'):
            KellySimInput(pd.DataFrame())


if __name__ == '__main__':
    unittest.main()
