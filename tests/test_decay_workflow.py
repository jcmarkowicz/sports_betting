import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np

from test_frozen_bootstrap_paths import FrozenBootstrapPathTests as Fixtures
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.Models.Search.decay_workflow import optimize_decay_params
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate_bankroll_paths


class DecayWorkflowTests(unittest.TestCase):
    def test_saved_policies_cohorts_and_paths_feed_joint_search(self):
        frame = Fixtures().predictions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policies = {market: optimize_kelly_params(
                validation_data={market: frame},
                n_trials=1, k=3, bootstrap_mode='save', bootstrap_dir=root / 'paths',
                parallel=False,
            ) for market in ('open', 'close1', 'close2')}
            for market in ('open', 'close1'):
                policies[market]['betting_data'].to_csv(root / f'{market}.csv', index=False)
                (root / f'{market}.json').write_text(json.dumps(policies[market]['best_params']))
            kwargs = dict(
                opening_data=root / 'open.csv', closing_data=root / 'close1.csv',
                opening_params=root / 'open.json', closing_params=root / 'close1.json',
                bootstrap_dir=root / 'paths', n_trials=1, bootstrap_mode='auto',
            )
            with patch('ufc_betting.Models.Search.frozen_betting.sample_schedule',
                       side_effect=AssertionError('Must load saved schedules')):
                result = optimize_decay_params(**kwargs, output_dir=root / 'decay')
            self.assertEqual(set(result['best_params']),
                             {'k_down', 'k_up', 'k_parlay_down', 'k_parlay_up'})
            self.assertEqual(result['simulation']['bankroll'].shape[0], 3)
            self.assertEqual(result['kelly_sim_input'].kwargs['decay_rates'], result['best_params'])
            self.assertIn('allocation_weights', result)
            self.assertIn('return_comparison', result)
            combined = BettingParams.load(root / 'decay/betting_params.json')
            self.assertEqual(combined, result['betting_params'])
            self.assertEqual(BettingParams.load(root / 'close1.json'), combined)
            self.assertEqual(result['bootstrap_mode'], 'load')
            self.assertNotIn('decay_rates', json.loads((root / 'open.json').read_text()))
            prepared = KellySimInput.from_predictions(
                policies['close1']['betting_data'], combined, 'close1',
                opening_data=policies['open']['betting_data'],
            )
            backtest, _ = simulate_kelly(prepared.data, **prepared.kwargs)
            events = result['event_results']
            expected = simulate_bankroll_paths(events, np.arange(len(events))[None, :])['bankroll'][0, 1:]
            np.testing.assert_allclose(backtest.groupby('date').bankroll_postevent.first(), expected)
            no_decay = KellySimInput.from_predictions(
                policies['close1']['betting_data'],
                replace(combined, decay_rates=None, opening_params=None), 'close1',
            )
            ordinary, _ = simulate_kelly(no_decay.data, **no_decay.kwargs)
            baseline_events = result['baselines']['closing']['events']
            baseline_balances = simulate_bankroll_paths(
                baseline_events, np.arange(len(baseline_events))[None, :]
            )['bankroll'][0, 1:]
            np.testing.assert_allclose(ordinary.groupby('date').bankroll_postevent.first(), baseline_balances)
            comparison = result['return_comparison'].iloc[0]
            self.assertAlmostEqual(comparison.no_decay_total_return_pct,
                                   (baseline_balances[-1] / 500 - 1) * 100)
            breakdown = result['objective_breakdown']
            np.testing.assert_allclose(breakdown.objective_pct_points,
                breakdown.mean_terminal_return_pct - breakdown.penalty_pp)
            self.assertAlmostEqual(breakdown.iloc[0].objective_fraction, result['best_value'])
            wrong = dict(policies['close1']['best_params'], z=99.)
            with self.assertRaisesRegex(ValueError, 'do not match'):
                optimize_decay_params(**{**kwargs, 'closing_params': wrong})

    def test_auto_generates_when_absent_and_saves_close2_rates(self):
        frame = Fixtures().predictions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controls = BettingParams()
            controls.save(root / 'open.json')
            controls.save(root / 'close2.json')
            opening_original = (root / 'open.json').read_bytes()
            kwargs = dict(
                opening_data=frame, closing_data=frame,
                opening_params=root / 'open.json', closing_params=root / 'close2.json',
                bootstrap_dir=root / 'absent', bootstrap_mode='auto',
                odds_type='close2', k=3, n_trials=1,
            )
            result = optimize_decay_params(**kwargs)
            saved = BettingParams.load(root / 'close2.json')
            self.assertEqual(saved.decay_rates, result['best_params'])
            self.assertEqual(saved.to_policy(), controls.to_policy())
            self.assertEqual((root / 'open.json').read_bytes(), opening_original)
            self.assertEqual(result['bootstrap_mode'], 'generate')
            self.assertIsNone(result['bootstrap_dir'])
            self.assertFalse((root / 'absent').exists())
            self.assertEqual(result['simulation']['bankroll'].shape[0], 3)
            # An existing incomplete folder must not silently trigger regeneration.
            (root / 'absent/close2').mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                optimize_decay_params(**kwargs)


if __name__ == '__main__':
    unittest.main()
