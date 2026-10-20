import ast
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator
from ufc_betting.BettingStrategy.Convergence import realized_ev_target_bands as workflow
from ufc_betting.BettingStrategy.Convergence import model_input
from ufc_betting.BettingStrategy.kelly_scaling import scale_kelly_for_mdd


class ConvergenceCalculatorTests(unittest.TestCase):
    def test_target_path_updates_and_no_match_preserve_rng_behavior(self):
        rng = np.random.default_rng(42)
        balances = np.array([-50., 90., 500.])
        selected = workflow.sample_target_paths(balances, np.array([.1]), np.array([.2]), .2, 50, rng)
        np.testing.assert_allclose(selected['balances'], [-50, 90, 600])
        np.testing.assert_array_equal(selected['chosen_ids'], [0, 0, 0])
        state = rng.bit_generator.state
        unmatched = workflow.sample_target_paths(balances, np.array([.1]), np.array([.2]), .2, 100, rng)
        np.testing.assert_array_equal(unmatched['balances'], balances)
        np.testing.assert_array_equal(unmatched['chosen_ids'], [-1, -1, -1])
        self.assertEqual(rng.bit_generator.state, state)
        report = workflow.event_diagnostics('2026-01-01', 100, np.array([.1]), .2,
                                            unmatched['eligible'], unmatched['returns'])
        self.assertEqual(report['status'], 'no_sampled_match_no_bet')

    def test_event_evaluation_uses_combined_betting_params(self):
        frame = self.frame(2)
        opening = BettingParams(z=.5)
        closing = BettingParams(mdd=.2, z=.5)
        params = closing.with_decay(dict.fromkeys(['k_down', 'k_up', 'k_parlay_down', 'k_parlay_up'], 0.), opening)
        open_cache = Calculator.calculate_scenario_cache(frame, opening)
        close_cache = Calculator.calculate_scenario_cache(frame, closing)
        event = workflow.evaluate_event_scenarios(
            frame, frame[['dec_open_red', 'dec_open_blue']].to_numpy(),
            frame[['dec_close1_red', 'dec_close1_blue']].to_numpy(),
            np.repeat(close_cache[:, None, :], 101, axis=1), open_cache, params,
            samples=101, rng=np.random.default_rng(42), parlay=True,
            opening_policy_cache=np.repeat(open_cache[:, None, :], 101, axis=1),
        )
        expected = Calculator.calculate_scenario_returns(open_cache[None, :, :], frame.winner, opening, 'parlay')
        np.testing.assert_allclose(event['realized'], expected[1][0])
        self.assertAlmostEqual(event['opening_return'], expected[1][0])

    @staticmethod
    def frame(count=4):
        return pd.DataFrame(dict(
            date=['2026-01-01'] * count, fighter_red=[f'r{i}' for i in range(count)],
            fighter_blue=[f'b{i}' for i in range(count)], winner=[i % 2 for i in range(count)],
            pred_winner=[1] * count, proba_red=[.8] * count, proba_blue=[.2] * count,
            risk_red_proba=[.75] * count, prediction_se=[.04] * count,
            dec_close1_red=[2.] * count, dec_close1_blue=[2.] * count,
            dec_fair_close1_red=[2.] * count, dec_fair_close1_blue=[2.] * count,
            dec_open_red=[2.1] * count, dec_open_blue=[1.9] * count,
        ))

    def test_batch_matches_event_calculator_with_ties_risk_and_single_fight(self):
        for count in (1, 4, 12):
            data = self.frame(count)
            for z in (None, .5):
                with self.subTest(count=count, z=z):
                    params = BettingParams(z=z)
                    calc = Calculator(data[['dec_close1_blue', 'dec_close1_red']],
                                      data[['dec_fair_close1_blue', 'dec_fair_close1_red']], data.date)
                    cache = calc.calculate_scenario_cache(data, params)[None, :, :]
                    batch = calc.calculate_scenario_returns(cache, data.winner, params, 'combined')
                    inputs = dict(red_proba=data.proba_red.to_numpy(), risk_red_proba=data.risk_red_proba.to_numpy(),
                                  se=data.prediction_se.to_numpy(), winner_fold=data.winner.to_numpy(),
                                  val_idx=np.arange(count), param=params.to_policy())
                    events = calc.calculate_fight_returns(**inputs)
                    np.testing.assert_allclose(batch[1], events.event_return, atol=1e-12)
                    ticket = calc.calculate_parlay_allocations(**inputs)
                    parlay = calc.calculate_scenario_returns(cache, data.winner, params, 'parlay')
                    np.testing.assert_allclose(parlay[2], ticket.fstar, atol=1e-12)
                    self.assertLessEqual(batch[2][0], 1 + 1e-12)

    def test_decay_settlement_is_uncapped_and_rejects_different_predictions(self):
        data = self.frame(2)
        params = BettingParams(z=None)
        cache = Calculator.calculate_scenario_cache(data, params)[None, :, :]
        close = cache.copy()
        close[:, :, 0] *= .5
        rates = dict(k_down=0., k_up=0.)
        expected = Calculator.calculate_scenario_returns(cache, data.winner, params)
        actual = Calculator.calculate_decay_scenario_returns(
            cache, close, data[['dec_open_red', 'dec_open_blue']], data.winner,
            params, params, rates)
        np.testing.assert_allclose(actual, expected)
        close[:, :, 5] += .01
        with self.assertRaisesRegex(ValueError, 'must share'):
            Calculator.calculate_decay_scenario_returns(
                cache, close, data[['dec_open_red', 'dec_open_blue']], data.winner,
                params, params, rates)
        np.testing.assert_allclose(Calculator.settle_allocations([[.8, .8]], [[.2, .2]], [[-1, -1]])[1], [-1.6])

    def test_vectorized_sizing_matches_known_scalar_results(self):
        p = np.array([.6, .75, .9])
        odds = np.array([2., 1.8, 1.4])
        full = (p * odds - 1) / (odds - 1)
        batch = scale_kelly_for_mdd(p, odds, full, 500, .3)
        # Scalar reference bisection kept only here as an independent regression oracle.
        from ufc_betting.BettingStrategy.kelly_scaling import log_return_volatility, expected_max_drawdown
        expected = []
        for prob, price, fraction in zip(p, odds, full):
            low, high = 0., 1.
            while high - low > 1e-4:
                middle = (low + high) / 2
                sigma, _ = log_return_volatility(middle * fraction, price - 1, prob)
                if expected_max_drawdown(sigma, 500) <= -np.log1p(-.3):
                    low = middle
                else:
                    high = middle
            expected.append(low * fraction)
        np.testing.assert_allclose(batch, expected, atol=1e-14)

    def test_convergence_has_no_riskmanagement_imports(self):
        for path in Path(workflow.__file__).parent.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn('RiskManagement', node.module or '', str(path))

    def test_price_prediction_refreshes_calculator_risk_inputs(self):
        frame = self.frame(2).assign(x=[0., 1.], prediction_se=.99, risk_red_proba=.01)
        model = SimpleNamespace(
            model=SimpleNamespace(exog_names=['const', 'x']), params=np.array([1., 1.]),
            predict=lambda design: np.array([.7, .6]), cov_params=lambda: np.eye(2),
        )
        scaler = SimpleNamespace(feature_names_in_=['x'], transform=lambda values: values.to_numpy())
        predicted = model_input.predict_probabilities(frame, model, scaler_path=scaler)
        np.testing.assert_allclose(predicted.risk_red_proba, [.7, .6])
        np.testing.assert_allclose(predicted.prediction_se, [.21, .24 * np.sqrt(2)])
        np.testing.assert_allclose(predicted.prediction_se, predicted.proba_se)

    def test_model_path_and_feature_dataframe_need_no_odds_or_outcomes(self):
        frame = pd.DataFrame({'x': [0., 1.]}, index=[8, 3])
        original = frame.copy(deep=True)
        model = SimpleNamespace(
            model=SimpleNamespace(exog_names=['const', 'x']), params=np.array([1., 1.]),
            predict=lambda design: np.array([.7, .6]), cov_params=lambda: np.eye(2),
        )
        with patch.object(model_input.sm, 'load', return_value=model) as load, \
             patch.object(model_input.joblib, 'load', side_effect=AssertionError('No implicit scaler')):
            predicted = model_input.predict_probabilities(frame, Path('chosen_model.pkl'))
            load.assert_called_once_with(Path('chosen_model.pkl'))
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(predicted.index.tolist(), [8, 3])
        self.assertNotIn('correct_pred', predicted)
        np.testing.assert_allclose(predicted.proba_red, [.7, .6])
        with self.assertRaisesRegex(KeyError, 'Model feature'):
            model_input.predict_probabilities(pd.DataFrame({'wrong': [1.]}), model)

    def test_explicit_scaler_path_is_applied(self):
        frame = pd.DataFrame({'x': [2.]})
        model = SimpleNamespace(
            model=SimpleNamespace(exog_names=['const', 'x']), params=np.array([1., 1.]),
            predict=lambda design: np.full(len(design), .5), cov_params=lambda: np.eye(2),
        )
        scaler = SimpleNamespace(feature_names_in_=['x'], transform=lambda data: data.to_numpy() / 2)
        with patch.object(model_input.sm, 'load', return_value=model), \
             patch.object(model_input.joblib, 'load', return_value=scaler) as load:
            result = model_input.predict_probabilities(frame, 'model.pkl', scaler_path='scaler.pkl')
            load.assert_called_once_with('scaler.pkl')
        np.testing.assert_allclose(result.prediction_se, [.25 * np.sqrt(2)])

    def test_small_workflow_saves_paths_and_renders_using_shared_calculator(self):
        frame = self.frame(2).drop(columns=['risk_red_proba'])
        def predict(data, model, scaler_path=None):
            return data.copy()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(workflow.sm, 'load', return_value=object()), \
                 patch.object(workflow.joblib, 'load', return_value=object()), \
                 patch.object(workflow, 'predict_probabilities', side_effect=predict), \
                 patch.object(Calculator, 'calculate_scenario_returns', wraps=Calculator.calculate_scenario_returns) as settle:
                result = workflow.run(samples=101, n_paths=3, parlay=True, output_dir=directory,
                                      model_path='model.pkl', validation_data=frame, opening_data=frame)
                self.assertGreater(settle.call_count, 0)
            path = Path(directory)
            with np.load(path / 'bankroll_paths.npz') as saved:
                self.assertEqual(saved['bankroll'].shape, (6, 3, 2))
                self.assertTrue(np.isfinite(saved['bankroll']).all())
            self.assertTrue((path / 'realized_bankroll_bands.png').is_file())
            self.assertEqual(len(result['summary']), 12)


if __name__ == '__main__':
    unittest.main()
