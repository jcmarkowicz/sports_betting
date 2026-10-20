import unittest
import numpy as np
import test_joint_kelly_decay_search as fixtures
from ufc_betting.Models.Search.frozen_betting import path_mean_log_growth, path_max_drawdowns, path_ulcer_indices, validate_score_settings, worst_tail_drawdown
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.BettingStrategy.betting_params import BettingParams


class EventLogGrowthTests(unittest.TestCase):
    def test_ulcer_depth_duration_and_actual_balances(self):
        paths = np.array([[100., 80., 100., 100., 100.],
                          [100., 80., 80., 80., 80.],
                          [100., 110., 120., 130., 140.],
                          [100., 0., 0., 0., 0.],
                          [100., -10., -10., -10., -10.],
                          [100., 200., 100., 200., 200.]])
        original = paths.copy()
        np.testing.assert_allclose(path_ulcer_indices(paths), [.1, .2, 0., 1., 1.1, .25])
        np.testing.assert_array_equal(paths, original)
        for invalid in ([], [[0., 1.]], [[100.]], [[100., np.nan]]):
            with self.assertRaises(ValueError):
                path_ulcer_indices(invalid)
        for value in (-1., np.inf, np.nan, True):
            with self.assertRaises(ValueError):
                validate_score_settings('event_log_growth', ulcer_penalty=value)
        with self.assertRaises(ValueError):
            validate_score_settings('mean_std', ulcer_penalty=.1)

    def test_worst_tail_selection_and_rounding(self):
        values = np.array([.1, .9, .2, .8, .3, .7, .4, .6, .5, 0.])
        original = values.copy()
        self.assertAlmostEqual(worst_tail_drawdown(values), .9)
        self.assertAlmostEqual(worst_tail_drawdown(values, .2), .85)
        self.assertAlmostEqual(worst_tail_drawdown(values, 1.), values.mean())
        self.assertAlmostEqual(worst_tail_drawdown([.1, .9, .5], .5), .7)
        np.testing.assert_array_equal(values, original)
        for fraction in (0, -1, 1.1, np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                worst_tail_drawdown(values, fraction)

    def test_peak_drawdown_includes_initial_balance_and_ruin(self):
        paths = np.array([[100., 150., 75., 120.], [100., 80., 90., 100.],
                          [100., 120., 130., 140.], [100., 0., 0., 0.],
                          [100., -10., -10., -10.]])
        original = paths.copy()
        np.testing.assert_allclose(path_max_drawdowns(paths), [.5, .2, 0., 1., 1.1])
        np.testing.assert_array_equal(paths, original)
        for value in (-1., np.inf, np.nan, True):
            with self.assertRaises(ValueError):
                validate_score_settings('event_log_growth', drawdown_penalty=value)
        with self.assertRaises(ValueError):
            validate_score_settings('log_mean_roi', drawdown_penalty=.1)

    def test_drawdown_penalizes_score_not_reported_growth(self):
        frame = fixtures.JointKellyDecayTests().frame()
        for market in ('open', 'close1'):
            extra = {} if market == 'open' else dict(use_decay=True, opening_params=BettingParams())
            result = optimize_kelly_params(validation_data={market: frame}, k=7, n_trials=2,
                score_type='event_log_growth', drawdown_penalty=.1, drawdown_tail_fraction=.3, ulcer_penalty=.02, **extra)
            growth = result['path_mean_event_log_growth'].mean()
            drawdown = np.sort(result['path_max_drawdowns'])[-3:].mean()
            ulcer = result['path_ulcer_indices'].mean()
            self.assertAlmostEqual(result['best_value'], growth - .1*drawdown - .02*ulcer)
            self.assertAlmostEqual(result['objective_breakdown'].mean_ulcer_index.iloc[0], ulcer)
            self.assertAlmostEqual(result['objective_breakdown'].ulcer_penalty_amount.iloc[0], .02*ulcer)
            self.assertEqual(result['studies']['full_validation'].user_attrs['ulcer_penalty'], .02)
            self.assertAlmostEqual(result['geometric_growth_per_event'], np.expm1(growth))
            self.assertAlmostEqual(result['objective_breakdown'].mean_event_log_growth.iloc[0], growth)
            self.assertAlmostEqual(result['objective_breakdown'].drawdown_penalty_amount.iloc[0], .1*drawdown)
            self.assertEqual(result['studies']['full_validation'].user_attrs['drawdown_penalty'], .1)
            self.assertEqual(result['drawdown_tail_fraction'], .3)
            self.assertAlmostEqual(result['objective_breakdown'].worst_tail_max_drawdown.iloc[0], drawdown)

    def test_compounding_and_equal_event_weights(self):
        actual = path_mean_log_growth([[100., 110., 99.], [100., 100., 120.]])
        np.testing.assert_allclose(actual, [np.log(.99)/2, np.log(1.2)/2])
        self.assertLess(np.expm1(actual[0]), 0.)
        np.testing.assert_allclose(path_mean_log_growth([[500., 550., 495.]]), actual[:1])

    def test_floor_ruin_and_no_bets(self):
        paths = np.array([[100., 0., 0.], [100., -10., -10.], [100., 100., 100.]])
        original = paths.copy()
        np.testing.assert_allclose(path_mean_log_growth(paths, .01), [np.log(.01)/2]*2+[0.])
        np.testing.assert_array_equal(paths, original)
        # A temporary dip below the scoring floor followed by recovery telescopes.
        self.assertAlmostEqual(path_mean_log_growth([[100., .001, 10.]], .01)[0], np.log(.1)/2)
        for invalid in ([], [[0., 1.]], [[100.]], [[100., np.nan]]):
            with self.assertRaises(ValueError):
                path_mean_log_growth(invalid)

    def test_open_and_closing_integration(self):
        frame = fixtures.JointKellyDecayTests().frame()
        for market in ('open', 'close1'):
            extra = {} if market == 'open' else dict(use_decay=True, opening_params=BettingParams())
            result = optimize_kelly_params(validation_data={market: frame}, k=7, n_trials=2,
                score_type='event_log_growth', wealth_floor=.01, risk_penalty=100., **extra)
            study = result['studies']['full_validation']
            self.assertAlmostEqual(result['best_value'], result['path_mean_event_log_growth'].mean())
            self.assertAlmostEqual(result['geometric_growth_per_event'], np.expm1(study.best_value))
            legacy = optimize_kelly_params(validation_data={market: frame}, k=7, n_trials=2,
                score_type='log_growth', wealth_floor=.01, **extra)
            self.assertAlmostEqual(result['best_value'], legacy['best_value']/frame.date.nunique())
            self.assertEqual(result['trial_params'], legacy['trial_params'])
