import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import optuna
import pandas as pd

from ufc_betting.Models.RiskManagement.delayed_bayesian_search import (
    FrozenBettingSearch, log_growth, terminal_return_score,
    bootstrap_event_indices, simulate_bankroll_paths,
)
from ufc_betting.Models.RiskManagement.delayed_walk_forward_logit import (
    window_masks, train_opening_bundle, predict_odds, run_delayed_walk_forward,
    chronological_folds, prepare_history, validate_alpha_history,
    predict_from_alpha_history, search_alpha_history,
)


def synthetic_data():
    rng = np.random.default_rng(41)
    dates = np.repeat(pd.date_range('2019-01-01', '2021-06-30', freq='14D'), 4)
    x = rng.normal(size=len(dates))
    p = 1 / (1 + np.exp(-.4 * x))
    frame = pd.DataFrame(dict(date=dates, winner=rng.binomial(1, p), x=x))
    frame['fighter_red'] = [f'red_{i}' for i in range(len(frame))]
    frame['fighter_blue'] = [f'blue_{i}' for i in range(len(frame))]
    for kind, offset in [('open', 0), ('close1', .15), ('close2', -.12)]:
        frame[f'dec_{kind}_red'] = rng.uniform(1.6, 2.6, len(frame)) + offset
        frame[f'dec_{kind}_blue'] = rng.uniform(1.6, 2.6, len(frame)) - offset
    inv = 1 / frame[['dec_open_red', 'dec_open_blue']].to_numpy()
    fair = inv / inv.sum(axis=1)[:, None]
    frame['proba_fair_open_diff'] = fair[:, 0] - fair[:, 1]
    frame['winner'] = rng.binomial(1, 1 / (1 + np.exp(
        -(.4 * x + 4 * frame.proba_fair_open_diff.to_numpy()))))
    return frame


def betting_predictions():
    frame = synthetic_data().iloc[:24].copy()
    frame['proba_red'] = .8
    frame['risk_red_proba'] = .78
    frame['prediction_se'] = .03
    for kind in ('open', 'close1', 'close2'):
        frame[f'dec_fair_{kind}_red'] = 2.
        frame[f'dec_fair_{kind}_blue'] = 2.
    return frame


PARAMS = dict(mdd=.3, N=250, mdd_parlay=.3, N_parlay=500, z=0.,
              moneyline_min_american=-600, moneyline_max_american=500)


class DelayedWalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cls.data = synthetic_data()
        cls.features = ['x', 'proba_fair_open_diff']
        cls.bundle, cls.study = train_opening_bundle(
            cls.data[cls.data.date < '2020-10-01'], '2020-10-01',
            features=cls.features, n_splits=3, n_trials=2,
        )

    def test_exclusive_cutoffs_and_configurable_months(self):
        dates = pd.Series(pd.to_datetime(['2020-09-30', '2020-10-01', '2020-12-31',
                                         '2021-01-01', '2021-03-31', '2021-04-01']))
        train, tune, test = window_masks(dates, '2021-01-01', '2021-04-01')
        self.assertEqual(np.flatnonzero(train).tolist(), [0])
        self.assertEqual(np.flatnonzero(tune).tolist(), [1, 2])
        self.assertEqual(np.flatnonzero(test).tolist(), [3, 4])
        _, tune, _ = window_masks(dates, '2021-01-01', '2021-04-01', n_months=1)
        self.assertEqual(np.flatnonzero(tune).tolist(), [2])

    def test_actual_time_series_split_and_no_event_overlap(self):
        from sklearn.model_selection import TimeSeriesSplit
        events = self.data.date.drop_duplicates().to_numpy()
        expected = list(TimeSeriesSplit(n_splits=3).split(events))
        actual = list(chronological_folds(self.data.date, n_splits=3))
        for (tr, va), (et, ev) in zip(actual, expected):
            np.testing.assert_array_equal(self.data.date.iloc[tr].unique(), events[et])
            np.testing.assert_array_equal(self.data.date.iloc[va].unique(), events[ev])
            self.assertLess(self.data.date.iloc[tr].max(), self.data.date.iloc[va].min())

    def test_alpha_only_and_mean_fold_brier(self):
        self.assertEqual(set(self.study.best_params), {'alpha'})
        for trial in self.study.trials:
            if trial.value is not None:
                self.assertAlmostEqual(trial.value, np.mean(trial.user_attrs['fold_brier']))
        for fold in self.study.user_attrs['folds']:
            self.assertLess(fold['train_end'], fold['validation_start'])

    def test_fold_preprocessors_fit_only_training_rows(self):
        from ufc_betting.Models.RiskManagement import delayed_walk_forward_logit as module
        seen = []
        original = module.make_preprocessor
        def capture(frame):
            seen.append(frame.index.to_numpy())
            return original(frame)
        train = self.data[self.data.date < '2020-10-01'].reset_index(drop=True)
        with patch.object(module, 'make_preprocessor', side_effect=capture):
            train_opening_bundle(train, '2020-10-01', features=self.features,
                                 n_splits=3, n_trials=1)
        for fitted, (tr, _) in zip(seen[:3], chronological_folds(train.date, 3)):
            np.testing.assert_array_equal(fitted, tr)

    def test_builder_keeps_missing_closing_odds_and_event_aligned_split(self):
        data = self.data.copy()
        data['dec_close1_red'] = np.nan
        filtered, split = prepare_history(data, features=self.features)
        self.assertEqual(len(filtered), len(data))
        self.assertLess(split['dates_train'].max(), split['dates_test'].min())
        target = int(.85 * len(data))
        self.assertLessEqual(len(split['train_df']), target)
        self.assertLess(target - len(split['train_df']), 4)

    def test_builder_date_filter(self):
        old = self.data.iloc[:2].copy()
        old['date'] = pd.to_datetime(['2010-02-25', '2010-02-26'])
        filtered, _ = prepare_history(pd.concat([old, self.data]), features=self.features)
        self.assertTrue((filtered.date > pd.Timestamp('2010-02-26')).all())

    def test_future_training_and_closing_features_rejected(self):
        with self.assertRaises(ValueError):
            train_opening_bundle(self.data, '2020-01-01', features=['x'])
        with self.assertRaisesRegex(ValueError, 'opening'):
            train_opening_bundle(self.data, '2022-01-01', features=['dec_close1_red'])

    def test_all_odds_types_feed_opening_slots_and_use_real_se(self):
        test = self.data[self.data.date >= '2020-10-01'].copy()
        for kind in ('open', 'close1', 'close2'):
            first = predict_odds(test, self.bundle, kind)
            changed = test.copy()
            changed['proba_fair_open_diff'] = 900
            if kind != 'open':
                changed['dec_open_red'] = 900
            second = predict_odds(changed, self.bundle, kind)
            np.testing.assert_allclose(first.proba_red, second.proba_red)
            self.assertTrue((first.prediction_se > 0).all())
            self.assertTrue(first.odds_type.eq(kind).all())
            self.assertTrue(first.risk_red_proba.between(0, 1).all())
        self.assertFalse(np.allclose(predict_odds(test, self.bundle, 'open').proba_red,
                                     predict_odds(test, self.bundle, 'close1').proba_red))
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            predict_odds(self.data.iloc[:4], self.bundle)

    def test_inference_se_matches_statsmodels(self):
        test = self.data[self.data.date >= '2020-10-01'].copy()
        prediction = predict_odds(test, self.bundle, 'open')
        inputs = prediction[self.features]
        design = self.bundle['preprocessor'].transform(inputs)
        design = np.column_stack([np.ones(len(design)), design])
        infer = self.bundle['inference_model']
        expected = infer.get_prediction(design[:, self.bundle['inference_columns']]).summary_frame()
        np.testing.assert_allclose(prediction.prediction_se, expected['se'])

    def test_alpha_history_rejects_future_selection_and_gaps(self):
        history = pd.DataFrame(dict(effective_date=['2020-10-01'],
                                    training_cutoff=['2020-10-01'], alpha=[.5]))
        with self.assertRaisesRegex(ValueError, 'cover'):
            predict_from_alpha_history(self.data.iloc[:4], self.data, history, features=self.features)
        history['training_cutoff'] = '2020-11-01'
        with self.assertRaises(ValueError):
            validate_alpha_history(history)

    def test_historical_predictions_ignore_future_outcomes_and_alphas(self):
        history = pd.DataFrame(dict(effective_date=['2020-10-01', '2021-01-01'],
                                    training_cutoff=['2020-10-01', '2021-01-01'], alpha=[.5, 1.]))
        target = self.data[(self.data.date >= '2020-10-01') & (self.data.date < '2021-01-01')]
        first = predict_from_alpha_history(target, self.data, history, features=self.features)
        changed = self.data.copy()
        changed.loc[changed.date >= '2020-10-01', 'winner'] = 1 - changed.loc[
            changed.date >= '2020-10-01', 'winner']
        history.loc[1, 'alpha'] = 7.
        second = predict_from_alpha_history(target, changed, history, features=self.features)
        np.testing.assert_allclose(first.proba_red, second.proba_red)
        np.testing.assert_allclose(first.prediction_se, second.prediction_se)

    def test_bootstrap_paths_settle_every_occurrence_and_both_families(self):
        events = pd.DataFrame(dict(moneyline_return=[.1, -.04, .03],
                                   parlay_return=[.1, -.06, .02],
                                   event_return=[.2, -.1, .05]))
        indices = np.array([[0, 0, 1, 2], [1, 1, 1, 2]])
        paths = simulate_bankroll_paths(events, indices, 500)
        np.testing.assert_allclose(paths['bankroll'][:, 0], 500)
        self.assertEqual(paths['bankroll'].shape, (2, 5))
        self.assertAlmostEqual(paths['bankroll'][0, -1], 500 * 1.2 * 1.2 * .9 * 1.05)
        self.assertNotEqual(*paths['terminal_returns'])
        np.testing.assert_allclose(
            paths['moneyline_cumulative_returns'] + paths['parlay_cumulative_returns'],
            paths['bankroll'][:, 1:] / 500 - 1)
        score = paths['terminal_returns'].mean() / paths['terminal_returns'].std(ddof=1)
        self.assertAlmostEqual(terminal_return_score(paths['terminal_returns']), score)

    def test_bootstrap_is_reproducible_and_reuses_scenarios(self):
        a = bootstrap_event_indices(10, 100, 42)
        np.testing.assert_array_equal(a, bootstrap_event_indices(10, 100, 42))
        self.assertTrue(any(len(set(row)) < 10 for row in a))
        self.assertFalse(np.array_equal(a, np.sort(a, axis=1)))
        search = FrozenBettingSearch(betting_predictions(), k=100)
        first = search.simulate(PARAMS)
        second = search.simulate(PARAMS)
        np.testing.assert_array_equal(first['bankroll'], second['bankroll'])
        self.assertGreater(np.std(first['terminal_returns']), 0)

    def test_compounding_ruin_and_zero_variance(self):
        self.assertAlmostEqual(np.expm1(log_growth([.2, -.1, 0])), .08)
        self.assertEqual(log_growth([-.1, -1, .5]), -np.inf)
        with self.assertRaises(ValueError):
            log_growth([-1.01])
        self.assertEqual(terminal_return_score([0, 0]), 0)
        self.assertEqual(terminal_return_score([.2, .2]), -np.inf)
        events = pd.DataFrame(dict(event_return=[-1., .5]))
        result = simulate_bankroll_paths(events, np.array([[0, 1]]))
        np.testing.assert_array_equal(result['bankroll'], [[500, 0, 0]])

    def test_joint_search_never_fits_model_and_passes_inference_se(self):
        search = FrozenBettingSearch(betting_predictions(), k=20)
        original = search.engine.calculate_fight_returns
        def capture(**kwargs):
            np.testing.assert_allclose(kwargs['se'], .03)
            np.testing.assert_allclose(kwargs['risk_red_proba'], .78)
            return original(**kwargs)
        with patch('statsmodels.api.Logit', side_effect=AssertionError('Must stay frozen')):
            with patch.object(search.engine, 'calculate_fight_returns', side_effect=capture):
                study = search.run(n_trials=2)
        self.assertEqual(set(study.best_params), set(PARAMS))
        self.assertAlmostEqual(study.best_value, terminal_return_score(
            search.simulate(study.best_params)['terminal_returns']))

    def test_parlays_never_cross_events_and_se_changes_exposure(self):
        frame = betting_predictions()
        frame['winner'] = 1
        frame['dec_open_red'] = 2.
        frame['dec_open_blue'] = 2.
        search = FrozenBettingSearch(frame, k=20)
        original = search.engine.parlay_returns
        def capture(dat, winner_id, param, **kwargs):
            result = original(dat, winner_id, param, **kwargs)
            for _, row in result.iterrows():
                if row['selected_ids']:
                    selected = dat[dat.id.isin(row['selected_ids'])]
                    self.assertEqual(selected.date.nunique(), 1)
                    self.assertEqual(selected.date.iloc[0], row.date)
            return result
        with patch.object(search.engine, 'parlay_returns', side_effect=capture):
            events = search.event_returns(PARAMS)
        self.assertTrue((events.moneyline_return > 0).any())
        self.assertTrue((events.parlay_return > 0).any())
        np.testing.assert_allclose(events.moneyline_return + events.parlay_return, events.event_return)
        reduced = search.event_returns({**PARAMS, 'z': 2.5})
        self.assertTrue((reduced.event_return < events.event_return).all())

    def test_no_bet_events_are_zero_and_remain_in_sampling(self):
        frame = betting_predictions()
        frame['proba_red'] = .5
        frame['risk_red_proba'] = .5
        frame[['dec_open_red', 'dec_open_blue']] = 1.8
        search = FrozenBettingSearch(frame, k=20)
        events = search.event_returns(PARAMS)
        self.assertEqual(len(events), frame.date.nunique())
        np.testing.assert_array_equal(events.event_return, 0)
        self.assertEqual(terminal_return_score(search.simulate(PARAMS)['terminal_returns']), 0)

    def test_end_to_end_refits_and_reuses_historical_alphas(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_delayed_walk_forward(
                self.data, '2021-01-01', '2021-07-01', directory,
                features=self.features, n_splits=3, model_trials=2, betting_trials=2, k=30)
            self.assertEqual(len(result['windows']), 2)
            self.assertFalse(result['events'].date.duplicated().any())
            self.assertAlmostEqual(result['events'].bankroll.iloc[-1],
                                   500 * np.prod(1 + result['events'].event_return))
            first = joblib.load(Path(directory) / 'models/2021-01-01/opening_model.joblib')
            self.assertEqual(first['training_cutoff'], '2021-01-01')
            tune = pd.read_csv(Path(directory) / '2021-01-01/tuning_predictions.csv')
            self.assertTrue(tune.model_training_cutoff.eq('2020-10-01').all())
            self.assertTrue((Path(directory) / '2021-04-01/bootstrap_paths.npz').exists())
            with patch('ufc_betting.Models.RiskManagement.delayed_walk_forward_logit.train_opening_bundle',
                       side_effect=AssertionError('Must not rerun alpha search')):
                again = run_delayed_walk_forward(
                    self.data, '2021-01-01', '2021-07-01', Path(directory) / 'reuse',
                    features=self.features, betting_trials=2, k=30,
                    alpha_history=result['alpha_history'], odds_type='close2')
            self.assertTrue(again['predictions'].odds_type.eq('close2').all())
            from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
            for run in (result, again):
                betting_data = run['betting_data']
                self.assertIsInstance(betting_data, pd.DataFrame)
                np.testing.assert_allclose(betting_data.se, betting_data.prediction_se)
                np.testing.assert_allclose(betting_data.risk_proba_blue, 1 - betting_data.risk_red_proba)
                sim, _ = simulate_kelly(betting_data, **run['kelly_sim_kwargs'])
                actual = sim.groupby('date', sort=True).bankroll_postevent.first()
                np.testing.assert_allclose(actual, run['events'].bankroll, rtol=1e-10)
            # CSV/JSON roundtrip is usable without DataFrame attributes.
            import json
            saved = pd.read_csv(Path(directory) / 'betting_data.csv', parse_dates=['date'])
            settings = json.loads((Path(directory) / 'kelly_sim_kwargs.json').read_text())
            sim, _ = simulate_kelly(saved, **settings)
            self.assertAlmostEqual(sim.bankroll_postevent.iloc[-1], result['events'].bankroll.iloc[-1])

    def test_default_split_and_partial_final_window(self):
        with tempfile.TemporaryDirectory() as directory:
            _, split = prepare_history(self.data, features=self.features)
            start = split['dates_test'].min()
            result = run_delayed_walk_forward(
                self.data, output_dir=directory, features=self.features, n_splits=3,
                model_trials=1, betting_trials=1, k=20)
            self.assertEqual(result['windows'].iloc[0].test_start, str(start.date()))
            self.assertTrue(result['windows'].iloc[-1].partial_test_window)

    def test_month_end_warmup_does_not_add_an_extra_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_delayed_walk_forward(
                self.data, '2021-05-31', '2021-06-30', directory,
                features=self.features, n_splits=3, model_trials=1, betting_trials=1, k=20)
            self.assertEqual(result['alpha_history'].effective_date.dt.strftime('%Y-%m-%d').tolist(),
                             ['2021-02-28', '2021-05-31'])

    def test_only_regular_within_event_parlay_is_settled(self):
        frame = betting_predictions().iloc[:2].copy()
        frame['proba_red'], frame['risk_red_proba'] = .95, .95
        frame['winner'] = 1
        frame['dec_open_red'], frame['dec_open_blue'] = 1.2, 6.
        frame['dec_fair_open_red'], frame['dec_fair_open_blue'] = 1.25, 5.
        search = FrozenBettingSearch(frame, k=20)
        original = search.engine.parlay_returns
        with patch.object(search.engine, 'parlay_returns', wraps=original) as tickets:
            events = search.event_returns({**PARAMS, 'moneyline_min_american': -100,
                                           'moneyline_max_american': 100})
        self.assertEqual(tickets.call_count, 1)
        self.assertEqual(tickets.call_args.kwargs['strategy'], 'top_ev')
        self.assertEqual(events.moneyline_return.iloc[0], 0)
        self.assertGreater(events.parlay_return.iloc[0], 0)
        self.assertEqual(events.parlay_return.iloc[0], events.event_return.iloc[0])


if __name__ == '__main__':
    unittest.main()
