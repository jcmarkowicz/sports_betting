import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import optuna
import pandas as pd

from ufc_betting.Models.RiskManagement.delayed_bayesian_search import FrozenBettingSearch, log_growth
from ufc_betting.Models.RiskManagement.delayed_walk_forward_logit import (
    window_masks, train_opening_bundle, predict_close1, run_delayed_walk_forward,
)


def synthetic_data():
    rng = np.random.default_rng(41)
    dates = np.repeat(pd.date_range('2019-01-01', '2021-06-30', freq='14D'), 4)
    x = rng.normal(size=len(dates))
    p = 1 / (1 + np.exp(-.4 * x))
    return pd.DataFrame(dict(date=dates, winner=rng.binomial(1, p), x=x,
                             proba_fair_open_diff=rng.uniform(-.3, .3, len(dates)),
                             dec_close1_red=1.8, dec_close1_blue=2.2))


class DelayedWalkForwardTests(unittest.TestCase):
    def setUp(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def test_exclusive_cutoffs_and_three_month_delay(self):
        dates = pd.Series(pd.to_datetime(['2020-09-30', '2020-10-01', '2020-12-31',
                                         '2021-01-01', '2021-03-31', '2021-04-01']))
        train, tune, test = window_masks(dates, '2021-01-01', '2021-04-01')
        self.assertEqual(np.flatnonzero(train).tolist(), [0])
        self.assertEqual(np.flatnonzero(tune).tolist(), [1, 2])
        self.assertEqual(np.flatnonzero(test).tolist(), [3, 4])

    def test_compounding_and_ruin(self):
        self.assertAlmostEqual(np.expm1(log_growth([.2, -.1, 0])), .08)
        self.assertEqual(log_growth([-.1, -1, .5]), -np.inf)
        with self.assertRaises(ValueError):
            log_growth([-1.01])

    def test_future_training_rows_rejected(self):
        with self.assertRaises(ValueError):
            train_opening_bundle(synthetic_data(), '2020-01-01', features=['x'])

    def test_frozen_inference_uses_close1_not_opening_prices(self):
        data = synthetic_data()
        train = data[data.date < '2020-10-01']
        bundle, study = train_opening_bundle(
            train, '2020-10-01', features=['x', 'proba_fair_open_diff'],
            n_splits=3, n_trials=2,
        )
        self.assertEqual(study.direction, optuna.study.StudyDirection.MINIMIZE)
        test = data[data.date >= '2020-10-01'].copy()
        first = predict_close1(test, bundle)
        test['proba_fair_open_diff'] = 900
        second = predict_close1(test, bundle)
        np.testing.assert_allclose(first.proba_red, second.proba_red)
        np.testing.assert_allclose(1 / first.dec_fair_close1_red +
                                   1 / first.dec_fair_close1_blue, 1)

    def test_joint_search_never_fits_model_and_counts_no_bets(self):
        frame = synthetic_data().iloc[:8].copy()
        frame['proba_red'] = .8
        frame['dec_fair_close1_red'] = 2.0
        frame['dec_fair_close1_blue'] = 2.0
        search = FrozenBettingSearch(frame)
        with patch.object(search.engine, 'calculate_fight_returns', return_value=[.2, np.nan]):
            events = search.event_returns(dict(mdd_parlay=.3))
            self.assertEqual(events.event_return.tolist(), [.2, 0])
        with patch('statsmodels.api.Logit', side_effect=AssertionError('Must stay frozen')):
            study = search.run(n_trials=2)
        self.assertEqual(set(study.best_params), {'mdd', 'N', 'mdd_parlay', 'N_parlay',
                                                'moneyline_min_american', 'moneyline_max_american'})

    def test_end_to_end_two_windows_and_saved_model(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_delayed_walk_forward(
                synthetic_data(), '2021-01-01', '2021-07-01', directory,
                features=['x', 'proba_fair_open_diff'], n_splits=3,
                model_trials=2, betting_trials=2,
            )
            self.assertEqual(len(result['windows']), 2)
            self.assertFalse(result['events'].date.duplicated().any())
            self.assertAlmostEqual(result['events'].bankroll.iloc[-1],
                                   500 * np.prod(1 + result['events'].event_return))
            first = joblib.load(Path(directory) / '2021-01-01/opening_model.joblib')
            self.assertEqual(first['training_cutoff'], '2020-10-01')
            self.assertLess(first['training_last_date'], first['training_cutoff'])
            self.assertTrue((Path(directory) / '2021-04-01/betting_trials.csv').exists())

    def test_heavy_favorites_excluded_from_returns_and_exposure(self):
        # Both legs would qualify as heavy favorites. Exclude straight bets so
        # the event must settle exactly one top-EV ticket, with no second stake.
        frame = pd.DataFrame(dict(
            date=['2025-01-01'] * 2, winner=[1, 1], proba_red=[.95, .95],
            dec_close1_red=[1.2, 1.2], dec_close1_blue=[6., 6.],
            dec_fair_close1_red=[1.25, 1.25], dec_fair_close1_blue=[5., 5.],
        ))
        params = dict(mdd=.3, N=250, mdd_parlay=.3, N_parlay=500,
                      moneyline_min_american=-100, moneyline_max_american=100)
        search = FrozenBettingSearch(frame)
        tickets = []
        original = search.engine.parlay_returns
        def capture(*args, **kwargs):
            self.assertEqual(kwargs['strategy'], 'top_ev')
            result = original(*args, **kwargs)
            tickets.append(result)
            return result
        with patch.object(search.engine, 'parlay_returns', side_effect=capture):
            events = search.event_returns(params)
        self.assertEqual(len(tickets), 1)
        self.assertGreater(tickets[0].pct_return.iloc[0], 0)
        self.assertAlmostEqual(events.event_return.iloc[0], tickets[0].pct_return.iloc[0])


if __name__ == '__main__':
    unittest.main()
