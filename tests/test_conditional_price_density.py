import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
from ufc_betting.Models.LineMovement import conditional_price_density as density


class ConditionalDensityTests(unittest.TestCase):
    def test_fit_supplied_dataframe_without_loading_or_split_selection(self):
        observations = self.observations().iloc[::2].copy()
        original = observations.copy(deep=True)
        with patch.object(density, 'exact_day_observations', side_effect=AssertionError('No loading')):
            fitted = density.fit_price_density(observations, days_out=6, odds_type='close2')
        self.assertNotIn('split', fitted)
        self.assertEqual(len(fitted['model'].endog), len(observations))
        pd.testing.assert_frame_equal(fitted['observations'], original)
        pd.testing.assert_frame_equal(observations, original)

    @staticmethod
    def observations():
        t = np.arange(36)
        return pd.DataFrame(dict(date=np.repeat(['2025-01-01', '2025-02-01', '2025-03-01'], 12),
                                 opening=1.6 + t / 40, closing=1.5 + t / 50 + .1 * np.sin(t),
                                 price=1.7 + t / 60 + .2 * np.cos(t)))

    def fit(self):
        with patch.object(density, 'exact_day_observations', return_value=self.observations()):
            return density.fit_day_price_density('.', 6)

    def test_fits_library_estimator_from_historical_records_and_persists(self):
        fitted = self.fit()
        self.assertIsInstance(fitted['model'], KDEMultivariateConditional)
        self.assertNotIn('training', fitted)
        self.assertFalse(hasattr(density, 'PriceDensity'))
        self.assertFalse(hasattr(density, 'condition_fitted_density'))
        self.assertFalse(hasattr(fitted['model'], 'sample'))
        pd.testing.assert_frame_equal(fitted['observations'], self.observations())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(fitted, path)
            loaded = joblib.load(path)
        np.testing.assert_allclose(density.price_pdf(fitted['model'], [1.5, 2., 3.], 2., 1.8),
                                   density.price_pdf(loaded['model'], [1.5, 2., 3.], 2., 1.8))

    def test_decimal_pdf_has_correct_jacobian_and_integrates_to_one(self):
        model = self.fit()['model']
        prices = 1 + np.exp(np.linspace(-4, 3, 2000))
        pdf = density.price_pdf(model, prices, 2., 1.8)
        self.assertAlmostEqual(np.trapezoid(pdf, prices), 1., places=4)
        interval = (prices >= 1.8) & (prices <= 2.4)
        self.assertAlmostEqual(np.trapezoid(pdf[interval], prices[interval]),
                               density.price_probability(model, 1.8, 2.4, 2., 1.8), delta=.01)
        np.testing.assert_array_equal(density.price_pdf(model, [.5, 1.], 2., 1.8), [0, 0])
        self.assertEqual(density.price_probability(model, 1, np.inf, 2., 1.8), 1.)

    def test_plot_is_deterministic_without_sampling(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(density, 'exact_day_observations', return_value=self.observations()):
            result = density.plot_day_price_density('.', days_out=6, output_dir=directory)
            self.assertTrue(result['plot_path'].is_file())
            self.assertEqual(set(result['models']), {'close1', 'close2'})
            self.assertEqual(len(result['summary']), 2)

    def test_plot_fitted_model_does_not_retrain_for_new_odds(self):
        model = self.fit()['model']
        prices = np.linspace(1.1, 5., 100)
        with patch.object(density, 'fit_day_price_density', side_effect=AssertionError('No refitting')), \
             patch.object(density.plt, 'show'):
            for opening, closing in ((2., 1.8), (2.4, 2.1)):
                fig, ax = density.plot_price_density(model, opening, closing, prices=prices)
                np.testing.assert_allclose(ax.lines[0].get_ydata(),
                                           density.price_pdf(model, prices, opening, closing))
                density.plt.close(fig)

    def test_samples_follow_conditional_cdf_and_are_reproducible(self):
        model = self.fit()['model']
        prices = density.sample_prices(model, 2., 1.8, 10000, random_state=42)
        self.assertEqual(prices.shape, (10000,))
        self.assertTrue(np.isfinite(prices).all() and (prices > 1).all())
        np.testing.assert_array_equal(prices, density.sample_prices(
            model, 2., 1.8, 10000, random_state=np.random.default_rng(42)))
        # Inverting the fitted CDF should recover the original uniform draws.
        recovered = model.cdf(endog_predict=np.log(prices[:100] - 1),
                              exog_predict=np.tile(density._query(2., 1.8), (100, 1)))
        np.testing.assert_allclose(recovered, np.random.default_rng(42).random(100), atol=2e-6)
        expected = density.price_probability(model, 1., 2., 2., 1.8)
        self.assertAlmostEqual(np.mean(prices <= 2.), expected, delta=.02)
        self.assertTrue((prices > 2.).any())  # No clipping to endpoint prices.

    def test_small_draws_solve_the_fitted_cdf_without_building_a_grid(self):
        model = self.fit()['model']
        for opening, closing in ((2., 1.8), (1.6, 2.4), (3., 2.)):
            for count in (1, 16):
                with patch.object(model, 'cdf', side_effect=AssertionError('No grid evaluation')):
                    prices = density.sample_prices(model, opening, closing, count, random_state=42)
                recovered = model.cdf(endog_predict=np.log(prices - 1),
                    exog_predict=np.tile(density._query(opening, closing), (count, 1)))
                np.testing.assert_allclose(recovered, np.random.default_rng(42).random(count), atol=1e-10)

    def test_sampling_validates_inputs_and_supports_empty_samples(self):
        model = self.fit()['model']
        self.assertEqual(density.sample_prices(model, 2., 1.8, 0).shape, (0,))
        for kwargs in ({'n_samples': -1}, {'n_samples': 1.5}, {'n_samples': True}, {'grid_size': 1}):
            with self.assertRaises(ValueError):
                density.sample_prices(model, 2., 1.8, **kwargs)
        with self.assertRaises(ValueError):
            density.sample_prices(model, 1., 1.8)

    def test_exact_day_selects_only_actual_latest_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model_results').mkdir()
            archive = root / 'intermediate_odds_stats'; archive.mkdir()
            pd.DataFrame([dict(date='2025-01-10', fighter_red='Red', fighter_blue='Blue',
                               dec_open_red=2., dec_open_blue=2., dec_close1_red=1.8, dec_close1_blue=2.2)]).to_csv(
                root / 'model_results/test_set_data_open_with_close1.csv', index=False)
            rows = []
            for i, timestamp in enumerate(['2025-01-04T01:00:00Z', '2025-01-04T20:00:00Z', '2025-01-05T01:00:00Z']):
                filename = f'{i}.csv'
                rows.append(dict(event_date='2025-01-10', commit_datetime_utc=timestamp,
                                 commit=str(i), original_path=filename, filename=filename))
                pd.DataFrame([dict(fighter_red='Red', fighter_blue='Blue',
                                   dec_close1_red=2. + i / 10, dec_close1_blue=3. + i / 10)]).to_csv(archive / filename, index=False)
            pd.DataFrame(rows).to_csv(archive / 'manifest.csv', index=False)
            observed = density.exact_day_observations(root, 6, 'close1')
            np.testing.assert_allclose(observed.price, [2.1, 3.1])
            self.assertEqual(observed.source_file.unique().tolist(), ['1.csv'])


if __name__ == '__main__':
    unittest.main()
