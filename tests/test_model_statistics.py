import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.model_statistics import (
    align_model_columns, model_covariance_and_variance, probability_standard_error,
)


class ModelStatisticsTests(unittest.TestCase):
    def model(self, covariance, params=(1., 1.)):
        return SimpleNamespace(model=SimpleNamespace(exog_names=['const', 'x']),
                               params=np.asarray(params), cov_params=lambda: covariance)

    def test_custom_aliases_raw_fallback_and_input_preservation(self):
        scaled = pd.DataFrame({'old': [.2, .4]}, index=[9, 3])
        raw = pd.DataFrame({'old': [2, 4], 'category': [0, 1]}, index=[9, 3])
        actual = align_model_columns(scaled, ['const', 'new', 'category'],
                                     fallback=raw, aliases={'new': 'old'})
        np.testing.assert_allclose(actual, [[1, .2, 0], [1, .4, 1]])
        self.assertEqual(actual.index.tolist(), [9, 3])
        self.assertEqual(scaled.columns.tolist(), ['old'])
        with self.assertRaisesRegex(ValueError, 'matching rows'):
            align_model_columns(scaled, ['category'], fallback=raw.iloc[::-1])

    def test_named_covariance_and_design_are_reordered_together(self):
        covariance = pd.DataFrame([[4., .5], [.5, 1.]], index=['x', 'const'], columns=['x', 'const'])
        model = self.model(covariance)
        cov, variance = model_covariance_and_variance(model, pd.DataFrame({'x': [2., 3.], 'const': [1., 1.]}))
        np.testing.assert_allclose(cov, [[1, .5], [.5, 4]])
        np.testing.assert_allclose(variance, [19, 40])
        np.testing.assert_allclose(probability_standard_error([.5, .8], variance),
                                   [.25 * np.sqrt(19), .16 * np.sqrt(40)])

    def test_trimmed_nan_covariance_is_omitted_but_valid_zero_coefficient_is_retained(self):
        design = pd.DataFrame({'const': [1.], 'x': [2.]})
        model = self.model([[1., np.nan], [np.nan, np.nan]], params=(1., 0.))
        cov, variance = model_covariance_and_variance(model, design)
        np.testing.assert_allclose(cov, [[1., 0.], [0., 0.]])
        np.testing.assert_allclose(variance, [1.])
        _, variance = model_covariance_and_variance(self.model(np.eye(2), params=(1., 0.)), design)
        np.testing.assert_allclose(variance, [5.])

    def test_bad_active_covariance_and_negative_variance_are_rejected(self):
        design = np.array([[1., 2.]])
        for cov in ([[1., np.nan], [np.nan, np.nan]], [[1., 0.], [0., -1.]], [[1., 1.], [0., 1.]]):
            with self.subTest(cov=cov), self.assertRaises(ValueError):
                model_covariance_and_variance(self.model(cov), design)

    def test_standard_errors_validate_scale_and_shape(self):
        for probabilities, variance in (([.2], [-1]), ([1.1], [1]), ([.2, .3], [1]), ([np.nan], [1])):
            with self.assertRaises(ValueError):
                probability_standard_error(probabilities, variance)
        np.testing.assert_array_equal(probability_standard_error([0., .5, 1.], [1., 0., 1.]), [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
