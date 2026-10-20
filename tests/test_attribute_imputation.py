import unittest

import numpy as np
import pandas as pd

from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.final_feats_df import (
    fill_attributes_from_prior_dates, compute_differences,
)


class AttributeImputationTests(unittest.TestCase):
    def frame(self):
        df = pd.DataFrame({
            'date': ['2020-01-03', '2020-01-01', '2020-01-01',
                     '2020-01-02', '2020-01-03', '2020-01-04'],
            'weight_class': ['LW', 'LW', 'LW', 'LW', 'HW', 'LW'],
        }, index=[9, 8, 7, 6, 5, 4])
        for attr in ['reach', 'age', 'height']:
            df[f'{attr}_red'] = [np.nan, 60., 80., 90., np.nan, 200.]
            df[f'{attr}_blue'] = [np.nan, np.nan, 70., 80., 100., 210.]
        return df

    def test_prior_observations_only_and_recomputed_differences(self):
        original = self.frame()
        result = fill_attributes_from_prior_dates(original)
        pd.testing.assert_index_equal(result.index, original.index)
        for attr in ['reach', 'age', 'height']:
            self.assertAlmostEqual(result.loc[9, f'{attr}_red'], 230 / 3)
            self.assertEqual(result.loc[9, f'{attr}_blue'], 75.)
            self.assertTrue(pd.isna(result.loc[8, f'{attr}_blue']))  # Same-day value excluded.
            self.assertTrue(pd.isna(result.loc[5, f'{attr}_red']))  # No HW history.
            result = compute_differences(result, attr, None)
            self.assertAlmostEqual(result.loc[9, f'{attr}_diff'], 230 / 3 - 75)
        self.assertTrue(pd.isna(original.loc[9, 'reach_red']))

    def test_future_changes_do_not_affect_earlier_imputation(self):
        original = self.frame()
        changed = original.copy()
        changed.loc[4, 'reach_red'] = 9999.
        self.assertEqual(fill_attributes_from_prior_dates(original).loc[9, 'reach_red'],
                         fill_attributes_from_prior_dates(changed).loc[9, 'reach_red'])


if __name__ == '__main__':
    unittest.main()
