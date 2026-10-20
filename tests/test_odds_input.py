import unittest

import numpy as np
import pandas as pd

from ufc_betting.Models.LogisticRegression.odds_input import replace_opening_odds
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.odds_features import devig_two_way


class OddsInputTests(unittest.TestCase):
    def test_both_markets_replace_features_without_mutating_original(self):
        frame = pd.DataFrame({'dec_open_red': [2.], 'dec_open_blue': [2.],
                              'dec_close1_red': [1.5], 'dec_close1_blue': [3.],
                              'dec_close2_red': [1.8], 'dec_close2_blue': [2.2],
                              'proba_fair_open_diff': [0.], 'other_feature': [9.]}, index=[7])
        original = frame.copy()
        for market in ('close1', 'close2'):
            inputs = replace_opening_odds(frame, market,
                                          features=['proba_fair_open_diff', 'dec_open_red', 'other_feature'])
            red, blue = devig_two_way(frame[f'dec_{market}_red'].iloc[0],
                                      frame[f'dec_{market}_blue'].iloc[0], method='power')[:2]
            self.assertAlmostEqual(inputs['proba_fair_open_diff'].iloc[0], red - blue)
            self.assertAlmostEqual(inputs[f'dec_fair_{market}_red'].iloc[0], 1 / red)
            self.assertEqual(inputs['dec_open_red'].iloc[0], frame[f'dec_{market}_red'].iloc[0])
            self.assertEqual(inputs['other_feature'].iloc[0], 9.)
            self.assertEqual(inputs.index.tolist(), [7])
        pd.testing.assert_frame_equal(frame, original)

    def test_explicit_sampled_prices_override_historical_prices(self):
        frame = pd.DataFrame({'dec_close1_red': [2.], 'dec_close1_blue': [2.]})
        inputs = replace_opening_odds(frame, prices=[[1.5, 3.]])
        self.assertAlmostEqual(inputs['proba_fair_open_diff'].iloc[0], 1 / 3)
        np.testing.assert_allclose(inputs[['dec_close1_red', 'dec_close1_blue']], [[1.5, 3.]])

    def test_feature_only_notebook_compatibility(self):
        opening = pd.DataFrame({'proba_fair_open_diff': [0.]}, index=[4])
        closing = pd.DataFrame({'proba_fair_close2_diff': [.2]}, index=[4])
        result = TrainTestBuilder.close_odds_to_open(opening, closing, 'close2')
        self.assertIs(result, opening)
        self.assertEqual(result['proba_fair_open_diff'].iloc[0], .2)

    def test_invalid_and_misaligned_prices_fail(self):
        for prices in ([[1., 2.]], [[np.nan, 2.]], [[2., 2.], [2., 2.]]):
            with self.subTest(prices=prices), self.assertRaises(ValueError):
                replace_opening_odds(pd.DataFrame(index=[0]), prices=prices)


if __name__ == '__main__':
    unittest.main()
