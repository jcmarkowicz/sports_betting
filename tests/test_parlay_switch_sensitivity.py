import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from ufc_betting.Models.RiskManagement.parlay_switch_sensitivity import build_paths, switch_intervals


class SensitivityTests(unittest.TestCase):
    def test_intervals_preserve_reversals_and_strict_threshold(self):
        self.assertEqual(switch_intervals([0, 25, 50, 75, 100], [.03, .02, 0, .03, .04], .02),
                         [(0., 0.), (75., 100.)])
        self.assertEqual(switch_intervals([0, 100], [-1, 0]), [])

    def test_fixed_fighters_opposite_endpoints_and_shared_midpoint(self):
        frame = pd.DataFrame(dict(date=['2025-01-01'] * 3,
            fighter_red=['A', 'B', 'C'], fighter_blue=['X', 'Y', 'Z'], winner=[1, 0, 1],
            dec_open_red=[2.5, 2., 1.2], dec_open_blue=[2., 2., 2.],
            dec_close1_red=[1.2, 2., 2.5], dec_close1_blue=[2., 2., 2.]))
        # Freeze probabilities for this fixture to isolate odds interpolation.
        with patch('ufc_betting.Models.RiskManagement.parlay_switch_sensitivity.predict',
                   return_value=np.array([.8, .8, .8])):
            paths, legs = build_paths(frame, None, None, steps=2)
        self.assertEqual(paths.opening_fighters.unique().tolist(), ['A | B'])
        self.assertEqual(paths.close1_fighters.unique().tolist(), ['B | C'])
        self.assertAlmostEqual(paths.opening_decimal_odds.iloc[0], 5.)
        self.assertAlmostEqual(paths.opening_decimal_odds.iloc[-1], 2.4)
        self.assertAlmostEqual(paths.close1_decimal_odds.iloc[0], 5.)
        self.assertAlmostEqual(paths.close1_decimal_odds.iloc[-1], 2.4)
        midpoint = legs.loc[legs.shift_pct.eq(50) & legs.fighter.eq('B')]
        self.assertEqual(midpoint.decimal_odds.nunique(), 1)
        self.assertAlmostEqual(paths.opening_decimal_odds.iloc[1], 3.7)


if __name__ == '__main__':
    unittest.main()
