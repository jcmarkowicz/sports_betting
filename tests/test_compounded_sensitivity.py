import unittest
import pandas as pd
from ufc_betting.Models.RiskManagement.compounded_sensitivity import compound


class CompoundTests(unittest.TestCase):
    def test_compounding_in_date_order(self):
        data = pd.DataFrame({"date": ["2020-02-01", "2020-01-01"] * 2,
                             "shift_pct": [0, 0, 100, 100],
                             "expected_bankroll_return_pct": [-10, 10, 20, 10]})
        paths, ending = compound(data, 100)
        values = ending.set_index("shift_pct")
        self.assertAlmostEqual(values.loc[0, "ending_bankroll"], 99)
        self.assertAlmostEqual(values.loc[100, "ending_bankroll"], 132)
        self.assertAlmostEqual(paths.iloc[1].bankroll_before_event, 110)

    def test_missing_event_rejected(self):
        data = pd.DataFrame({"date": ["2020-01-01", "2020-02-01", "2020-01-01"],
                             "shift_pct": [0, 0, 100], "expected_bankroll_return_pct": [1, 2, 3]})
        with self.assertRaises(ValueError):
            compound(data, 100)
