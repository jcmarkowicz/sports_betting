import unittest
import pandas as pd
from ufc_betting.Models.RiskManagement.target_ev_bankroll import build_paths


class TargetBankrollTests(unittest.TestCase):
    def test_exact_target_and_compounding(self):
        data = pd.DataFrame({"date":["2020-02-01","2020-01-01"],"target_pct_opening_ev":[50,50],
            "opening_expected_return_pct":[20,10],"achieved_pct_opening_ev":[100,80]})
        result = build_paths(data,100)
        self.assertAlmostEqual(result.bankroll_after_event.iloc[-1],115.5)
        self.assertAlmostEqual(result.stake_multiplier_to_hold_target.iloc[0],.625)

    def test_unreachable_rejected(self):
        data = pd.DataFrame({"date":["2020-01-01"],"target_pct_opening_ev":[90],
            "opening_expected_return_pct":[10],"achieved_pct_opening_ev":[80]})
        with self.assertRaises(ValueError):
            build_paths(data)
