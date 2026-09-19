import unittest
import numpy as np
import pandas as pd
from ufc_betting.Models.RiskManagement.model_switch_evaluation import choose_open, losses, settle


class SwitchTests(unittest.TestCase):
    def test_observable_distance_boundary(self):
        np.testing.assert_array_equal(choose_open([0, 12, 13], "open_near:12"), [True, True, False])
        np.testing.assert_array_equal(choose_open([0, 12, 13], "close1_near:12"), [False, False, True])

    def test_settlement_cap_applied_after_selection(self):
        rows = pd.DataFrame({"date": ["2020-01-01"] * 2, "scenario_pct": [0, 0],
            "fight_id": [0, 1], "distance_pp": [1, 20], "winner": [1, 0],
            "p_open": [.8, .7], "p_close1": [.6, .4], "f_open": [.8, .1],
            "f_close1": [.1, .8], "unit_profit_open": [1, -1], "unit_profit_close1": [-1, 2]})
        result = settle(rows, "open_near:12")
        np.testing.assert_allclose(result.fstar, [.5, .5])
        np.testing.assert_allclose(result.realized_return, [.5, 1])
        self.assertEqual(result.correct.sum(), 2)

    def test_probability_metrics(self):
        ll, brier = losses([.8, .2], [1, 0])
        np.testing.assert_allclose(ll, -np.log(.8))
        np.testing.assert_allclose(brier, .04)
