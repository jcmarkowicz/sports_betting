import unittest
import numpy as np
from ufc_betting.Models.RiskManagement.uneven_odds_sensitivity import sample_movements, scenario_totals


class UnevenTests(unittest.TestCase):
    def test_unique_and_reproducible(self):
        a = sample_movements(4, 1000, np.random.default_rng(42))
        b = sample_movements(4, 1000, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 1000)
        self.assertEqual(len(np.unique(a, axis=0)), 1000)
        self.assertTrue(((a >= 0) & (a <= 100)).all())
        self.assertEqual(np.all(a == a[:, :1], axis=1).sum(), 101)

    def test_uneven_can_exceed_uniform_range(self):
        ev = np.array([[1, 5], [5, 1]])
        vectors = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        np.testing.assert_array_equal(scenario_totals(ev, vectors), [6, 6, 10, 2])

    def test_single_fight_exhausts_grid(self):
        result = sample_movements(1, 5000, np.random.default_rng(42))
        self.assertEqual(len(result), 101)
