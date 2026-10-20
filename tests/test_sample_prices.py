import unittest
import numpy as np
from ufc_betting.BettingStrategy.Convergence.odds_scenarios import sample_prices, sample_movements


class SamplePricesTests(unittest.TestCase):
    def test_quotes_match_grid_and_both_sides_share_movement(self):
        opening = np.array([[2., 3.], [1.8, 2.4]])
        closing = np.array([[1.5, 4.], [2.2, 2.]])
        result = sample_prices(opening, closing, 200, random_state=42)
        indices = sample_movements(2, 200, np.random.default_rng(42))
        np.testing.assert_array_equal(result['movement_indices'], indices)
        np.testing.assert_allclose(result['prices'][0], closing)
        np.testing.assert_allclose(result['prices'][100], opening)
        np.testing.assert_allclose(
            (result['prices'] - closing) / (opening - closing),
            np.repeat((indices / 100)[:, :, None], 2, axis=2))

    def test_single_fight_has_only_one_scenario_per_grid_point(self):
        result = sample_prices([[2., 2.]], [[1.5, 3.]], 1000, levels=11, random_state=1)
        self.assertEqual(result['prices'].shape, (11, 1, 2))
        repeated = sample_prices([[2., 2.]], [[1.5, 3.]], 1000, levels=11, random_state=1)
        np.testing.assert_array_equal(result['prices'], repeated['prices'])

    def test_rejects_invalid_prices_and_sample_counts(self):
        for opening, closing, count in (([[1., 2.]], [[2., 2.]], 101),
                                         ([[2., 2.]], [[2., 2.], [2., 2.]], 101),
                                         ([[2., 2.]], [[2., 2.]], 10),
                                         ([[2., 2.]], [[2., 2.]], 101.5)):
            with self.assertRaises(ValueError):
                sample_prices(opening, closing, count)


if __name__ == '__main__':
    unittest.main()
