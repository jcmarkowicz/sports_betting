import unittest

import numpy as np

from ufc_betting.BettingStrategy.MonteCarlo.mc_analysis import (
    analyze_bankroll_targets, bankroll_target_distribution, summarize_simulations,
)


class SimulationSummaryTests(unittest.TestCase):
    def test_severe_drawdown_episodes_require_full_recovery_to_reset(self):
        _, details = summarize_simulations({'sample': (None, None, [
            [100, 50, 70, 40, 100, 50, 60],
            [100, 110, 100, 90, 100, 110, 120],
            [100, 0, 0, 0, 0, 0, 0],
        ])}, return_path_statistics=True)
        np.testing.assert_array_equal(details['Drawdown episodes >= 50%'], [2, 0, 1])

    def test_returns_drawdowns_and_path_mean_sharpe_including_ruin(self):
        paths = [[100, 120, 90, 120], [100, 80, 40, 60], [100, -20, -20, -20]]
        summary, details = summarize_simulations(
            {'intermediate': (None, None, paths)}, return_path_statistics=True,
        )
        row = summary.iloc[0]
        self.assertAlmostEqual(row['Mean ending bankroll'], 160 / 3)
        self.assertNotIn('Mean total profit', row.index)
        self.assertEqual(row['P25 ending bankroll'], 20)
        self.assertAlmostEqual(row['P05 ending bankroll'], -12)
        self.assertEqual(row['P75 ending bankroll'], 90)
        np.testing.assert_allclose(details['Total return (%)'], [20, -40, -120])
        np.testing.assert_allclose(details['Max drawdown (%)'], [25, 60, 120])
        np.testing.assert_array_equal(details['Longest drawdown (settlements)'], [1, 3, 3])
        self.assertAlmostEqual(row['Mean max drawdown (%)'], 205 / 3)
        self.assertEqual(row['Worst max drawdown (%)'], 120)
        self.assertAlmostEqual(row['Ruined paths (%)'], 100 / 3)
        self.assertAlmostEqual(row['Paths with drawdown >= 50% (%)'], 200 / 3)
        averages = np.array([(.2 - .25 + 1/3)/3, (-.2 - .5 + .5)/3, -1.2/3])
        self.assertAlmostEqual(row['Sharpe (path-average event returns)'], averages.mean() / averages.std(ddof=1))
        np.testing.assert_allclose(details['Average event return (%)'], 100 * averages)
        ulcers = np.sqrt(np.mean(np.array([[0, 25, 0], [20, 60, 40], [120, 120, 120]]) ** 2, axis=1))
        np.testing.assert_allclose(details['Ulcer index (%)'], ulcers)
        self.assertAlmostEqual(row['Mean ulcer index (%)'], ulcers.mean())
        np.testing.assert_array_equal(details['Time underwater (settlements)'], [1, 3, 3])
        self.assertAlmostEqual(row['Mean time underwater (%)'], 100 * 7/9)

    def test_recovery_resets_duration_and_constant_returns_have_no_sharpe(self):
        summary, details = summarize_simulations({
            'recovery': (None, None, [[100, 80, 90, 100, 90]]),
            'flat': (None, None, [[100, 100, 100]]),
            'initial_only': (None, None, [[100]]),
        }, return_path_statistics=True)
        self.assertEqual(details.iloc[0]['Longest drawdown (settlements)'], 2)
        self.assertTrue(summary.iloc[1:]['Sharpe (path-average event returns)'].isna().all())
        self.assertEqual(summary.iloc[2]['Mean max drawdown (%)'], 0)

    def test_batched_returns_use_original_event_count(self):
        summary, details = summarize_simulations(
            {'batch': (None, None, [[100, 120, 132], [100, 80, 88]])},
            batch_size=3, event_count=4, return_path_statistics=True)
        np.testing.assert_allclose(details['Average event return (%)'], [7.5, -2.5])
        self.assertAlmostEqual(summary.iloc[0]['Sharpe (path-average event returns)'],
                               .025 / np.std([.075, -.025], ddof=1))
        with self.assertRaises(ValueError):
            summarize_simulations({'batch': (None, None, [[100, 120]])}, batch_size=3)

    def test_invalid_paths_are_rejected(self):
        for paths in ([], [[0, 1]], [[100, np.nan]], [100, 200]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                summarize_simulations({'bad': (None, None, paths)})

    def test_target_summary_includes_mean_bankroll_and_return(self):
        summary, _ = analyze_bankroll_targets(
            {'intermediate': [[100, 200], [100, 50]]}, [150], event_count=1,
        )
        self.assertEqual(summary.iloc[0]['Mean ending bankroll'], 125)
        self.assertEqual(summary.iloc[0]['Mean total return (%)'], 25)


class BankrollTargetTests(unittest.TestCase):
    def test_first_passage_includes_paths_that_later_drop(self):
        result = bankroll_target_distribution([
            [500, 1000, 400, 1200],
            [500, 700, 1100, 900],
            [500, 600, 800, 900],
            [500, 1000, 1100, 1200],
        ], 1000)
        self.assertEqual(result["probability_reached"], .75)
        np.testing.assert_allclose(result["events_to_target"], [1, 2, np.nan, 1])
        distribution = result["distribution"]
        np.testing.assert_array_equal(distribution["count"], [0, 2, 1, 0])
        np.testing.assert_allclose(distribution.probability, [0, .5, .25, 0])
        np.testing.assert_allclose(distribution.conditional_probability, [0, 2/3, 1/3, 0])
        np.testing.assert_allclose(distribution.cumulative_probability, [0, .5, .75, .75])

    def test_target_already_met(self):
        result = bankroll_target_distribution([[500, 200], [500, 600]], 500)
        self.assertEqual(result["probability_reached"], 1)
        self.assertEqual(result["events_to_target"].tolist(), [0, 0])

    def test_no_path_reaches_target(self):
        result = bankroll_target_distribution([[500, 600], [500, 400]], 1000)
        self.assertEqual(result["probability_reached"], 0)
        self.assertTrue(result["events_to_target"].isna().all())
        self.assertTrue(result["distribution"].conditional_probability.isna().all())
        self.assertEqual(result["distribution"].probability.sum(), 0)

    def test_batch_counts_include_partial_final_batch(self):
        result = bankroll_target_distribution(
            [[500, 600, 800, 1000], [500, 1000, 900, 800]], 1000,
            event_counts=[0, 3, 6, 8],
        )
        self.assertEqual(result["events_to_target"].tolist(), [8, 3])


if __name__ == "__main__":
    unittest.main()
