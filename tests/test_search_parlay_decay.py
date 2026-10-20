import unittest
import tempfile
from pathlib import Path

import numpy as np

from test_search_exponential_decay import SearchDecayTests
from ufc_betting.Models.Search.parlay_decay import ParlayDecaySearch


class JointDecayTests(unittest.TestCase):
    def test_joint_search_uses_saved_paths_and_has_no_backtest_method(self):
        with tempfile.TemporaryDirectory() as root:
            helper = SearchDecayTests()
            saved = helper.make_search(bootstrap_mode='save', bootstrap_dir=Path(root) / 'paths')
            loaded = helper.make_search(bootstrap_mode='load', bootstrap_dir=Path(root) / 'paths')
            loaded.metadata['opening_params'].update(mdd_parlay=.3, N_parlay=500)
            loaded.metadata['closing_params'].update(mdd_parlay=.2, N_parlay=800)
            search = ParlayDecaySearch(loaded)
            np.testing.assert_array_equal(search.path_indices, saved.path_indices)
            result = search.run(n_trials=2)
            self.assertEqual(set(result['best_params']),
                             {'k_down', 'k_up', 'k_parlay_down', 'k_parlay_up'})
            self.assertFalse(hasattr(search, 'simulate_kelly'))
            events = result['event_results']
            np.testing.assert_allclose(events.event_return, events.moneyline_return + events.parlay_return)
            self.assertEqual(len(result['return_comparison']), 2)
            self.assertFalse(result['allocation_weights'].empty)


if __name__ == '__main__':
    unittest.main()
