import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import test_joint_kelly_decay_search as fixtures
from ufc_betting.Models.Search.frozen_betting import terminal_return_score
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.BettingStrategy.betting_params import BettingParams
from Notebooks.Search.run_search import run_search
from ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling import generate_ordered_paths


class LogGrowthTests(unittest.TestCase):
    def test_floor_and_average_logs_not_log_average(self):
        returns = np.array([-1.2, -1., -.5, 1.])
        original = returns.copy()
        expected = np.log([.001, .001, .5, 2.]).mean()
        actual = terminal_return_score(returns, 100, score_type='log_growth', wealth_floor=.001)
        self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(actual, terminal_return_score(returns, 0, score_type='log_growth'))
        np.testing.assert_array_equal(returns, original)
        self.assertAlmostEqual(terminal_return_score([-.5, 1.], score_type='log_growth'), 0.)
        self.assertGreater(terminal_return_score([-.5, 2.], score_type='log_growth'), 0.)

    def test_legacy_and_validation(self):
        returns = np.array([-.2, .4, .8])
        self.assertAlmostEqual(terminal_return_score(returns, .3), returns.mean()-.3*returns.std(ddof=1))
        for floor in (0, -1, 1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                terminal_return_score(returns, score_type='log_growth', wealth_floor=floor)
        with self.assertRaises(ValueError):
            terminal_return_score(returns, score_type='bad')

    def test_open_and_closing_objective_metadata(self):
        frame = fixtures.JointKellyDecayTests().frame()
        for market in ('open', 'close1'):
            extra = {} if market == 'open' else dict(use_decay=True, opening_params=BettingParams())
            result = optimize_kelly_params(validation_data={market: frame}, k=4, n_trials=2,
                score_type='log_growth', wealth_floor=.02, **extra)
            study = result['studies']['full_validation']
            self.assertEqual(study.user_attrs['score_type'], 'log_growth')
            self.assertEqual(result['wealth_floor'], .02)
            self.assertAlmostEqual(result['objective_breakdown'].mean_log_growth.iloc[0], study.best_value)
            self.assertNotIn('objective_pct', result['objective_breakdown'])

    def test_run_search_saved_paths_and_pickle(self):
        frame = fixtures.JointKellyDecayTests().frame().assign(bout_position=[1,2]*3, event_name='UFC 300')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root/'predictions.csv'; frame.to_csv(csv,index=False)
            paths = generate_ordered_paths(frame,n_paths=3,odds_type=None,storage='compact',
                                          output_path=root/'paths',progress_every=0)
            paths.close()
            result = run_search('open',paths_path=root/'paths',predictions_path=csv,params_dir=root/'params',
                                n_paths=3,n_trials=2,score_type='log_growth',wealth_floor=.01)
            expected = np.log(np.maximum(result['sampled_path_results'].terminal_return.to_numpy()+1,.01)).mean()
            self.assertAlmostEqual(result['best_value'],expected)
            saved = pd.read_pickle(root/'params/search_results/open_search_results.pkl')
            self.assertEqual(saved['score_type'],'log_growth')
            self.assertEqual(saved['wealth_floor'],.01)


if __name__ == '__main__':
    unittest.main()
