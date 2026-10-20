import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.Models.Search.frozen_betting import JointDecayBettingSearch, terminal_return_score
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate_bankroll_paths


class JointKellyDecayTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame(dict(
            date=np.repeat(['2025-01-01', '2025-02-01', '2025-03-01'], 2),
            fighter_red=list('abcdef'), fighter_blue=list('ghijkl'),
            winner=[1, 1, 0, 1, 1, 0],
            proba_red=[.75, .65, .3, .8, .7, .4], prediction_se=[.02] * 6,
        ))
        for i, stage in enumerate(('open', 'close1', 'close2')):
            frame[f'dec_{stage}_red'] = [2., 1.9, 2.5, 1.5, 2.1, 2.6]
            frame[f'dec_{stage}_blue'] = [1.9, 2., 1.6, 2.8, 1.8, 1.5]
            for side in ('blue', 'red'):
                frame[f'dec_{stage}_{side}'] += i * .1
                frame[f'dec_fair_{stage}_{side}'] = frame[f'dec_{stage}_{side}'] * 1.04
        return frame

    def test_separate_calls_save_replay_and_match_backtest(self):
        frame = self.frame()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = dict(n_trials=2, k=4, parallel=False, params_dir=root / 'params',
                          bootstrap_dir=root / 'paths')
            opening = optimize_kelly_params(validation_data={'open': frame},
                                            bootstrap_mode='save', **common)
            self.assertIsNone(opening['betting_params'].decay_rates)
            self.assertEqual(len(opening['studies']['full_validation'].best_params), 6)
            original = (root / 'params/open.json').read_text()
            args = dict(use_decay=True,
                        opening_params=root / 'params/open.json', opening_data=frame.iloc[::-1], **common)
            closing = {market: optimize_kelly_params(validation_data={market: frame}, **args, bootstrap_mode='save')
                       for market in ('close1', 'close2')}
            with patch('ufc_betting.Models.Search.frozen_betting.sample_schedule',
                       side_effect=AssertionError('Must reuse stored schedules')):
                replay = {market: optimize_kelly_params(validation_data={market: frame}, **args, bootstrap_mode='load')
                          for market in ('close1', 'close2')}
            self.assertEqual((root / 'params/open.json').read_text(), original)
            for market, result in closing.items():
                study = result['studies']['full_validation']
                self.assertEqual(result['odds_type'], market)
                self.assertEqual(result['best_trial_number'], study.best_trial.number)
                self.assertEqual(len(study.best_params), 10)
                self.assertEqual(result['best_params'], replay[market]['best_params'])
                saved = BettingParams.load(root / f'params/{market}.json')
                self.assertEqual(saved.to_dict(), result['best_params'])
                self.assertEqual(saved.opening_params.to_dict(), opening['betting_params'].to_dict())
                self.assertEqual(set(saved.decay_rates), set(JointDecayBettingSearch.rate_names))
                data, kwargs = result['kelly_sim_input'].to_simulation_args()
                bets, _ = simulate_kelly(data, **kwargs)
                np.testing.assert_allclose(bets.groupby('date').bankroll_postevent.first(),
                                           result['event_results'].bankroll, atol=1e-10)
                indices = np.load(root / f'paths/{market}/paths.npy')
                sim = simulate_bankroll_paths(result['event_results'], indices, 500)
                self.assertAlmostEqual(result['best_value'], terminal_return_score(sim['terminal_returns']))
                self.assertLessEqual(result['event_results'].exposure.max(), 1 + 1e-12)
            self.assertEqual({p.name for p in root.iterdir()}, {'params', 'paths'})

    def test_alignment_and_mode_errors(self):
        frame = self.frame()
        params = BettingParams()
        with patch('ufc_betting.Models.Search.engine.FrozenBettingSearch') as search:
            for data in ({}, {'open': frame, 'close1': frame}, {'invalid': frame}):
                with self.assertRaisesRegex(ValueError, 'exactly one market'):
                    optimize_kelly_params(validation_data=data)
            search.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'closing markets'):
            optimize_kelly_params(validation_data={'open': frame}, use_decay=True, opening_params=params)
        with self.assertRaisesRegex(ValueError, 'same fights'):
            JointDecayBettingSearch(frame, 'close1', opening_params=params, opening_data=frame.iloc[:-1], k=2)
        wrong = frame.copy()
        wrong.loc[0, 'winner'] = 0
        with self.assertRaisesRegex(ValueError, 'same fights'):
            JointDecayBettingSearch(frame, 'close1', opening_params=params, opening_data=wrong, k=2)
        with self.assertRaisesRegex(ValueError, 'without decay'):
            JointDecayBettingSearch(frame, 'close1', k=2,
                opening_params=params.with_decay(dict(k_down=1, k_up=1), params))

    def test_joint_trials_use_both_closing_controls_and_decay(self):
        search = JointDecayBettingSearch(self.frame(), 'close1', opening_params=BettingParams(), k=3)
        params = dict(mdd=.3, N=500, mdd_parlay=.3, N_parlay=500, z=.2, moneyline_min_american=-300)
        params.update({'log1p_' + key: np.log1p(20.) for key in search.rate_names})
        before = search.path_indices.copy()
        base = search.event_returns(params).event_return.to_numpy()
        changed_sizing = search.event_returns(dict(params, mdd=.05)).event_return.to_numpy()
        changed_decay = search.event_returns(dict(params, log1p_k_down=0., log1p_k_up=0.)).event_return.to_numpy()
        self.assertFalse(np.allclose(base, changed_sizing))
        self.assertFalse(np.allclose(base, changed_decay))
        np.testing.assert_array_equal(search.path_indices, before)


if __name__ == '__main__':
    unittest.main()
