import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling import generate_ordered_paths
from ufc_betting.BettingStrategy.MonteCarlo.sampled_paths import attach_path_predictions


class SharedValidationPathsTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            dict(date=date, event_name='UFC 300', bout_position=pos,
                 fighter_red=f'red_{i}_{pos}', fighter_blue=f'blue_{i}_{pos}', winner=pos % 2)
            for i, date in enumerate(['2025-01-04', '2025-02-01', '2025-03-01'])
            for pos in (1, 2)
        ])
        for market in ('open', 'close1', 'close2'):
            for prefix in ('dec', 'dec_fair'):
                for side in ('red', 'blue'):
                    self.frame[f'{prefix}_{market}_{side}'] = 2.
        self.paths = generate_ordered_paths(self.frame, odds_type=None, n_paths=2, progress_every=0)
        self.predictions = self.frame.assign(proba_red=np.linspace(.2, .8, 6),
                                              proba_se=np.linspace(.01, .06, 6))

    def test_choice_preserves_paths_and_matches_donors_not_csv_order(self):
        original = [p.copy(deep=True) for p in self.paths]
        for market, delta in [('open', 0), ('close1', .01), ('close2', .02)]:
            predictions = self.predictions.copy()
            predictions['proba_red'] += delta
            predictions['proba_se'] += delta
            chosen = attach_path_predictions(self.paths, predictions.iloc[::-1], odds_type=market)
            for before, after in zip(original, chosen):
                donors = before.source_row_position.to_numpy()
                np.testing.assert_allclose(after.proba_red, predictions.proba_red.to_numpy()[donors])
                np.testing.assert_allclose(after.proba_se, predictions.proba_se.to_numpy()[donors])
                pd.testing.assert_frame_equal(before[['date', 'sampled_event_id', 'source_row_position']],
                                              after[['date', 'sampled_event_id', 'source_row_position']])
        for before, after in zip(original, self.paths):
            pd.testing.assert_frame_equal(before, after)
            self.assertNotIn('proba_red', after)

    def test_missing_duplicate_and_mismatched_predictions_raise(self):
        donor = self.paths[0].source_row_position.iloc[0]
        for predictions, error in [
            (self.predictions.drop(index=donor), 'Missing predictions'),
            (pd.concat([self.predictions, self.predictions.iloc[:1]]), 'unique'),
            (self.predictions.assign(winner=9), 'outcomes'),
            (self.predictions.assign(dec_close1_red=3.), 'prices'),
            (self.predictions.assign(proba_se=-1), 'prediction_se'),
        ]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                attach_path_predictions(self.paths, predictions, odds_type='close1')

    def test_stale_aliases_cannot_override_selected_predictions(self):
        paths = [p.assign(choice_proba=.99, se=4., risk_proba_red=.99) for p in self.paths]
        chosen = attach_path_predictions(paths, self.predictions, odds_type='close2')
        for path in chosen:
            self.assertNotIn('choice_proba', path)
            self.assertNotIn('se', path)
            np.testing.assert_array_equal(path.risk_proba_red, path.proba_red)

    def test_subset_is_reproducible_unique_and_preserves_selected_paths(self):
        source = generate_ordered_paths(self.frame, odds_type=None, n_paths=8, progress_every=0)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'paths.pkl'
            pd.to_pickle(source, file)
            first = attach_path_predictions(file, self.predictions, odds_type='open', n_paths=3, random_state=42)
            second = attach_path_predictions(source, self.predictions, odds_type='close1', n_paths=3, random_state=42)
            self.assertEqual(len(pd.read_pickle(file)), 8)
        ids = [p.attrs['generation']['path_id'] for p in first]
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ids, [p.attrs['generation']['path_id'] for p in second])
        for i, path in zip(ids, first):
            pd.testing.assert_frame_equal(path[source[i].columns], source[i])
        self.assertEqual(len(attach_path_predictions(source, self.predictions, odds_type='open')), 8)

    def test_invalid_subset_size_raises(self):
        for count in (0, -1, 3, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'n_paths'):
                attach_path_predictions(self.paths, self.predictions, odds_type='open', n_paths=count)


if __name__ == '__main__':
    unittest.main()
