import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder


class SavedSplitTests(unittest.TestCase):
    def test_three_date_splits_fit_training_only_and_save_full_rows(self):
        builder = self.builder()
        builder.df['bout_position'] = np.tile([2, 7], 10)
        with tempfile.TemporaryDirectory() as directory:
            pkt = builder.prepare_train_test(
                val_start_date='2026-01-06', test_start_date='2026-01-09',
                save_split=True, output_dir=directory,
            )
            self.assertEqual([len(pkt[f'X_{s}']) for s in ('train', 'val', 'test')], [10, 6, 4])
            self.assertAlmostEqual(pkt['scaler'].mean_[0], 4.5)
            for name in ('train', 'val', 'test'):
                saved = pd.read_csv(pkt['split_paths'][name])
                self.assertEqual(saved['feature'].tolist(), pkt[f'{name}_df']['feature'].tolist())
                self.assertEqual(saved['bout_position'].tolist(), pkt[f'{name}_df']['bout_position'].tolist())
                self.assertEqual(list(pkt[f'X_{name}']), ['feature'])
            self.assertEqual(pkt['dates_val'].min(), pd.Timestamp('2026-01-06'))
            self.assertEqual(pkt['dates_test'].min(), pd.Timestamp('2026-01-09'))
            self.assertEqual(pkt['split_summary'].percentage.tolist(), [50, 30, 20])

    def test_invalid_date_splits_and_colliding_exports(self):
        for kwargs in (
            dict(val_start_date='2026-01-06'),
            dict(val_start_date='2026-01-09', test_start_date='2026-01-06'),
            dict(val_start_date='2026-01-06', test_start_date='2027-01-01'),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.builder().prepare_train_test(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicate.csv'
            with self.assertRaisesRegex(ValueError, 'must be different'):
                self.builder().prepare_train_test(
                    val_start_date='2026-01-06', test_start_date='2026-01-09',
                    val_split_path=path, test_split_path=path,
                )
            self.assertFalse(path.exists())

    def test_sampling_metadata_survives_default_builder_and_saved_validation(self):
        prototype = self.builder()
        frame = prototype.all_processed_df.copy()
        for col in prototype.valid_cols + prototype.cat_cols:
            if col not in frame:
                frame[col] = 1
        frame['bout_position'] = np.tile([2., 7.], 10)
        frame.loc[19, 'bout_position'] = np.nan
        frame['event_name'] = 'UFC 300'
        builder = TrainTestBuilder(frame, ['feature'], 'winner', 'date', 'open', 2025, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'validation.csv'
            result = builder.prepare_train_test(scale=True, test_split_path=path)
            saved = pd.read_csv(path)
        self.assertEqual(len(builder.df), len(frame))
        np.testing.assert_allclose(saved['bout_position'], [2, 7, 2, np.nan], equal_nan=True)
        self.assertEqual(saved['event_name'].tolist(), ['UFC 300'] * 4)
        self.assertEqual(list(result['X_test']), ['feature'])

    def test_explicit_paths_save_unscaled_rows_even_when_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / 'training' / 'raw.csv'
            test_path = Path(directory) / 'validation' / 'raw.csv'
            result = self.builder().prepare_train_test(
                scale=True, train_split_path=train_path, test_split_path=test_path,
            )
            for path, key in ((train_path, 'train_df'), (test_path, 'test_df')):
                saved = pd.read_csv(path, parse_dates=['date'])
                pd.testing.assert_frame_equal(saved, result[key].reset_index(drop=True))
            self.assertEqual(result['split_paths'], {'train': train_path, 'val': test_path})

    def test_single_explicit_path_and_duplicate_path_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.csv'
            result = self.builder().prepare_train_test(train_split_path=path)
            self.assertEqual(result['split_paths'], {'train': path})
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with self.assertRaisesRegex(ValueError, 'must be different'):
                self.builder().prepare_train_test(train_split_path=path, test_split_path=path)

    def builder(self):
        frame = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=10).repeat(2),
            'feature': np.arange(20), 'winner': np.tile([0, 1], 10),
            'fighter_red': [f'fighter_{i}' for i in range(20)],
        })
        return TrainTestBuilder(frame, ['feature'], 'winner', 'date', 'open',
                                2025, 1, 1, required_features_only=True)

    def test_default_split_saves_train_and_validation_full_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('ufc_betting.Models.LogisticRegression.train_test_builder.settings',
                       SimpleNamespace(data_dir=root)):
                result = self.builder().prepare_train_test(scale=False, save_split=True)
            self.assertEqual(len(result['train_df']), 16)
            self.assertEqual(len(result['test_df']), 4)
            for name, key in [('train', 'train_df'), ('val', 'test_df')]:
                path = root / 'train_val_test' / 'split' / f'{name}_open.csv'
                self.assertEqual(result['split_paths'][name], path)
                saved = pd.read_csv(path, parse_dates=['date'])
                pd.testing.assert_frame_equal(saved, result[key].reset_index(drop=True))
            self.assertFalse((root / 'train_val_test' / 'split' / 'test_open.csv').exists())

    def test_saving_is_optional_and_event_boundary_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'custom'
            result = self.builder().prepare_train_test(.75, scale=False, output_dir=target)
            self.assertFalse(target.exists())
            self.assertNotIn('split_paths', result)
            self.assertTrue(set(result['dates_train']).isdisjoint(result['dates_test']))
            result = self.builder().prepare_train_test(
                .75, scale=False, save_split=True, output_dir=target)
            self.assertEqual(len(pd.read_csv(result['split_paths']['train'])), 14)
            self.assertEqual(len(pd.read_csv(result['split_paths']['val'])), 6)


if __name__ == '__main__':
    unittest.main()
