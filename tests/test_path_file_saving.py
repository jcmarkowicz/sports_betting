import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling import _save_paths


def sharing_violation():
    error = PermissionError('File is in use')
    error.winerror = 32
    return error


class PathSavingTests(unittest.TestCase):
    def test_writer_closed_before_replace_and_transient_lock_retried(self):
        real_pickle, real_replace = pd.to_pickle, os.replace
        handles = []

        def write(value, file):
            handles.append(file)
            real_pickle(value, file)

        def replace(source, destination):
            self.assertTrue(handles[0].closed)
            if replace_mock.call_count == 1:
                raise sharing_violation()
            real_replace(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paths.pkl'
            with patch('pandas.to_pickle', side_effect=write), \
                 patch('os.replace', side_effect=replace) as replace_mock, patch('time.sleep'):
                _save_paths([pd.DataFrame({'a': [1]})], path)
            self.assertEqual(replace_mock.call_count, 2)
            self.assertEqual(pd.read_pickle(path)[0].a.tolist(), [1])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_replace_keeps_complete_recoverable_file_and_old_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paths.pkl'
            path.write_bytes(b'previous output')
            with patch('os.replace', side_effect=sharing_violation()), patch('time.sleep'):
                with self.assertRaises(PermissionError) as caught:
                    _save_paths([pd.DataFrame({'a': [1]})], path)
            self.assertIn('No regeneration is needed', caught.exception.__notes__[0])
            self.assertEqual(path.read_bytes(), b'previous output')
            saved = [p for p in Path(directory).iterdir() if p != path]
            self.assertEqual(len(saved), 1)
            self.assertEqual(pd.read_pickle(saved[0])[0].a.tolist(), [1])

    def test_cleanup_failure_does_not_mask_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paths.pkl'
            original = OSError('disk full')
            with patch('pandas.to_pickle', side_effect=original), \
                 patch.object(Path, 'unlink', side_effect=sharing_violation()), patch('time.sleep'):
                with self.assertRaises(OSError) as caught:
                    _save_paths([], path)
            self.assertIs(caught.exception, original)
            self.assertIn('incomplete temporary file', original.__notes__[0])
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
