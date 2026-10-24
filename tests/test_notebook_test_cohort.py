"""Test exports must not depend on validation data left in notebook state."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd


class TestNotebookCohort(unittest.TestCase):
    def test_all_test_markets_use_the_fresh_test_csv(self):
        notebook = Path(__file__).resolve().parents[1] / 'Notebooks/Models/training/lr_open_close.ipynb'
        cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
        rows = pd.DataFrame({'date': ['2026-02-07', '2026-02-21', '2026-10-17'],
                             'marker': ['old', 'test-first', 'test-last']})
        infer = Mock()
        env = dict(pd=pd, date_col='date', TEST_START_DATE=pd.Timestamp('2026-02-21'),
                   df_inference=pd.DataFrame({'marker': ['stale-validation']}),
                   run_inference=infer, results_dir=Path('results'), plots_dir=Path('plots'),
                   config=SimpleNamespace(test_feats_file='fresh-test.csv',
                       model_open_test_path='model', model_open_test_se_path='se',
                       scaler_open_test_path='scaler', encoder_open_test_path='encoder'))
        with patch.object(pd, 'read_csv', return_value=rows) as read:
            for cell in cells:
                source = ''.join(cell.get('source', []))
                if 'pred_save_path=results_dir / \'test_logit_open_with_' in source:
                    exec(compile(source, str(notebook), 'exec'), env)
        read.assert_called_once_with('fresh-test.csv')
        self.assertEqual(infer.call_count, 3)
        self.assertEqual({c.kwargs['odds_type'] for c in infer.call_args_list},
                         {'open', 'close1', 'close2'})
        for call in infer.call_args_list:
            self.assertEqual(call.args[0].marker.tolist(), ['test-first', 'test-last'])
            self.assertTrue(call.kwargs['plot_title'].startswith('Test Results'))
