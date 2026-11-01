"""Placed-ticket accuracy and report-viewer contracts."""
import ast
import json
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Notebooks/Models/training'))
from train_lr_models import placed_parlay_accuracy


class ComparisonReportTests(unittest.TestCase):
    def test_counts_tickets_not_legs_and_excludes_refunds_and_unplaced(self):
        tickets = pd.DataFrame({
            'date': ['2026-01-01']*2 + ['2026-01-02']*2 + ['2026-01-03']*2 + ['2026-01-04']*2,
            'fstar_parlay': [.1]*6 + [0]*2,
            'parlay_net_odds': [2,2,-1,-1,0,0,-1,-1],
        })
        self.assertEqual(placed_parlay_accuracy(tickets, pd.DataFrame()), 50.)

    def test_no_placed_tickets_is_undefined(self):
        self.assertTrue(pd.isna(placed_parlay_accuracy(pd.DataFrame(), pd.DataFrame())))

    def test_zero_bankroll_events_do_not_count_as_placed(self):
        tickets = pd.DataFrame(dict(date=['2026-01-01', '2026-01-02'],
            fstar_parlay=[.1,.1], parlay_net_odds=[2,-1]))
        moneylines = pd.DataFrame(dict(date=['2026-01-01','2026-01-02'],
            bankroll_postevent=[120.,0.], event_payout_money_line=[0.,0.],
            parlay_net=[20.,0.]))
        self.assertEqual(placed_parlay_accuracy(tickets, moneylines), 100.)

    def test_manual_viewer_has_five_groups_and_separate_probability_tables(self):
        notebook = json.loads((ROOT/'Notebooks/Models/training/lr_open_close_manual.ipynb').read_text(encoding='utf-8'))
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        ast.parse(source)
        self.assertNotIn('trainval_val', source)
        self.assertNotIn('_metrics.png', source)
        self.assertEqual(source.count('_metrics.csv'), 10)
        self.assertEqual(source.count('_calibration.png'), 10)


if __name__ == '__main__':
    unittest.main()
