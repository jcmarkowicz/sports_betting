import unittest

import pandas as pd

from ufc_betting.BettingStrategy.MonteCarlo.event_types import event_types
from ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling import sample_ordered_fights


class NocheEventTests(unittest.TestCase):
    def test_noche_names_and_explicit_override(self):
        frame = pd.DataFrame({'event_name': [
            'Noche UFC: Lopes vs. Silva', ' noche ufc: A vs. B ',
            'UFC 306: Noche UFC', 'UFC Fight Night: A vs. B',
            'UFC Freedom 250', 'ufc noche: A vs. B',
        ]})
        self.assertEqual(event_types(frame).tolist(),
                         ['numbered', 'numbered', 'numbered', 'fight_night',
                          'numbered', 'numbered'])
        self.assertEqual(event_types(frame.assign(event_type='fight_night')).tolist(),
                         ['fight_night'] * 6)
        self.assertEqual(event_types(frame.assign(event_type='noche_ufc')).tolist(),
                         ['numbered'] * 6)
        with self.assertRaises(ValueError):
            event_types(pd.DataFrame({'event_name': ['Unknown event']}))

    def test_noche_sampling_preserves_family_and_bout_position(self):
        frame = pd.DataFrame([
            dict(date=date, event_name=name, bout_position=position)
            for date, name in [('2025-09-13', 'Noche UFC: A vs. B'),
                               ('2026-09-12', 'Noche UFC: C vs. D'),
                               ('2026-09-19', 'UFC Fight Night: E vs. F')]
            for position in (1, 2)
        ])
        sampled = sample_ordered_fights(frame, random_state=42)
        self.assertEqual(len(sampled), len(frame))
        families = event_types(frame)
        for row in sampled.itertuples():
            self.assertEqual(row.event_type, families.iloc[row.source_row_position])
            self.assertEqual(row.bout_position,
                             frame.iloc[row.source_row_position].bout_position)

    def test_special_cards_participate_in_monthly_balance(self):
        frame = pd.DataFrame([
            dict(date=date, event_name=name, bout_position=1)
            for date, name in [
                ('2026-01-03', 'UFC 300: A vs. B'),
                ('2026-01-10', 'Noche UFC: C vs. D'),
                ('2026-01-17', 'UFC Freedom 250'),
                ('2026-02-07', 'UFC Fight Night: E vs. F'),
                ('2026-03-07', 'UFC Fight Night: G vs. H'),
            ]
        ])
        for seed in range(5):
            sampled = sample_ordered_fights(frame, random_state=seed)
            monthly = sampled.loc[sampled.event_type.eq('numbered')]
            self.assertEqual(monthly.groupby(monthly.date.dt.to_period('M')).size().tolist(),
                             [1, 1, 1])
            self.assertEqual(set(monthly.event_name), set(frame.event_name.iloc[:3]))


if __name__ == '__main__':
    unittest.main()
