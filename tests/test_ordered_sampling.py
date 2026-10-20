import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling import sample_ordered_fights
from ufc_betting.BettingStrategy.MonteCarlo.schedule_sampling import sample_schedule
from ufc_betting.BettingStrategy.MonteCarlo.event_types import event_types


class OrderedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            dict(date=date, event_name=name, bout_position=position,
                 fight_id=f'{date}-{position}', winner=position % 2,
                 proba_red=.1 * position, title_fight=int('Night' in name))
            for date, name, positions in [
                ('2025-01-04', 'UFC 300: A vs B', [1, 2, 4]),
                ('2025-01-18', 'UFC 301: C vs D', [1, 2, 3, 4]),
                ('2025-02-01', 'UFC Fight Night: E vs F', [1, 2, 4]),
                ('2025-02-15', 'UFC Fight Night: G vs H', [1, 2, 3, 4]),
            ] for position in positions
        ])

    def test_uses_scheduler_and_preserves_slots_and_donor_data(self):
        original = self.data.copy(deep=True)
        crossed = False
        for seed in range(10):
            with patch('ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling.sample_schedule',
                       wraps=sample_schedule) as scheduler:
                result = sample_ordered_fights(self.data, random_state=seed)
                scheduler.assert_called_once()
                self.assertFalse(scheduler.call_args.kwargs['replace'])
            self.assertEqual(len(result), len(self.data))
            cards = result.groupby('sampled_event_id').first()
            self.assertEqual(cards.template_date.nunique(), 4)
            counts = cards.event_type.eq('numbered').groupby(cards.date.dt.to_period('M')).sum()
            self.assertEqual(counts.tolist(), [1, 1])
            for row in result.itertuples():
                donor = self.data.iloc[row.source_row_position]
                template = self.data.iloc[row.template_row_position]
                self.assertEqual(row.bout_position, donor.bout_position)
                self.assertEqual(row.bout_position, template.bout_position)
                self.assertEqual(row.event_name, template.event_name)
                self.assertEqual(row.source_event_name, donor.event_name)
                self.assertEqual(row.source_date, pd.Timestamp(donor.date))
                self.assertEqual(row.event_type, event_types(donor.to_frame().T).iloc[0])
                self.assertEqual(row.fight_id, donor.fight_id)
                self.assertEqual(row.winner, donor.winner)
                self.assertEqual(row.proba_red, donor.proba_red)
                crossed |= row.source_date != row.template_date
            for _, card in result.groupby('sampled_event_id'):
                self.assertTrue(card.bout_position.is_monotonic_increasing)
        self.assertTrue(crossed)
        pd.testing.assert_frame_equal(self.data, original)

    def test_reproducible_and_works_with_explicit_types(self):
        a = sample_ordered_fights(self.data, random_state=42)
        pd.testing.assert_frame_equal(a, sample_ordered_fights(self.data, random_state=42))
        data = self.data.assign(event_type=event_types(self.data)).drop(columns='event_name')
        b = sample_ordered_fights(data, random_state=42)
        self.assertEqual(a.source_row_position.tolist(), b.source_row_position.tolist())

    def test_requires_actual_positions_and_unambiguous_events(self):
        for data in (self.data.drop(columns='bout_position'),
                     self.data.assign(bout_position=1),
                     self.data.assign(bout_position=1.5),
                     self.data.assign(bout_position=np.nan),
                     self.data.assign(event_name='Unknown promotion')):
            with self.subTest(columns=data.columns.tolist()), self.assertRaises(ValueError):
                sample_ordered_fights(data)

    def test_other_event_families_remain_separate(self):
        data = self.data.copy()
        data.loc[data.date.eq('2025-02-15'), 'event_name'] = 'UFC on ABC: Example'
        result = sample_ordered_fights(data, random_state=42)
        abc = result.loc[result.event_type.eq('ufc_on_abc')]
        self.assertTrue(abc.source_event_name.eq('UFC on ABC: Example').all())


if __name__ == '__main__':
    unittest.main()
