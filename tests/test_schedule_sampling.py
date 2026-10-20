import unittest

import pandas as pd

from ufc_betting.BettingStrategy.MonteCarlo.schedule_sampling import sample_schedule


class ScheduleSamplingTests(unittest.TestCase):
    def setUp(self):
        # Numbered cards have no titles; Fight Nights do. Classification must
        # use event names rather than the presence of a belt.
        self.fights = pd.DataFrame([
            {"date": date, "event_index": index, "title_fight": int(not title and index == 0),
             "event_name": f'UFC {300 + size}: Example' if title else f'UFC Fight Night: {date}',
             "fight_id": f"{date}-{index}", "payout": index + 10}
            for date, title, size in [
                ("2025-01-04", True, 3), ("2025-01-18", True, 2),
                ("2025-02-01", False, 4), ("2025-02-15", False, 3),
            ]
            for index in range(size)
        ])

    def test_monthly_guarantee_and_intact_events(self):
        original = self.fights.copy(deep=True)
        orders = set()
        for seed in range(30):
            sampled = sample_schedule(self.fights, random_state=seed)
            self.assertEqual(len(sampled), len(self.fights))
            self.assertCountEqual(sampled.fight_id, self.fights.fight_id)
            self.assertEqual(
                sampled.date.drop_duplicates().tolist(),
                sorted(pd.to_datetime(self.fights.date).unique()),
            )
            cards_by_date = sampled.groupby('date').first()
            monthly_numbered = cards_by_date.event_name.str.match(r'^UFC \d').groupby(cards_by_date.index.to_period('M')).sum()
            self.assertTrue((monthly_numbered == 1).all())
            cards = list(sampled.groupby("sampled_event_id", sort=False))
            order = tuple(card.source_date.iloc[0] for _, card in cards)
            orders.add(order)
            self.assertEqual(set(order), set(pd.to_datetime(self.fights.date)))
            self.assertEqual(len(set(order)), len(cards))
            for _, card in cards:
                template = self.fights[
                    pd.to_datetime(self.fights.date) == card.source_date.iloc[0]
                ]
                restored = card[self.fights.columns].copy()
                restored["date"] = card.source_date.dt.strftime("%Y-%m-%d")
                pd.testing.assert_frame_equal(
                    restored.reset_index(drop=True), template.reset_index(drop=True)
                )
                self.assertEqual(card.event_index.tolist(), template.event_index.tolist())
                self.assertEqual(card.title_fight.tolist(), template.title_fight.tolist())
            for row in sampled.itertuples():
                source = self.fights.iloc[row.source_row_position]
                self.assertEqual(row.fight_id, source.fight_id)
                self.assertEqual(row.event_index, source.event_index)
                self.assertEqual(row.title_fight, source.title_fight)
                self.assertEqual(row.payout, source.payout)
                self.assertEqual(row.source_date, pd.Timestamp(source.date))
        self.assertGreater(len(orders), 1)
        pd.testing.assert_frame_equal(self.fights, original)

    def test_reproducibility_without_fight_indices(self):
        expected = sample_schedule(self.fights, random_state=42)
        pd.testing.assert_frame_equal(expected, sample_schedule(self.fights, random_state=42))
        pd.testing.assert_frame_equal(
            expected.drop(columns="event_index"), sample_schedule(self.fights.drop(columns="event_index"), random_state=42),
            check_like=True,
        )

    def test_fewer_numbered_than_months_and_no_numbered(self):
        fights=self.fights.copy()
        fights.loc[fights.date.eq('2025-01-18'),'event_name']='UFC Fight Night: Example'
        sampled=sample_schedule(fights,random_state=12)
        self.assertCountEqual(sampled.fight_id,fights.fight_id)
        self.assertEqual(sampled.groupby('date').event_name.first().str.match(r'^UFC \d').sum(),1)
        fights['event_name']='UFC Fight Night: Example'
        sampled=sample_schedule(fights,random_state=12)
        self.assertCountEqual(sampled.fight_id,fights.fight_id)
        self.assertFalse(sampled.event_name.str.match(r'^UFC \d').any())

    def test_surplus_numbered_cards_are_balanced(self):
        dates = ['2025-01-04','2025-01-11','2025-01-18','2025-02-01','2025-02-08','2025-02-15']
        data = pd.DataFrame({'date': dates, 'event_type': ['numbered'] * 5 + ['fight_night']})
        for seed in range(10):
            result = sample_schedule(data, random_state=seed)
            counts = result.event_type.eq('numbered').groupby(result.date.dt.to_period('M')).sum()
            self.assertEqual(sorted(counts), [2, 3])
            self.assertEqual(result.source_date.nunique(), 6)

    def test_title_flag_alone_is_insufficient(self):
        with self.assertRaisesRegex(ValueError, 'event_name or event_type'):
            sample_schedule(self.fights.drop(columns='event_name'))

if __name__ == "__main__":
    unittest.main()
