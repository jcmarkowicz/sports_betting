import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from test_adjusted_performance_rates import fight_history
from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.chronology import chronological_fights
from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.rolling_stats import apply_rolling_stats
from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.final_feats_df import non_rolling_stats
from ufc_betting.config import config
from ufc_betting.DataPipeline.FeatureEngineering.features_pipeline import FeatureEngineering
from ufc_betting.DataPipeline.webscrapers import odds_scraper


class FeatureIntegrityTests(unittest.TestCase):
    def test_card_order_and_history_are_independent_of_input_order(self):
        frame = fight_history()
        frame['bout_position'] = [3, 2, 1, 2, 1]
        frame.loc[1:2, 'date'] = frame.loc[0, 'date']
        expected = apply_rolling_stats(frame)
        actual = apply_rolling_stats(frame.sample(frac=1, random_state=42))
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(chronological_fights(frame).bout_position.tolist(), [3, 2, 1, 2, 1])

    def test_appending_future_bouts_does_not_change_rolling_history(self):
        frame = fight_history()
        expected = apply_rolling_stats(frame.iloc[:4])
        actual = apply_rolling_stats(frame).iloc[:4]
        pd.testing.assert_frame_equal(actual, expected)

    def test_selected_stats_do_not_use_current_or_future_results(self):
        frame = fight_history()
        changed = frame.copy()
        changed.loc[3:, 'winner'] = 'B'
        changed.loc[3:, 'sig_str_landed_red'] = 90.
        changed.loc[3:, 'sig_str_absorbed_blue'] = 90.
        changed.loc[3:, 'td_landed_red'] = 9.
        columns = [c for c in config.open_feats if not c.startswith('proba_fair')]
        expected = non_rolling_stats(apply_rolling_stats(frame))[columns]
        actual = non_rolling_stats(apply_rolling_stats(changed))[columns]
        np.testing.assert_allclose(actual.iloc[:4].astype(float), expected.iloc[:4].astype(float), equal_nan=True)
        self.assertFalse(np.allclose(actual.iloc[4].astype(float), expected.iloc[4].astype(float), equal_nan=True))

    def test_revised_quotes_keep_first_row_without_raising(self):
        row = dict(red_clean='a', blue_clean='b', date=pd.Timestamp('2026-01-01'))
        row.update({f'{market}_{side}': -110 for market in ('open', 'close1', 'close2')
                    for side in ('red', 'blue')})
        frame = pd.DataFrame([row, dict(row, open_red=200)])
        for data in (frame, frame.iloc[::-1]):
            actual = FeatureEngineering.unique_fight_odds(data)
            pd.testing.assert_frame_equal(actual, data.iloc[:1].reset_index(drop=True))
        self.assertEqual(len(FeatureEngineering.unique_fight_odds(pd.DataFrame([row, row]))), 1)

    def test_name_matching_rejects_character_overlap(self):
        self.assertFalse(odds_scraper.names_match('Chris Curtis', 'Christian Leroy Duncan'))
        self.assertFalse(odds_scraper.names_match('Andre Fili', 'Andre Fialho'))
        self.assertFalse(odds_scraper.names_match('', 'Some Fighter'))
        self.assertTrue(odds_scraper.names_match('Some Fighter Jr.', 'Some Fighter'))

    def test_odds_must_match_opponent_and_requested_date(self):
        driver, red, blue = Mock(), Mock(), Mock()
        driver.find_elements.return_value = [red]
        red.find_element.side_effect = [Mock(text='Fighter'), blue]
        blue.find_element.return_value = Mock(text='Opponent')
        blue.find_elements.side_effect = [[Mock(text='Jan 1, 2026')],
                                         [Mock(text='+110')] * 3]
        red.find_elements.return_value = [Mock(text='-120')] * 3
        result = odds_scraper.scrape_odds(driver, ['Opponent'], ['2025-01-01'])
        self.assertEqual(result['open_red'], [])
        red.find_element.side_effect = [Mock(text='Fighter'), blue]
        blue.find_elements.side_effect = [[Mock(text='Jan 1, 2026')],
                                         [Mock(text='+110')] * 3]
        result = odds_scraper.scrape_odds(driver, ['Opponent'], ['2026-01-02'])
        self.assertEqual(result['open_red'], ['-120'])

    def test_direct_page_for_wrong_fighter_is_rejected(self):
        driver, red = Mock(), Mock()
        driver.find_elements.return_value = [red]
        red.find_element.return_value = Mock(text='Andre Fialho')
        result = odds_scraper.scrape_odds(driver, ['Opponent'], ['2026-01-01'], expected_red='Andre Fili')
        self.assertEqual(result['open_red'], [])


if __name__ == '__main__':
    unittest.main()
