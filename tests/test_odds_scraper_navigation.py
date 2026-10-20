"""Navigation regressions without live requests or a Chrome process."""
import unittest
from unittest.mock import Mock, patch

from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait

from ufc_betting.DataPipeline.webscrapers import odds_scraper as scraper


class OddsNavigationTests(unittest.TestCase):
    def test_stale_table_is_reacquired_before_reading_matches(self):
        driver = Mock()
        driver.execute_script.return_value = 'complete'
        stale_table, table, row, link = Mock(), Mock(), Mock(), Mock()
        stale_table.find_elements.side_effect = StaleElementReferenceException()
        driver.find_element.side_effect = [stale_table, table]
        table.find_elements.return_value = [row]
        row.find_element.return_value = link
        link.text = 'Some Fighter'
        link.get_attribute.return_value = 'https://example.test/fighter'
        result = WebDriverWait(driver, 1, poll_frequency=.001).until(scraper._search_result_snapshot)
        self.assertEqual(result, ('search', [('some fighter', 'https://example.test/fighter')]))
        self.assertEqual(driver.find_element.call_count, 2)

    def test_missing_table_waits_for_document_then_allows_empty_fighter_page(self):
        driver = Mock()
        driver.find_element.side_effect = NoSuchElementException()
        driver.execute_script.return_value = 'loading'
        self.assertIs(scraper._search_result_snapshot(driver), False)
        driver.execute_script.return_value = 'complete'
        self.assertEqual(scraper._search_result_snapshot(driver), ('fighter', []))

    def test_matched_page_without_odds_rows_preserves_empty_result(self):
        driver, box = Mock(), Mock()
        driver.find_element.return_value = box
        box.is_enabled.side_effect = StaleElementReferenceException()
        driver.find_elements.return_value = []
        with patch.object(scraper, '_search_result_snapshot', return_value=(
                'search', [('some fighter', 'https://example.test/empty')])):
            result = scraper.fighter_odds_search('Some Fighter', ['Opponent'], ['2026-01-01'], driver)
        self.assertEqual(result['red_fighter'], [])
        driver.get.assert_called_once_with('https://example.test/empty')
        driver.back.assert_called_once()

    def test_multiple_matches_use_urls_and_preserve_matching_and_aggregation(self):
        driver, box = Mock(), Mock()
        driver.find_element.return_value = box
        box.is_enabled.side_effect = StaleElementReferenceException()
        driver.find_elements.return_value = [Mock()]
        matches = [('some fighter', 'https://example.test/one'),
                   ('unrelated', 'https://example.test/no'),
                   ('some fighter jr', 'https://example.test/two')]
        with patch.object(scraper, '_search_result_snapshot', return_value=('search', matches)), \
             patch.object(scraper, 'scrape_odds', side_effect=[
                 {'red_fighter': ['Some Fighter'], 'open_red': [120]},
                 {'red_fighter': ['Some Fighter Jr'], 'open_red': [-110]},
             ]) as extract:
            result = scraper.fighter_odds_search('Some Fighter', ['Opponent'], ['2026-01-01'], driver)
        self.assertEqual([call.args[0] for call in driver.get.call_args_list],
                         ['https://example.test/one', 'https://example.test/two'])
        self.assertEqual(result['open_red'], [120, -110])
        self.assertEqual(result['og_red_fighter'], ['Some Fighter'] * 2)
        self.assertEqual(extract.call_count, 2)
        self.assertEqual(driver.back.call_count, 2)

    def test_direct_fighter_result_keeps_existing_return_behavior(self):
        driver, box = Mock(), Mock()
        driver.find_element.return_value = box
        box.is_enabled.side_effect = StaleElementReferenceException()
        expected = {'red_fighter': ['Some Fighter'], 'open_red': [120]}
        with patch.object(scraper, '_search_result_snapshot', return_value=('fighter', [])), \
             patch.object(scraper, 'scrape_odds', return_value=expected):
            result = scraper.fighter_odds_search('Some Fighter', ['Opponent'], ['2026-01-01'], driver)
        self.assertIs(result, expected)
        driver.back.assert_called_once()
        driver.get.assert_not_called()


if __name__ == '__main__':
    unittest.main()
