import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from ufc_betting.DataPipeline.webscrapers.scrape_diagnostics import save_listing_diagnostics


class ScrapeDiagnosticsTests(unittest.TestCase):
    def test_captures_page_and_metadata(self):
        driver=Mock(current_url='http://ufcstats.com/statistics/events/completed',title='Loading',page_source='<html>Loading</html>')
        driver.find_elements.return_value=[]
        with tempfile.TemporaryDirectory() as directory:
            save_listing_diagnostics(driver,page_number=2,directory=directory)
            path=Path(directory)
            self.assertEqual(json.loads((path/'completed_page_2.json').read_text())['matching_rows'],0)
            self.assertEqual((path/'completed_page_2.html').read_text(),driver.page_source)
            driver.save_screenshot.assert_called_once_with(str(path/'completed_page_2.png'))

    def test_screenshot_error_does_not_mask_scrape_failure(self):
        driver=Mock(current_url='url',title='title',page_source='<html/>')
        driver.find_elements.side_effect=RuntimeError('browser unavailable')
        driver.save_screenshot.side_effect=RuntimeError('screenshot unavailable')
        with tempfile.TemporaryDirectory() as directory:
            save_listing_diagnostics(driver,page_number=1,upcoming=True,directory=directory)
            self.assertTrue((Path(directory)/'upcoming_page_1.html').exists())
