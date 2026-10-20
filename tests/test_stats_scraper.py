"""Scraper regression tests without network access or a running Chrome browser."""
import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch

from ufc_betting.DataPipeline.webscrapers import stats_scraper as scraper


def node(text='', attrs=None, **children):
    return dict(text=text, attrs=attrs or {}, children=children)


class Element:
    def __init__(self, driver, data):
        self.driver, self.data, self.generation = driver, data, driver.generation

    def check(self):
        if self.generation != self.driver.generation:
            raise AssertionError('Stale element reused after navigation')

    @property
    def text(self):
        self.check()
        return self.data['text']

    def get_attribute(self, name):
        self.check()
        return self.data['attrs'].get(name)

    def find_elements(self, by, selector):
        self.check()
        return [Element(self.driver, item) for item in self.data['children'].get(selector, [])]

    def find_element(self, by, selector):
        return self.find_elements(by, selector)[0]


class Browser:
    def __init__(self, pages):
        self.pages, self.generation, self.visited, self.quit_count = pages, 0, [], 0

    def get(self, url):
        self.generation += 1
        self.visited.append(url)
        self.root = Element(self, self.pages[url])

    def find_elements(self, by, selector):
        return self.root.find_elements(by, selector)

    def find_element(self, by, selector):
        return self.root.find_element(by, selector)

    def quit(self):
        self.quit_count += 1


def fixture(statuses=('W', 'L')):
    pages = {}
    links = []
    for event_id, date in enumerate(['January 18, 2025', 'January 18, 2025', 'January 11, 2025']):
        event_url = f'event/{event_id}'
        links.append(node(a=[node(attrs={'href': event_url})]))
        rows = []
        for position in range(1, 3):
            url = f'fight/{event_id}/{position}'
            fighter_links = [node(attrs={'href': f'fighter/{side}'}) for side in ('red', 'blue')]
            columns = [node() for _ in range(10)]
            columns[1] = node('Red Fighter\nBlue Fighter', a=fighter_links)
            columns[6:10] = [node('Lightweight'), node('KO/TKO'), node('2'), node('1:23')]
            rows.append(node(attrs={'onclick': f"window.location='{url}'"}, **{
                'b-fight-details__table-col': columns,
                "img[src*='belt.png']": [node()] if position == 1 else [],
            }))
            totals = [node(a=fighter_links)] + [node('1\n0') for _ in range(9)]
            sig = [node('1\n0') for _ in range(9)]
            pages[url] = node(**{
                'b-fight-details__table-body': [
                    node(**{'b-fight-details__table-row': [node(**{'b-fight-details__table-col': totals})]}),
                    node(),
                    node(**{'b-fight-details__table-row': [node(td=sig)]}),
                ],
                "//i[@class='b-fight-details__charts-name b-fight-details__charts-name_pos_left js-chart-name' and @data-color='red']": [node('Red Fighter')],
                "//i[@class='b-fight-details__charts-name b-fight-details__charts-name_pos_right js-chart-name' and @data-color='blue']": [node('Blue Fighter')],
                "//a[contains(@class, 'b-fight-details__person-link')]": [node('Red Fighter'), node('Blue Fighter')],
                "//i[contains(@class, 'b-fight-details__person-status')]": [node(s) for s in statuses],
                "img[src*='belt.png']": [node()] if position == 1 else [],
            })
        pages[event_url] = node(**{
            "img[src*='belt.png']": [node()],
            'b-fight-details__table-body': [node(**{'b-fight-details__table-row': rows})],
            '.b-list__box-list-item': [node('DATE: ' + date)],
            '.b-content__title-highlight': [node(f'UFC {300 + event_id}: Example')],
            '.b-list__box-list-item:nth-child(2)': [node('LOCATION: Las Vegas')],
        })
    for side in ('red', 'blue'):
        pages[f'fighter/{side}'] = node(**{
            "//li[i[normalize-space(text())='Height:']]": [node('HEIGHT: 6 ft')],
            "//li[i[normalize-space(text())='Reach:']]": [node('REACH: 72')],
            "//li[i[normalize-space(text())='STANCE:']]": [node('STANCE: Orthodox')],
            "//li[i[normalize-space(text())='DOB:']]": [node('DOB: Jan 1, 1990')],
            "//span[@class='b-content__title-record']": [node('RECORD: 10-1-0')],
        })
    listing = node(**{'.b-statistics__table-row': [node(th=[node()])] + links})
    pages['http://ufcstats.com/statistics/events/completed'] = listing
    pages['http://ufcstats.com/statistics/events/upcoming'] = listing
    return Browser(pages)


class StatsScraperTests(unittest.TestCase):
    def test_upcoming_event_is_skipped_by_content_not_position(self):
        for upcoming_event in (0, 1, 2):
            with self.subTest(upcoming_event=upcoming_event):
                driver = fixture()
                table = driver.pages[f'event/{upcoming_event}']['children']['b-fight-details__table-body'][0]
                for fight in table['children']['b-fight-details__table-row']:
                    columns = fight['children']['b-fight-details__table-col']
                    columns[7:10] = [node('--'), node('--'), node('--')]
                output = io.StringIO()
                with redirect_stdout(output):
                    frame = self.run_scrape(driver)
                self.assertEqual(len(frame), 4)
                self.assertNotIn(f'fight/{upcoming_event}/1', driver.visited)
                self.assertNotIn(f'fight/{upcoming_event}/2', driver.visited)
                expected_first = 301 if upcoming_event == 0 else 300
                self.assertIn(f'Page 1: first event scraped: UFC {expected_first}: Example', output.getvalue())
                self.assertNotIn(f'UFC {300 + upcoming_event}: Example', frame.event_name.tolist())

    def test_first_event_is_retained_on_each_page(self):
        base = 'http://ufcstats.com/statistics/events/completed'
        for headers in ([], [node(th=[node()])]):
            for end_page in (3, None):
                driver = fixture()
                for page, event in enumerate(range(3), start=1):
                    url = base if page == 1 else f'{base}?page={page}'
                    driver.pages[url] = node(**{'.b-statistics__table-row': headers + [
                        node(a=[node(attrs={'href': f'event/{event}'})])
                    ]})
                driver.pages[base + '?page=4'] = node(**{
                    '.b-statistics__table-row': [node(th=[node()])]
                })
                with patch.object(scraper.webdriver, 'Chrome', return_value=driver), \
                     patch.object(scraper.time, 'sleep'):
                    frame = scraper.scrape_ufc(end_page=end_page)
                self.assertEqual(len(frame), 6)
                for event in range(3):
                    self.assertEqual(driver.visited.count(f'event/{event}'), 1)

    def test_unavailable_first_listing_continues_to_page_two(self):
        driver = fixture()
        base = 'http://ufcstats.com/statistics/events/completed'
        driver.pages[base + '?page=2'] = node(**{'.b-statistics__table-row': [node(th=[node()])] + driver.pages[base]['children']['.b-statistics__table-row'][1:]})
        driver.pages[base] = node()
        driver.pages[base + '?page=3'] = node(**{'.b-statistics__table-row': [node(th=[node()])]})
        real_wait = scraper.WebDriverWait
        with patch.object(scraper.webdriver, 'Chrome', return_value=driver), \
             patch.object(scraper.time, 'sleep'), \
             patch.object(scraper, 'WebDriverWait', side_effect=lambda d, _: real_wait(d, .001, poll_frequency=.001)):
            frame = scraper.scrape_ufc()
        self.assertEqual(len(frame), 6)
        self.assertEqual(driver.visited.count(base), 3)
        self.assertIn(base + '?page=2', driver.visited)

    def test_repeated_listing_failures_return_incomplete_without_raising(self):
        driver = fixture()
        base = 'http://ufcstats.com/statistics/events/completed'
        for url in [base, base + '?page=2', base + '?page=3']:
            driver.pages[url] = node()
        real_wait = scraper.WebDriverWait
        with patch.object(scraper.webdriver, 'Chrome', return_value=driver), \
             patch.object(scraper.time, 'sleep'), \
             patch.object(scraper, 'WebDriverWait', side_effect=lambda d, _: real_wait(d, .001, poll_frequency=.001)):
            frame = scraper.scrape_ufc()
        self.assertTrue(frame.empty)
        self.assertFalse(frame.attrs['scrape_complete'])
        self.assertEqual(driver.quit_count, 1)

    def test_all_pages_without_pagination_links_and_bounded_mode(self):
        base = 'http://ufcstats.com/statistics/events/completed'
        for end_page, expected in [(None, list(range(27, 34))), (28, [27, 28])]:
            driver = fixture()
            for n in range(27, 33):
                driver.pages[f'event/{n}'] = driver.pages['event/0']
                driver.pages[f'{base}?page={n}'] = node(**{
                    '.b-statistics__table-row': [node(th=[node()]), node(a=[node(attrs={'href': f'event/{n}'})])],
                })
            driver.pages[f'{base}?page=33'] = node(**{'.b-statistics__table-row': [node(th=[node()])]})
            with patch.object(scraper.webdriver, 'Chrome', return_value=driver), patch.object(scraper.time, 'sleep'):
                frame = scraper.scrape_ufc(start_page=27, end_page=end_page)
            self.assertEqual([url for url in driver.visited if url.startswith(base)],
                             [f'{base}?page={n}' for n in expected])
            self.assertEqual(len(frame), 12 if end_page is None else 4)
            self.assertEqual(driver.quit_count, 1)

    def test_all_pages_stops_on_repeated_listing_without_duplicate_fights(self):
        driver = fixture()
        base = 'http://ufcstats.com/statistics/events/completed'
        driver.pages[base + '?page=2'] = node(**{'.b-statistics__table-row': [node(th=[node()])] + driver.pages[base]['children']['.b-statistics__table-row'][1:]})
        with patch.object(scraper.webdriver, 'Chrome', return_value=driver), patch.object(scraper.time, 'sleep'):
            frame = scraper.scrape_ufc()
        self.assertEqual(len(frame), 6)
        self.assertIn(base + '?page=2', driver.visited)
        self.assertNotIn(base + '?page=3', driver.visited)

    def test_missing_event_element_logs_name_and_continues(self):
        driver = fixture()
        listing = driver.pages['http://ufcstats.com/statistics/events/completed']
        listing['children']['.b-statistics__table-row'][1]['children']['a'][0]['text'] = 'UFC 300: Example'
        del driver.pages['event/0']['children']['.b-list__box-list-item:nth-child(2)']
        output = io.StringIO()
        with redirect_stdout(output):
            frame = self.run_scrape(driver)
        self.assertEqual(len(frame), 4)
        self.assertIn('Skipping event UFC 300: Example', output.getvalue())
        self.assertIn('fight/2/2', driver.visited)

    def test_missing_detail_element_logs_fight_and_continues_same_card(self):
        driver = fixture()
        del driver.pages['fight/0/1']['children']["//i[contains(@class, 'b-fight-details__person-status')]"]
        output = io.StringIO()
        with redirect_stdout(output):
            frame = self.run_scrape(driver)
        self.assertEqual(len(frame), 5)
        self.assertEqual(frame.bout_position.tolist(), [2, 1, 2, 1, 2])
        self.assertIn('UFC 300: Example: Red Fighter vs Blue Fighter', output.getvalue())

    def test_missing_upcoming_profile_skips_atomic_row_and_continues(self):
        driver = fixture()
        driver.pages['fighter/missing'] = node()
        table = driver.pages['event/0']['children']['b-fight-details__table-body'][0]
        first = table['children']['b-fight-details__table-row'][0]
        first['children']['b-fight-details__table-col'][1]['children']['a'][1]['attrs']['href'] = 'fighter/missing'
        output = io.StringIO()
        with redirect_stdout(output):
            frame = self.run_scrape(driver, upcoming=True)
        self.assertEqual(len(frame), 5)
        self.assertEqual(frame.bout_position.tolist(), [2, 1, 2, 1, 2])
        self.assertIn('Red Fighter vs Blue Fighter', output.getvalue())

    def test_missing_event_link_does_not_discard_other_events(self):
        driver = fixture()
        listing = driver.pages['http://ufcstats.com/statistics/events/completed']
        listing['children']['.b-statistics__table-row'][1] = node('UFC 300: Example')
        output = io.StringIO()
        with redirect_stdout(output):
            frame = self.run_scrape(driver)
        self.assertEqual(len(frame), 4)
        self.assertIn('Skipping UFC 300: Example', output.getvalue())

    def run_scrape(self, driver, cutoff=None, upcoming=False):
        with patch.object(scraper.webdriver, 'Chrome', return_value=driver), patch.object(scraper.time, 'sleep'):
            return scraper.scrape_ufc(cutoff, upcoming, end_page=1)

    def test_completed_navigation_preserves_all_positions(self):
        driver = fixture()
        frame = self.run_scrape(driver)
        self.assertEqual(len(frame), 6)
        self.assertEqual(frame.bout_position.tolist(), [1, 2, 1, 2, 1, 2])
        self.assertTrue(frame.winner.eq('Red Fighter').all())
        self.assertEqual(frame.fight_url.nunique(), 6)
        self.assertEqual(driver.quit_count, 1)
        for url in frame.fight_url:
            self.assertEqual(driver.visited.count(url), 1)

    def test_fight_page_is_not_reloaded_after_profile_navigation(self):
        driver = fixture()
        get = driver.get
        visited_fights = set()
        def get_once(url):
            if url.startswith('fight/'):
                if url in visited_fights:
                    raise AssertionError('Fight detail reload can replace stats with a browser check')
                visited_fights.add(url)
            get(url)
        driver.get = get_once
        frame = self.run_scrape(driver)
        self.assertEqual(len(frame), 6)
        self.assertTrue(frame.head_red.eq('1').all())
        self.assertTrue(frame.winner.eq('Red Fighter').all())

    def test_cutoff_includes_all_events_on_date_and_skips_older_fights(self):
        driver = fixture()
        frame = self.run_scrape(driver, '2025-01-18')
        self.assertEqual(len(frame), 4)
        self.assertEqual(frame.event_name.nunique(), 2)
        self.assertFalse(any(url.startswith('fight/2/') for url in driver.visited))
        self.assertEqual(driver.quit_count, 1)

    def test_cutoff_between_events_and_before_any_eligible_event(self):
        for cutoff, expected in [('2025-01-15', 4), ('2025-01-19', 0)]:
            driver = fixture()
            self.assertEqual(len(self.run_scrape(driver, cutoff)), expected)
            self.assertEqual(driver.quit_count, 1)

    def test_unrecognized_results_default_to_draw(self):
        driver = fixture(statuses=('', ''))
        frame = self.run_scrape(driver)
        self.assertTrue(frame.winner.eq('DRAW').all())
        self.assertEqual(driver.quit_count, 1)

    def test_navigation_failure_closes_browser(self):
        driver = fixture()
        del driver.pages['fighter/blue']
        with self.assertRaises(KeyError):
            self.run_scrape(driver)
        self.assertEqual(driver.quit_count, 1)

    def test_missing_fight_link_is_skipped_without_renumbering(self):
        driver = fixture()
        table = driver.pages['event/0']['children']['b-fight-details__table-body'][0]
        table['children']['b-fight-details__table-row'][0]['attrs'].pop('onclick')
        frame = self.run_scrape(driver)
        self.assertEqual(len(frame), 5)
        self.assertEqual(frame.bout_position.tolist(), [2, 1, 2, 1, 2])
        self.assertNotIn('fight/0/1', driver.visited)
        self.assertEqual(driver.quit_count, 1)

    def test_upcoming_navigation_and_single_owner_cleanup(self):
        driver = fixture()
        frame = self.run_scrape(driver, upcoming=True)
        self.assertEqual(len(frame), 6)
        self.assertEqual(frame.bout_position.tolist(), [1, 2, 1, 2, 1, 2])
        self.assertEqual(frame.title_fight.tolist(), [1, 1, 1, 1, 1, 1])
        self.assertEqual(frame.event_name.nunique(), 3)
        self.assertEqual(driver.quit_count, 1)

    def test_previous_winner_and_nc_behavior(self):
        for statuses, expected in [(['W', 'L'], 'Red Fighter'), (['L', 'W'], 'Blue Fighter'),
                                   (['D', 'D'], 'DRAW'), (['NC', 'NC'], 'NC'),
                                   (['NC', 'L'], 'NC'), (['W', 'W'], 'Blue Fighter')]:
            with self.subTest(statuses=statuses):
                frame = self.run_scrape(fixture(statuses=statuses))
                self.assertTrue(frame.winner.eq(expected).all())


if __name__ == '__main__':
    unittest.main()
