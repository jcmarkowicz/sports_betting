from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
from contextlib import contextmanager

import time
from datetime import datetime 
from dateutil import parser

import pandas as pd
from collections import defaultdict


FIGHTER_ATTRS = [
    "fighter", "record", "dob", "height", "reach", "stance",
    "kd", "sig_str", "sig_str_percent", "td", "td_pct",
    "clinch", "ground", "sub_att", "rev", "ctrl",
    "head", "body", "leg", "distance", "total_strikes"
]
EVENT_ATTRS = [
    "event_name",
    "event_date",
    "event_location",
    "fight_url",
    "weight_class",
    "method",
    "round",
    "time",
    "performance_bonus_winner",
    "fight_otn_bonus",
    "winner"
]

UPCOMING_ATTRS = [
    "event_name",
    "event_date",
    "event_location",
    "fight_url",
    'title_fight'
]

def clean_string(s):
    return s.lower().replace(" ", " ").replace("-", " ").replace(".", "").replace("'","")

def split_comma(text):
    text_red, text_blue = text.split(",", 1)
    before_comma = text_red.strip()
    after_comma = text_blue.strip()
    return before_comma, after_comma


@contextmanager
def _skip_missing(label):
    """Skip only unavailable page elements; retain the rest of the scrape."""
    try:
        yield
    except (NoSuchElementException, TimeoutException, StaleElementReferenceException, IndexError) as exc:
        import traceback
        location = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"Skipping {label}: missing or unavailable element ({type(exc).__name__}) "
              f"at {location.name}:{location.lineno}: {location.line}")


def _event_links(event_rows):
    links = []
    for position, event_row in enumerate(event_rows, start=1):
        if event_row.find_elements(By.TAG_NAME, 'th'):
            continue
        label = f'event row {position}'
        with _skip_missing(label):
            label = event_row.text.replace('\n', ' ').strip() or label
            with _skip_missing(label):
                link = event_row.find_element(By.CSS_SELECTOR, 'a')
                url = link.get_attribute('href')
                if url:
                    links.append((link.text.strip() or label, url))
                else:
                    print(f'Skipping {label}: missing event link.')
    return links


def _completed_pages(driver, start_page, end_page):
    """Visit pages sequentially; the caller stops at an empty/repeated list."""
    base_url = 'http://ufcstats.com/statistics/events/completed'
    page_num = start_page
    while end_page is None or page_num <= end_page:
        page_url = f'{base_url}?page={page_num}' if page_num > 1 else base_url
        print(f'Navigating to page {page_num}: {page_url}')
        driver.get(page_url)
        yield page_num
        page_num += 1


def scrape_upcoming(driver, event_rows):
    upcoming_rows = []    
    event_links = _event_links(event_rows)

    for event_label, event_link in event_links:
        with _skip_missing(f'event {event_label} ({event_link})'):
            driver.get(event_link)
            table_body = WebDriverWait(driver, 0.25).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "b-fight-details__table-body"))
                )
        
            current_event_date = driver.find_element(By.CSS_SELECTOR, ".b-list__box-list-item").text.replace('DATE:','')
            event_date = parser.parse(current_event_date).date()
        
            event_name = driver.find_element(By.CSS_SELECTOR, ".b-content__title-highlight").text
            location = driver.find_element(By.CSS_SELECTOR, ".b-list__box-list-item:nth-child(2)").text.replace('LOCATION:','')
        
            try:
                driver.find_element(By.CSS_SELECTOR, "img[src*='belt.png']")
                title_fight = 1
            except:
                title_fight = 0

            fights = table_body.find_elements(By.CLASS_NAME, "b-fight-details__table-row")
            fight_rows = []
            for bout_position, fight in enumerate(fights, start=1):
                with _skip_missing(f'{event_name}, bout {bout_position}'):
                    columns = fight.find_elements(By.CLASS_NAME, "b-fight-details__table-col")
                    fighter_names = columns[1].text.replace("\n", ", ")
                    red_fighter, blue_fighter = split_comma(fighter_names)
                    fighter_urls = [link.get_attribute('href') for link in columns[1].find_elements(By.TAG_NAME, 'a')]
                    fight_rows.append(dict(bout_position=bout_position,
                        red_fighter=red_fighter, blue_fighter=blue_fighter,
                        weight_class=columns[6].text, fighter_urls=fighter_urls,
                        title_fight=title_fight))

            for fight_data in fight_rows:
                with _skip_missing(f"{event_name}: {fight_data['red_fighter']} vs {fight_data['blue_fighter']}"):
                    red_fighter, blue_fighter = fight_data['red_fighter'], fight_data['blue_fighter']
                    weight_class = fight_data['weight_class']
                    title_fight = fight_data['title_fight']

                    heights = []
                    reaches = []
                    dobs = []

                    for url in fight_data['fighter_urls']:
                        driver.get(url)
                        height_li = driver.find_element(By.XPATH, "//li[i[normalize-space(text())='Height:']]")
                        reach_li = driver.find_element(By.XPATH, "//li[i[normalize-space(text())='Reach:']]")
                        dob_li = driver.find_element(By.XPATH, "//li[i[normalize-space(text())='DOB:']]")
                
                        height = height_li.text.replace("HEIGHT:", "").strip()  
                        reach = reach_li.text.replace("REACH:", "").strip()    
                        dob = dob_li.text.replace("DOB:", "").strip()  

                        heights.append(height)
                        reaches.append(reach)
                        dobs.append(dob)

                    row = {}
                    row['title_fight'] = title_fight
                    row['fighter_red'] = red_fighter
                    row['fighter_blue'] = blue_fighter
                    row['weight_class'] = weight_class
                    row['event_date'] = event_date
                    row['event_location'] = location
                    row['event_name'] = event_name
                    row['bout_position'] = fight_data['bout_position']

                    row['reach_red'] = reaches[0]
                    row['reach_blue'] = reaches[1]

                    row['height_red'] = heights[0]
                    row['height_blue'] = heights[1]

                    row['dob_red'] = dobs[0]
                    row['dob_blue'] = dobs[1]

                    upcoming_rows.append(row)

    upcoming_df = pd.DataFrame(upcoming_rows)
    return upcoming_df


def scrape_ufc(end_date=None, get_upcoming=False, start_page=1, end_page=None):
    """Scrape events on/after the inclusive YYYY-MM-DD cutoff; None has no cutoff.

    Browser cleanup is guaranteed on success, cutoff returns, and failures.
    end_page=None (default) follows all numbered event-list pages. Pass an
    integer end_page for a bounded scrape. Missing elements log the affected
    event/fight and skip it, preserving original bout positions.
    """
    
    if end_date is not None:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_date = None

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    try:
        return _scrape_ufc(driver, end_date, get_upcoming, start_page, end_page)
    finally:
        driver.quit()


def _scrape_ufc(driver, end_date, get_upcoming, start_page, end_page):
    
    df_rows = []
    seen_event_links = set()
    consecutive_failed_pages = 0
    # Preserve sequential numbered navigation without requiring pagination links.
    for page_num in _completed_pages(driver, start_page, start_page if get_upcoming else end_page):

        event_rows = None
        listing_url = ('http://ufcstats.com/statistics/events/upcoming' if get_upcoming else
                       'http://ufcstats.com/statistics/events/completed' +
                       (f'?page={page_num}' if page_num > 1 else ''))
        if get_upcoming:
            driver.get(listing_url)
        for attempt in range(1, 4):
            try:
                event_rows = WebDriverWait(driver, 0.25).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".b-statistics__table-row"))
                )
                break
            except (NoSuchElementException, TimeoutException, StaleElementReferenceException) as exc:
                print(f'Event listing page {page_num}: {type(exc).__name__}, attempt {attempt}/3.')
                if attempt < 3:
                    driver.get(listing_url)
        if event_rows is None:
            print(f'Skipping unavailable event listing page {page_num}; this is NOT the end of the event list.')
            consecutive_failed_pages += 1
            if consecutive_failed_pages >= 3 or get_upcoming:
                print('Scrape incomplete: event listings could not be loaded. Existing saved history must be retained.')
                result = pd.DataFrame(df_rows)
                result.attrs['scrape_complete'] = False
                return result
            continue
        consecutive_failed_pages = 0
        if get_upcoming:
            return scrape_upcoming(driver, event_rows)

        # outer page, iterate through events table 
        event_links = _event_links(event_rows)
        if end_page is None and not get_upcoming:
            if not event_links and not any(row.find_elements(By.CSS_SELECTOR, 'a') for row in event_rows):
                print(f'Stopping at page {page_num}: no event rows.')
                break
            if event_links and all(url in seen_event_links for _, url in event_links):
                print(f'Stopping at page {page_num}: event list repeats an earlier page.')
                break
            event_links = [(label, url) for label, url in event_links if url not in seen_event_links]
            seen_event_links.update(url for _, url in event_links)

        # navigate to a specific link 
        first_event_printed = False
        for event_label, event_link in event_links:
            with _skip_missing(f'event {event_label} ({event_link})'):
                driver.get(event_link)
                table_body = WebDriverWait(driver, 0.25).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "b-fight-details__table-body"))
                    )
                current_event_date = driver.find_element(By.CSS_SELECTOR, ".b-list__box-list-item").text.replace('DATE:','')
                event_date = parser.parse(current_event_date).date()
                if end_date is not None and event_date < end_date:
                    return pd.DataFrame(df_rows)
                # event_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                event_name = driver.find_element(By.CSS_SELECTOR, ".b-content__title-highlight").text
                location = driver.find_element(By.CSS_SELECTOR, ".b-list__box-list-item:nth-child(2)").text.replace('LOCATION:','')

                # iterate through individual fights 
                fights = table_body.find_elements(By.CLASS_NAME, "b-fight-details__table-row")
                # Snapshot every card row before navigating away. WebElements from
                # this document must not be reused after loading a fight page.
                fight_rows = []
                for bout_position, fight in enumerate(fights, start=1):
                    with _skip_missing(f'{event_name}, bout {bout_position}'):
                        columns = fight.find_elements(By.CLASS_NAME, "b-fight-details__table-col") # navigating the specific fights 
                        # Upcoming bouts can have links but no completed round/time.
                        round_text = columns[8].text.strip()
                        time_text = columns[9].text.strip()
                        if not round_text.isdigit() or ':' not in time_text:
                            continue
                        fight_link = fight.get_attribute("onclick")
                        if not fight_link:
                            print(f"Skipping {event_name}, {columns[1].text!r}: missing fight link.")
                            continue
                        fight_rows.append(dict(bout_position=bout_position,
                            fight_name=columns[1].text.replace("\n", " vs "),
                            weight_class=columns[6].text, method=columns[7].text.replace("\n", ", "),
                            round=round_text, fight_time=time_text,
                            fight_url=fight_link.split("'")[1]))

                if not fight_rows:
                    print(f'Skipping event {event_name}: no completed bouts with stats links available.')
                    continue
                if not first_event_printed:
                    print(f'Page {page_num}: first event scraped: {event_name}')
                    first_event_printed = True

                for fight_data in fight_rows:
                    with _skip_missing(f"{event_name}: {fight_data['fight_name']} ({fight_data['fight_url']})"):
                        bout_position = fight_data['bout_position']
                        weight_class = fight_data['weight_class']
                        method = fight_data['method']
                        round = fight_data['round']
                        fight_time = fight_data['fight_time']
                        fight_link = fight_data['fight_url']
                        if fight_link:
                            fight_url = fight_link

                            # Navigate to the fight details page
                            driver.get(fight_url)

                            table_body_totals = WebDriverWait(driver, 0.25).until(
                                EC.presence_of_element_located((By.CLASS_NAME, "b-fight-details__table-body"))
                            )            
                            rows = table_body_totals.find_elements(By.CLASS_NAME, "b-fight-details__table-row")
                            columns = rows[0].find_elements(By.CLASS_NAME, "b-fight-details__table-col")

                            # TOTALS TABLE, replace newline(\n) with comma, split by comma, red is 0, blue is 1 
                            # fighter_names = columns[0].text.replace("\n", ", ") 
                            kd = columns[1].text.replace("\n", ", ")             
                            sig_str = columns[2].text.replace("\n", ", ")        
                            sig_str_percent = columns[3].text.replace("\n", ", ")  
                            total_str = columns[4].text.replace("\n", ", ")     
                            td = columns[5].text.replace("\n", ", ")            
                            td_percent = columns[6].text.replace("\n", ", ")     
                            sub_att = columns[7].text.replace("\n", ", ")        
                            rev = columns[8].text.replace("\n", ", ")            
                            ctrl = columns[9].text.replace("\n", ", ") 

                            kd_red, kd_blue = split_comma(kd)
                            sig_str_red, sig_str_blue = split_comma(sig_str)
                            str_red, str_blue = split_comma(sig_str_percent)
                            td_red, td_blue = split_comma(td)
                            td_pct_red, td_pct_blue = split_comma(td_percent)
                            strikes_red, strikes_blue = split_comma(total_str)
                            sub_att_red, sub_att_blue = split_comma(sub_att)
                            rev_red, rev_blue = split_comma(rev)
                            ctrl_red, ctrl_blue = split_comma(ctrl)

                            perf_bonus_exists = bool(driver.find_elements("xpath", "//img[contains(@src, 'perf.png')]"))
                            fight_bonus_exists = bool(driver.find_elements("xpath", "//img[contains(@src, 'fight.png')]"))

                            win_bonus = 1 if perf_bonus_exists else 0 
                            fight_bonus = 1 if fight_bonus_exists else 0 

                            # Title Fight
                            title_fight = int(bool(driver.find_elements(By.CSS_SELECTOR, "img[src*='belt.png']")))

                            # GET LINK TO FIGHTERS PAGE 
                            fighters = columns[0].find_elements(By.TAG_NAME, 'a')
                            fighter_urls = [link.get_attribute('href') for link in fighters]
                            heights = []
                            reaches = []
                            stances = []
                            records = []
                            dobs = []
                    
                            # Capture fight data before leaving this document for fighter profiles.
                            # SIG STRIKES TABLE
                            tables = driver.find_elements(By.CLASS_NAME, "b-fight-details__table-body")

                            sig_strikes = tables[2]
                            rows = sig_strikes.find_elements(By.CLASS_NAME, "b-fight-details__table-row")
                            columns = rows[0].find_elements(By.TAG_NAME, "td")

                            head = columns[3].text.replace("\n", ", ")
                            body = columns[4].text.replace("\n", ", ")
                            leg = columns[5].text.replace("\n", ", ")
                            distance = columns[6].text.replace("\n", ", ")
                            clinch = columns[7].text.replace("\n", ", ")
                            ground = columns[8].text.replace("\n", ", ")

                            head_red, head_blue = split_comma(head)
                            body_red, body_blue = split_comma(body)
                            leg_red, leg_blue = split_comma(leg)
                            distance_red, distance_blue = split_comma(distance)
                            clinch_red, clinch_blue = split_comma(clinch)
                            ground_red, ground_blue = split_comma(ground)

                            red_fighter = driver.find_element(By.XPATH, "//i[@class='b-fight-details__charts-name b-fight-details__charts-name_pos_left js-chart-name' and @data-color='red']").text.strip()
                            blue_fighter = driver.find_element(By.XPATH, "//i[@class='b-fight-details__charts-name b-fight-details__charts-name_pos_right js-chart-name' and @data-color='blue']").text.strip()
                    
                            # names and statues under same class, so name will correspond to correct status 
                            names = driver.find_elements("xpath", "//a[contains(@class, 'b-fight-details__person-link')]")
                            statuses = driver.find_elements("xpath", "//i[contains(@class, 'b-fight-details__person-status')]")
                            fighter_element = driver.find_element("xpath", "//i[contains(@class, 'b-fight-details__person-status')]")
                            winner = ''
                            if "nc" in fighter_element.text.lower():
                                winner = 'NC'
                            for name_el, status_el in zip(names, statuses):
                                name = name_el.text.strip()
                                status = status_el.text.strip()
                                if status.lower() == 'w':
                                    if name.lower() == red_fighter.lower():
                                        winner = red_fighter
                                    else:
                                        winner = blue_fighter
                            if winner == '':
                                winner = 'DRAW'

                            # NAVIGATE TO FIGHTERS PAGE 
                            for url in fighter_urls:
                                driver.get(url)

                                height_li = driver.find_element(By.XPATH, "//li[i[normalize-space(text())='Height:']]")
                                reach_li = driver.find_element(By.XPATH, "//li[i[normalize-space(text())='Reach:']]")
                                stance_li = driver.find_element(By.XPATH, "//li[i[normalize-space(text())='STANCE:']]")
                                dob_li = driver.find_element(By.XPATH, "//li[i[normalize-space(text())='DOB:']]")
                                record_element = driver.find_element(By.XPATH, "//span[@class='b-content__title-record']")

                                height = height_li.text.replace("HEIGHT:", "").strip()  
                                reach = reach_li.text.replace("REACH:", "").strip()    
                                stance = stance_li.text.replace("STANCE:", "").strip() 
                                dob = dob_li.text.replace("DOB:", "").strip()  
                                record = record_element.text.replace("RECORD:", "").strip()  

                                heights.append(height)
                                reaches.append(reach)
                                stances.append(stance)
                                records.append(record)
                                dobs.append(dob)

                            # INCLUDE BASIC ATTRIBUTES FOUND ABOVE 
                            row = {
                                'title_fight':title_fight, 
                                'event_name': event_name,
                                'event_date': event_date,
                                'event_location': location,
                                'fight_url': fight_url,
                                'weight_class': weight_class,
                                'method': method,
                                'round': round,
                                'fight_time':fight_time, 
                                'performance_bonus_winner': win_bonus,
                                'fight_otn_bonus': fight_bonus,
                                'bout_position':bout_position,
                                'winner': winner
                            }

                            # fighter-level attributes
                            for attr, values in {
                                'fighter': (red_fighter, blue_fighter),
                                'record': records,
                                'dob': dobs,
                                'height': heights,
                                'reach': reaches,
                                'stance': stances,
                                'kd': (kd_red, kd_blue),
                                'sig_str': (sig_str_red, sig_str_blue),
                                'sig_str_percent': (str_red, str_blue),
                                'td': (td_red, td_blue),
                                'td_pct': (td_pct_red, td_pct_blue),
                                'clinch': (clinch_red, clinch_blue),
                                'ground': (ground_red, ground_blue),
                                'sub_att': (sub_att_red, sub_att_blue),
                                'rev': (rev_red, rev_blue),
                                'ctrl': (ctrl_red, ctrl_blue),
                                'head': (head_red, head_blue),
                                'body': (body_red, body_blue),
                                'leg': (leg_red, leg_blue),
                                'distance': (distance_red, distance_blue),
                                'total_strikes': (strikes_red, strikes_blue),
                            }.items():
                                row[f"{attr}_red"], row[f"{attr}_blue"] = values
                    
                            df_rows.append(row)
                
        time.sleep(0.25)
    
    return pd.DataFrame(df_rows) 


