from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
from dateutil import parser
from collections import defaultdict

import random 
import time
import undetected_chromedriver as uc

def fighter_odds_search(red_fighter, blue_fighters, event_dates, driver):

    search_name = red_fighter
    search_box = driver.find_element(By.ID, "search-box1")
    search_box.clear()
    search_box.send_keys(search_name)
    search_box.send_keys(Keys.RETURN)

    try: 
        table = driver.find_element(By.CSS_SELECTOR, "table.content-list")
    except:
        print(f"Found ONE matches: {red_fighter}")
        odds_dic = scrape_odds(driver, blue_fighters, event_dates)
        print(f'Confirming return vals: {odds_dic}')
        driver.back()  # Go back to the search results page
        return odds_dic

    rows = table.find_elements(By.TAG_NAME, "tr")
    possible_matches = []

    for row in rows:
        fighter_link = row.find_element(By.CSS_SELECTOR, "td a")
        fighter_name = fighter_link.text.lower()

        # Check if all parts of the search name are in the fighter's name
        if all(term in clean_string(fighter_name) for term in clean_string(search_name)) or \
            all(term in clean_string(search_name) for term in clean_string(fighter_name)):
            possible_matches.append(fighter_link)

    if possible_matches:
        print(f"Found possible matches: {[link.text for link in possible_matches]}")
        odds_dic = defaultdict(list)
        for link in possible_matches:
            link.click()
            return_dic = scrape_odds(driver, blue_fighters, event_dates)
            
            for key, value in return_dic.items():
                if len(value)!=0:
                    odds_dic[key].extend(value)
                    
            driver.back()  # Go back to the search results page
        if len(odds_dic['red_fighter']) != 0:
            odds_dic['og_red_fighter'].extend([red_fighter] * len(odds_dic['red_fighter']))
        return odds_dic
            # time.sleep(2)  # Allow time for the page to reload
    else:
        print(f"No match found for '{search_name}'.")


def scrape_odds(driver, blue_fighters, dates):

    red_fighter_row = driver.find_elements(By.CSS_SELECTOR, "tr.main-row") #red fighter row 
    odds_dic = {'blue_fighter':[], 'open_blue':[], 'close1_blue':[], 'close2_blue':[],
            'red_fighter':[], 'open_red':[], 'close1_red':[], 'close2_red':[], 'event_date':[],
            'og_blue_name':[]}

    for i in range(0, len(red_fighter_row)):

        red_name = red_fighter_row[i].find_element(By.CSS_SELECTOR, "th.oppcell").text
        blue_row = red_fighter_row[i].find_element(By.XPATH, "following-sibling::tr[1]")
        blue_name = blue_row.find_element(By.CSS_SELECTOR, "th.oppcell a").text
        # blue_name = clean_string_simple(blue_name)
        match, og_blue_name = is_two_way_partial_match(blue_name, blue_fighters)
        try:
            event_date_elements = blue_row.find_elements(By.CSS_SELECTOR, "td.item-non-mobile")
            current_event_date = event_date_elements[0].text.strip()
            dt = parser.parse(current_event_date).date()
            dt = str(dt)
        except:
            dt = None
        
        if match == 'True': 
            #case when page values give n/a, see Allan Nascimento
            try:
                
                moneylines_blue = blue_row.find_elements(By.CSS_SELECTOR, "td.moneyline span")
                open_range_blue = moneylines_blue[0].text
                closing_range_blue = [moneylines_blue[1].text, moneylines_blue[2].text]

                moneylines_red = red_fighter_row[i].find_elements(By.CSS_SELECTOR, "td.moneyline span")
                open_range_red = moneylines_red[0].text
                closing_range_red = [moneylines_red[1].text, moneylines_red[2].text]

                odds_dic['red_fighter'].append(red_name)
                odds_dic['blue_fighter'].append(blue_name)

                odds_dic['open_blue'].append(open_range_blue)
                odds_dic['close1_blue'].append(closing_range_blue[0])
                odds_dic['close2_blue'].append(closing_range_blue[1])

                odds_dic['open_red'].append(open_range_red)
                odds_dic['close1_red'].append(closing_range_red[0])
                odds_dic['close2_red'].append(closing_range_red[1])

                odds_dic['event_date'].append(dt)
                odds_dic['og_blue_name'].append(og_blue_name)
            
            except:
                continue

    return odds_dic

def clean_string(s):
    return s.lower().replace(" ", " ").replace("-", " ").replace(".", "").replace("'","")

def is_two_way_partial_match(scraped_name, search_names):
    
    scraped_parts = clean_string(scraped_name)
    
    for name in search_names:
        search_parts = clean_string(name)
        if all(part in scraped_parts for part in search_parts) or \
           all(part in search_parts for part in scraped_parts):
            return 'True', name  # Match found
    return 'False', 'None'  # No match found

def get_fighter_odds(fighter_df):
    
    url = 'https://www.bestfightodds.com'
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        # add more realistic user agents
    ]

    # options = Options()
    # options.add_argument(f'user-agent={random.choice(user_agents)}')
    # driver = webdriver.Chrome(options=options)

    options = uc.ChromeOptions()
    # options.add_argument('--headless=new')
    options.add_argument(f'user-agent={random.choice(user_agents)}')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--incognito")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = uc.Chrome(options=options)
    driver.get(url)
    print(driver.current_url)  # see where you landed
    print(driver.title)   
    red_groups = fighter_df.groupby('fighter_red')

    odds_df = pd.DataFrame()
    for red_name, group in red_groups:
        print(red_name)
        blue_fighters = group['fighter_blue'].values
        # blue_fighters = [clean_string_simple(name) for name in blue_fighters]
        event_dates = group['event_date'].values

        time.sleep(random.uniform(1,2))

        odds_dic = fighter_odds_search(red_name, blue_fighters, event_dates, driver)
        row = pd.DataFrame(odds_dic)
        odds_df = pd.concat([odds_df, row], ignore_index=True)

    driver.quit()

    return odds_df