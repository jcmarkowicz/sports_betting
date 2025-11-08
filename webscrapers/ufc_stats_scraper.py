from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException

from selenium.webdriver.support import expected_conditions as EC
import time

import re
from datetime import datetime 
from dateutil import parser

import pandas as pd
import numpy as np
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

def scrape_upcoming(driver, event_rows):
    upcoming_dict = defaultdict(list)    
    event_links = [event_row.find_element(By.CSS_SELECTOR, "a").get_attribute("href") for event_row in event_rows]

    for event_link in event_links:
        driver.get(event_link)
        table_body = WebDriverWait(driver, 1).until(
                EC.presence_of_element_located((By.CLASS_NAME, "b-fight-details__table-body"))
            )
        
        current_event_date = driver.find_element(By.CSS_SELECTOR, ".b-list__box-list-item").text.replace('DATE:','')
        event_date = parser.parse(current_event_date).date()\
        
        event_name = driver.find_element(By.CSS_SELECTOR, ".b-content__title-highlight").text
        location = driver.find_element(By.CSS_SELECTOR, ".b-list__box-list-item:nth-child(2)").text.replace('LOCATION:','')
        
        fights = table_body.find_elements(By.CLASS_NAME, "b-fight-details__table-row")
        for fight in fights: #navigating the specific fights 

            columns = fight.find_elements(By.CLASS_NAME, "b-fight-details__table-col")
            fighter_names = columns[1].text.replace("\n", ", ")

            weight_class = columns[6].text
            red_fighter, blue_fighter = split_comma(fighter_names)
            fighter_links = columns[1].find_elements(By.TAG_NAME, 'a')

            heights = []
            reaches = []
            dobs = []

            for link in fighter_links:
                url = link.get_attribute("href")
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
                driver.back()
                try:
                    img = driver.find_element(By.CSS_SELECTOR, "img[src*='belt.png']")
                    title_fight = 1
                except: 
                    title_fight = 0

            upcoming_dict['title_fight'].append(title_fight)
            upcoming_dict['fighter_red'].append(red_fighter)
            upcoming_dict['fighter_blue'].append(blue_fighter)
            upcoming_dict['weight_class'].append(weight_class)
            upcoming_dict['event_date'].append(event_date)
            upcoming_dict['event_location'].append(location)

            upcoming_dict['reach_red'].append(reaches[0])
            upcoming_dict['reach_blue'].append(reaches[1])

            upcoming_dict['height_red'].append(heights[0])
            upcoming_dict['height_blue'].append(heights[1])

            upcoming_dict['dob_red'].append(dobs[0])
            upcoming_dict['dob_blue'].append(dobs[1])
    
    upcoming_df = pd.DataFrame(upcoming_dict)
    return upcoming_df


def scrape_ufc(end_date, get_upcoming, start_page=1, end_page=28):
    """end date: year-month-date ('2025-01-01'), set to None if all data needs scraping"""
    
    if end_date is not None:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_date = None

    df_rows = []
    driver = webdriver.Chrome()
    base_url = 'http://ufcstats.com/statistics/events/completed?page='

    # Loop through each page
    for page_num in range(start_page, end_page + 1):
        page_url = f"{base_url}{page_num}"  # Build the URL for the current page
        print(f"Navigating to page {page_num}: {page_url}")
        
        # Navigate to the page
        driver.get(page_url)
    
        # get upcoming card with no fight results, return that data 
        if get_upcoming == True:
            event_rows = WebDriverWait(driver, 1).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".b-statistics__table-row_type_first"))
            )
            upcoming_df = scrape_upcoming(driver, event_rows)
            return upcoming_df
        
        # else, continue getting all data 
        else:
            event_rows = WebDriverWait(driver, 1).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".b-statistics__table-row"))
            )
            event_rows = event_rows[2:]

        # outer page, iterate through events table 
        event_links = [event_row.find_element(By.CSS_SELECTOR, "a").get_attribute("href") for event_row in event_rows]

        # navigate to a specific link 
        for event_link in event_links:
            driver.get(event_link)
            table_body = WebDriverWait(driver, 1).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "b-fight-details__table-body"))
                )
            
            current_event_date = driver.find_element(By.CSS_SELECTOR, ".b-list__box-list-item").text.replace('DATE:','')
            event_date = parser.parse(current_event_date).date()
            # event_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            event_name = driver.find_element(By.CSS_SELECTOR, ".b-content__title-highlight").text
            location = driver.find_element(By.CSS_SELECTOR, ".b-list__box-list-item:nth-child(2)").text.replace('LOCATION:','')

            # iterate through individual fights 
            fights = table_body.find_elements(By.CLASS_NAME, "b-fight-details__table-row")
            for fight in fights:
                columns = fight.find_elements(By.CLASS_NAME, "b-fight-details__table-col") # navigating the specific fights 

                weight_class = columns[6].text
                method = columns[7].text.replace("\n", ", ")
                round = columns[8].text
                fight_time = columns[9].text

                fight_link = fight.get_attribute("onclick")
                if fight_link:
                    fight_url = fight_link.split("'")[1] 

                    # Navigate to the fight details page
                    driver.get(fight_url)

                    table_body_totals = WebDriverWait(driver, 1).until(
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

                    win_bonus = 0
                    fight_bonus = 0

                    perf_bonus_exists = bool(driver.find_elements("xpath", "//img[contains(@src, 'perf.png')]"))
                    fight_bonus_exists = bool(driver.find_elements("xpath", "//img[contains(@src, 'fight.png')]"))

                    if perf_bonus_exists: 
                        win_bonus = 1
                    if fight_bonus_exists: 
                        fight_bonus = 1

                    # Title Fight
                    try:
                        img = driver.find_element(By.CSS_SELECTOR, "img[src*='belt.png']")
                        title_fight = 1
                    except: 
                        title_fight = 0

                    # GET LINK TO FIGHTERS PAGE 
                    fighters = columns[0].find_elements(By.TAG_NAME, 'a')
                    heights = []
                    reaches = []
                    stances = []
                    records = []
                    dobs = []
                    
                    # NAVIGATE TO FIGHTERS PAGE 
                    for link in fighters:
                        url = link.get_attribute("href") 
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
                        driver.back()

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

                        if str(status).lower() == 'w':
                            if str(name).lower() == str(red_fighter).lower():
                                winner = red_fighter
                            else: 
                                winner = blue_fighter
                    
                    if winner == '':
                        winner = 'DRAW'

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
                    driver.back()

            if end_date:
                if event_date <= end_date:
                    return pd.DataFrame(df_rows)  
                
        time.sleep(1)
    return pd.DataFrame(df_rows) 


#Index(['title_fight', 'event_name', 'event_date', 'event_location',
    #    'fight_url', 'weight_class', 'method', 'round', 'fight_time',
    #    'performance_bonus_winner', 'fight_otn_bonus', 'winner', 'fighter_red',
    #    'fighter_blue', 'record_red', 'record_blue', 'dob_red', 'dob_blue',
    #    'height_red', 'height_blue', 'reach_red', 'reach_blue', 'stance_red',
    #    'stance_blue', 'kd_red', 'kd_blue', 'sig_str_red', 'sig_str_blue',
    #    'sig_str_percent_red', 'sig_str_percent_blue', 'td_red', 'td_blue',
    #    'td_pct_red', 'td_pct_blue', 'clinch_red', 'clinch_blue', 'ground_red',
    #    'ground_blue', 'sub_att_red', 'sub_att_blue', 'rev_red', 'rev_blue',
    #    'ctrl_red', 'ctrl_blue', 'head_red', 'head_blue', 'body_red',
    #    'body_blue', 'leg_red', 'leg_blue', 'distance_red', 'distance_blue',
    #    'total_strikes_red', 'total_strikes_blue'],
    #   dtype='object')