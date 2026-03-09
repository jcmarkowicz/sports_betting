import numpy as np 
import pandas as pd 

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from webscrapers.ufc_odds_scraper import get_fighter_odds
from webscrapers.ufc_stats_scraper import scrape_ufc, scrape_upcoming

class UFC_Webscraper:

    def __init__(self):
        pass

    def scrape_until(self, date, upcoming=False):
        """If date==None, scrape all events"""
        updated_df = scrape_ufc(date, get_upcoming=upcoming) # if upcoming is True, scrape the upcoming fight card only 
        # updated_df.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        # updated_df.to_csv(file_path, index=False)
        return updated_df

    def get_fighter_odds(self, fighter_df):
        fighter_odds = get_fighter_odds(fighter_df)
        fighter_odds.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        return fighter_odds

    def merge_stats_odds(self, odds_df, stats_df, file_path):

        new_df = stats_df.merge(odds_df[['open_blue', 'close1_blue', 'close2_blue', 'red_fighter','blue_fighter', 'open_red',
       'close1_red', 'close2_red', 'event_date']], on = ['red_fighter','blue_fighter'], how='inner')

        new_df.sort_values(by='event_date', ascending=True)
        new_df.to_csv(file_path)

    def concat_old_new_stats(self, prev_scrape_df, new_scrape_df, file_path):
        
        new_df = pd.concat([prev_scrape_df, new_scrape_df], ignore_index=True)
        new_df.to_csv(file_path)

    def concat_old_new_odds(self, prev_scrape_odds, new_scrape_odds, file_path):

        df_combined = pd.concat([prev_scrape_odds, new_scrape_odds], ignore_index=True).drop_duplicates()
        df_combined.sort_values(by='event_date', ascending=True)
        df_combined.to_csv(file_path)

    def scrape_upcoming_card(self):
        upcoming_df = self.scrape_until(None, upcoming=True)
        return upcoming_df