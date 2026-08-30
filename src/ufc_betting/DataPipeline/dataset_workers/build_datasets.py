import numpy as np 
import pandas as pd 

import os
import sys 
from datetime import datetime 

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from DataPipeline.FeatureEngineering.features_pipeline import FeatureEngineering
from DataPipeline.webscrapers.scraping_pipeline import UFC_Webscraper

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]


get_all_stats = False
get_all_odds = False

get_next_fight_stats_odds = True

get_missing_stats = False
generate_model_df = True 



# find the most recent stats and odds history with most recent timestamp 
# recent date from last merger with missing stats
# from Data/HISTORY folder 
date_today = datetime.now().strftime("%Y-%m-%d") 
recent_date = r'2026-02-26'


#Folders for ALL scraped data
stats_history_file_string = BASE_DIR / "Data" / "history" / f"stats_history_{recent_date}.csv"
odds_history_file_string = BASE_DIR / "Data" / "history" / f"odds_history_{recent_date}.csv"

stats_history = pd.read_csv(f'{stats_history_file_string}_{recent_date}.csv')
stats_history = stats_history.drop(columns=[col for col in stats_history.columns if "Unnamed" in col])

odds_history = pd.read_csv(f'{odds_history_file_string}_{recent_date}.csv')
odds_history = odds_history.drop(columns=[col for col in odds_history.columns if "Unnamed" in col])

num_duplicate = stats_history.duplicated(
    subset=['fighter_red', 'fighter_blue', 'event_date']
).sum()
print(f'Number of duplicate rows={num_duplicate}')

stats_history = stats_history[~stats_history.duplicated(
    subset=['fighter_red', 'fighter_blue', 'event_date'],
    keep='first'
)]


# Path for scraped data for future events 
upcoming_scraped_stats_string = BASE_DIR / "Data/upcoming_events/scraped_data/upcoming_stats"
upcoming_scraped_odds_string = BASE_DIR / "Data/upcoming_events/scraped_data/upcoming_odds"


# dataframes for stats and odds with MULTIPLE upcoming event features  
# this date is set to the next event date available in UFC website, 
# only necessary to set when not scraping upcoming fights and want to generate model df only 
next_fight_date = r'2026-03-14'
next_event_folder = r'C:\Users\jcmar\my_files\SportsBetting\data\upcoming_events'

if __name__ == "__main__":
    scraper = UFC_Webscraper()
    features = FeatureEngineering()

    # scrape all stats
    if get_all_stats is True: 
        df_entire_stats_history = scraper.scrape_until(date=None)

        # timestamp date of scrape
        df_entire_stats_history.to_csv(f'{stats_history_file_string}_{date_today}.csv') 
        stats_history = df_entire_stats_history
    
    # scrape all odds 
    if get_all_odds is True: 
        df_entire_odds_history = scraper.get_fighter_odds(stats_history)
        df_entire_odds_history.to_csv(f'{odds_history_file_string}_{date_today}.csv')
        odds_history = df_entire_odds_history

    # Scrape upcoming fights
    if get_next_fight_stats_odds is True: 

        # scrape ALL upcoming events
        next_fight_stats = scraper.scrape_upcoming_card() # fix this name 
        next_fight_date = next_fight_stats['event_date'][0]
        
        # save upcoming scraped stats 
        next_stats_path = f'{upcoming_scraped_stats_string}_{next_fight_date}.csv'
        next_fight_stats.to_csv(next_stats_path, index=False)
        next_fight_stats = pd.read_csv(next_stats_path) # read back in to ensure same formatting for odds function

        # save upcoming scraped odds  
        next_odds_path = f'{upcoming_scraped_odds_string}_{next_fight_date}.csv' # upcoming odds for the next fight card
        next_odds_df = scraper.get_fighter_odds(next_fight_stats)
        next_odds_df.to_csv(next_odds_path, index=False)

    # Scrape odds/stats not in stats/odds history
    if get_missing_stats:

        # scrape stats from current date to prev_fight_date (last date in stats_history) 
        prev_fight_date = stats_history.sort_values(by='event_date', ascending=False, inplace=False).reset_index(drop=True)['event_date'][0]# get index 0, largest value in non ascending order  
        print(f"Previous fight date in stats history: {prev_fight_date}")

        missing_stats = scraper.scrape_until(prev_fight_date)
        missing_odds = scraper.get_fighter_odds(missing_stats)
    
        # merge with stats history, timestamp date of scrape/merge
        stats_history = pd.concat([stats_history, missing_stats], axis=0)
        stats_history['event_date'] = pd.to_datetime(stats_history['event_date'], errors='coerce')
        
        stats_history = stats_history.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        stats_history.to_csv(f'{stats_history_file_string}_{date_today}.csv')

        odds_history = pd.concat([odds_history, missing_odds], axis=0)
        odds_history['event_date'] = pd.to_datetime(odds_history['event_date'], errors='coerce')

        odds_history = odds_history.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        odds_history.to_csv(f'{odds_history_file_string}_{date_today}.csv')

    # Build features for history and upcoming 
    if generate_model_df is True: 

        next_stats_df = pd.read_csv(f'{upcoming_scraped_stats_string}_{next_fight_date}.csv')
        next_odds_df = pd.read_csv(f'{upcoming_scraped_odds_string}_{next_fight_date}.csv')

        odds_stats_df, upcoming_df = features.build_all_stats(stats_history, next_stats_df, odds_history, next_odds_df)
        odds_stats_df.to_csv(fr'C:\Users\jcmar\my_files\SportsBetting\data\training_data\entire_odds_stats_{date_today}.csv', index=False)
        
        upcoming_groups = upcoming_df.groupby('date')
        for date, group in upcoming_groups:
            date_str = date.strftime("%Y-%m-%d")   # or "%Y%m%d"
            filename = fr"{next_event_folder}\event_dfs\upcoming_odds_stats_{date_str}.csv"
            group.to_csv(filename, index=False)
