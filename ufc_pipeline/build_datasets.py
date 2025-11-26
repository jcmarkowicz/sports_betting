import numpy as np 
import pandas as pd 

import os
import sys
from datetime import datetime 

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from features_pipeline import FeatureEngineering
from scraping_pipeline import UFC_Webscraper

get_all_stats = False
get_all_odds = False
get_next_fight_stats_odds = False

get_missing_stats = False
generate_model_df = True 

stats_history_file_string = r'C:\Users\jcmar\my_files\SportsBetting\data\stats_history'
odds_history_file_string = r'C:\Users\jcmar\my_files\SportsBetting\data\odds_history'

upcoming_stats_string = r'C:\Users\jcmar\my_files\SportsBetting\data\upcoming_stats'
upcoming_odds_string = r'C:\Users\jcmar\my_files\SportsBetting\data\upcoming_odds'

# find the most recent stats and odds history by most recent date 
date_today = datetime.now().strftime("%Y-%m-%d") #use this to mark odds 
recent_date = r'2025-11-05'
stats_history = pd.read_csv(f'{stats_history_file_string}_{recent_date}.csv') # dataframes BEFORE any feature engineering 

recent_date = r'2025-11-06'
odds_history = pd.read_csv(f'{odds_history_file_string}_{recent_date}.csv')

next_fight_date = r'2025-11-08'

if __name__ == "__main__":
    scraper = UFC_Webscraper()
    features = FeatureEngineering()

    # scrape all stats
    if get_all_stats is True: 
        df_entire_stats_history = scraper.scrape_until(date=None)
        df_entire_stats_history.to_csv(f'{stats_history_file_string}_{date_today}.csv')
        stats_history = df_entire_stats_history
    
    # scrape all odds based on stats history 
    if get_all_odds is True: 
        df_entire_odds_history = scraper.get_fighter_odds(stats_history)
        df_entire_odds_history.to_csv(f'{odds_history_file_string}_{date_today}.csv')
        odds_history = df_entire_odds_history

    # get upcoming fight stats
    if get_next_fight_stats_odds is True: 

        # scrape upcoming fights 
        next_fight_stats = scraper.scrape_upcoming_card()
        next_fight_date = next_fight_stats['event_date'][0] # unique date returned here, only one fightcard is scraped 
        
        # save next stats 
        upcoming_stats_path = f'{upcoming_stats_string}_{next_fight_date}.csv'
        next_fight_stats.to_csv(upcoming_stats_path, index=False)

        # save next_odds 
        upcoming_odds_path = f'{upcoming_odds_string}_{next_fight_date}.csv' # upcoming odds for the next fight card
        next_odds_df = scraper.get_fighter_odds(next_fight_stats)
        next_odds_df.to_csv(upcoming_odds_path, index=False)

    # merge stats history scrape_until, merge odds_history with missing odds by fighter/date 
    if get_missing_stats:

        # scrape stats from current date to prev_fight_date 
        prev_fight_date = stats_history.sort_values(by='event_date', ascending=False, inplace=False)['event_date'][0]#get index 0, largest value in non ascending order  
        missing_stats = scraper.scrape_until(prev_fight_date)
        missing_odds = scraper.get_fighter_odds(missing_stats) # finds odds based on fighters/date
    
        # merge with stats history, save by current date
        stats_history = pd.concat([stats_history, missing_stats])
        stats_history = stats_history.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        stats_history.to_csv(f'{stats_history_file_string}_{date_today}.csv')

        odds_history = pd.concat([odds_history, missing_odds])
        odds_history = odds_history.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        odds_history.to_csv(f'{odds_history_file_string}_{date_today}.csv')

    # Take stats_history, odds_history, next_fight_stats, next_fight_odds
    # Build dfs with all features used for model training and 
    if generate_model_df is True: 

        next_fight_stats = pd.read_csv(f'{upcoming_stats_string}_{next_fight_date}.csv')
        next_odds_df = pd.read_csv(f'{upcoming_odds_string}_{next_fight_date}.csv')
        next_fight_date = next_fight_stats['event_date'][0]

        odds_stats_df, upcoming_df = features.build_all_stats(stats_history, next_fight_stats, odds_history, next_odds_df)

        odds_stats_df.to_csv(fr'C:\Users\jcmar\my_files\SportsBetting\data\entire_odds_stats_{date_today}.csv', index=False)
        upcoming_df.to_csv(fr'C:\Users\jcmar\my_files\SportsBetting\data\upcoming_odds_stats_{next_fight_date}.csv', index=False)

