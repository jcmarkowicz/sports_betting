import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np 
import pandas as pd 
from FeatureEngineering.ufc_features import single_event_features, apply_rolling_stats, non_rolling_stats, upcoming_event_features
from FeatureEngineering.odds_features import build_odds_features
from FeatureEngineering.feature_functions import count_fav_dog

class FeatureEngineering: 
    """Requires df with all stats and odds merged, computes ai model features"""

    def __init__(self):
        pass

    def standardize_features(self, df):
        single_features = single_event_features(df)
         #rolling features currently relies on these column names 
        rolling_features = apply_rolling_stats(single_features)
        all_features = non_rolling_stats(rolling_features)
        return all_features
    
    def build_all_stats(self, stats_df, upcoming_stats, odds_df, upcoming_odds):

        total_odds = pd.concat([odds_df, upcoming_odds]).reset_index(drop=True) # combine upcoming odds and odds history
        total_odds = build_odds_features(total_odds)

        past_event_stats = single_event_features(stats_df.copy())
        # past_event_stats.to_csv(r'C:\Users\jcmar\my_files\SportsBetting\data\ufc_singe_event_features.csv', index=False)
        past_event_stats = past_event_stats.loc[:, ~past_event_stats.columns.str.contains('^Unnamed')]

        upcoming_single_event = upcoming_event_features(upcoming_stats)
        
        # Create empty rows/columns for pre fight stats 
        exclude_cols = ['fighter_red', 'fighter_blue']
        stats_cols = [col for col in past_event_stats.columns if col not in exclude_cols]
        upcoming_features_NA = pd.DataFrame(index=range(upcoming_single_event.shape[0]), columns=stats_cols) 
        empty_df = pd.concat([upcoming_single_event[exclude_cols], upcoming_features_NA], axis=1) # Combine fighter names with NaN stats
        empty_df.columns = past_event_stats.columns # Assumes same order and length

        # Overwrite the scraped columns with actual values
        scraped_columns = [
            'fighter_red', 'fighter_blue', 'weight_class', 'event_date',
            'reach_red', 'reach_blue', 'height_red', 'height_blue',
            'red_age', 'blue_age', 'event_location'
        ]

        for col in scraped_columns:
            if col in empty_df.columns and col in upcoming_single_event.columns:
                empty_df[col] = upcoming_single_event[col]

        empty_df['event_date'] = pd.to_datetime(empty_df['event_date']) # Make sure event_date is datetime
        combined_df = pd.concat([empty_df, past_event_stats], axis=0).reset_index(drop=True) # Combine with past event stats

        rolling_fp = r'C:\Users\jcmar\my_files\SportsBetting\data\ufc_new_rolling.csv'
        rolling_df = apply_rolling_stats(combined_df) #sort here 
        rolling_df.to_csv(rolling_fp, index=False)

        total_df = non_rolling_stats(rolling_df)
        total_df.to_csv(r'C:\Users\jcmar\my_files\SportsBetting\data\new_combined.csv', index=False)

        merged_df = self.standardized_merge(total_df, total_odds)
        merged_df = merged_df.sort_values(by='date', ascending=True).reset_index(drop=True)

        # counts of fav and dog 
        merged_df[['fav_counts_red', 'dog_counts_red',
            'fav_counts_blue', 'dog_counts_blue']] = count_fav_dog(merged_df)

        odds_stats_history = merged_df.iloc[:-upcoming_stats.shape[0], :]
        # odds_stats_history.to_csv(full_file_path)
        upcoming_df = merged_df.iloc[-upcoming_stats.shape[0]:, :]
        # upcoming_df.to_csv(upcoming_file_path)
        
        return odds_stats_history, upcoming_df

    def standardize_dates(self, stats, odds):
        # dates in odds_df/stats_df are +- 1 day apart, need these to be equal 

        # iterate through stats_df rows, get target date fropm stats
        for i, row in stats.iterrows():

            #find all matchups between red and blue fighter in odds
            mask = (odds['red_clean'] == row['red_clean']) & (odds['blue_clean'] == row['blue_clean']) 

            # if odds contains the stats matchup 
            if mask.any():
                for i in odds[mask].index:

                    # iterate and find dates of each red/blue matchup 
                    target_date = odds.at[i, 'date']

                    # check if current stats date matches with +- odds date 
                    if (row['date'] - pd.Timedelta(days=1) == target_date) or (row['date'] + pd.Timedelta(days=1) == target_date):
                        # if found, replace odds_date with stats date for standardization 
                        odds.at[i, 'date'] = row['date'] # set the date in odds to the date in stats 
        return odds 

    def clean_col(self, col):
        col = col.str.lower() \
            .str.replace('-', ' ', regex=False) \
            .str.replace('.', '', regex=False) \
            .str.replace("'", '', regex=False) \
            .str.replace(r'\bsaint\b', 'st', case=False, regex=True) # add space here for fixxing other fighter names 
        return col 

    def standardized_merge(self, stats_df, odds_df):
        odds_df = odds_df.loc[:, ~odds_df.columns.str.contains('^Unnamed')] # filter out columns that contain 'Unamed'

        stats = stats_df.reset_index(drop=True).copy()
        odds = odds_df.reset_index(drop=True).copy()

        odds['date'] = pd.to_datetime(odds['event_date'])
        stats['date'] = pd.to_datetime(stats['date'])

        odds = odds.dropna(subset=['date']) # remove rows based on rows in 'date' that are NA 

        # clean names in stats col 
        stats['red_clean'] = self.clean_col(stats['red_fighter']) # clean names so that the names in odds and stats match 
        stats['blue_clean'] = self.clean_col(stats['blue_fighter'])

        stats['red_fighter_stats'] = stats['red_fighter'] # fighter names coming from stats df 
        stats['blue_fighter_stats'] = stats['blue_fighter']

        # clean names in odds col 
        odds['red_clean'] = self.clean_col(odds['red_fighter'])
        odds['blue_clean'] = self.clean_col(odds['blue_fighter'])

        odds['red_fighter_odds'] = odds['red_fighter']
        odds['blue_fighter_odds'] = odds['blue_fighter']

        odds = self.standardize_dates(stats, odds)
        odds.to_csv(r'C:\Users\jcmar\my_files\SportsBetting\data\look_at_odds.csv')

        # merge on fighter names and date 
        new_df = pd.merge(stats, odds, on=['red_clean', 'blue_clean','date'], how='left')

        new_df = new_df.loc[:, ~new_df.columns.str.contains('^Unnamed')]
        new_df = new_df.rename(columns={'red_clean': 'red_fighter', 'blue_clean':'blue_fighter'})

        new_df = new_df.drop(columns=['red_fighter_x','red_fighter_y','blue_fighter_x','blue_fighter_y','event_date_y','event_date_x', 
                                    'og_red_fighter', 'og_blue_name']) # remove cols with _x or _y, these are the uncleaned columns that merged over 
        
        new_df = new_df.drop_duplicates().reset_index(drop=True) # duplicat rows because of undstandardized fighter name columns 
        return new_df



        

   