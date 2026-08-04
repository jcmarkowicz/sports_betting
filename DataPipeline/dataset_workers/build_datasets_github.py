import pandas as pd 
from datetime import datetime 

from DataPipeline.utils.bets_utils import generate_bets
from DataPipeline.utils.email_utils import email_bets
from DataPipeline.utils.github_utils import commit_if_changed 

from DataPipeline.FeatureEngineering.features_pipeline import FeatureEngineering
from DataPipeline.webscrapers.scraping_pipeline import UFC_Webscraper

from config import config


# find the most recent stats and odds history with most recent timestamp 
# recent date from last merger with missing stats
# from Data/HISTORY folder 
date_today = datetime.now().strftime("%Y-%m-%d") 

get_all_stats = False
get_all_odds = False

get_next_fight_stats_odds = True

get_missing_stats = False
generate_model_df = True 




#Folders for ALL scraped data
non_merged_stats = pd.read_csv(config.non_merged_stats_fp)
non_merged_stats = non_merged_stats.drop(columns=[col for col in non_merged_stats.columns if "Unnamed" in col])

non_merged_odds = pd.read_csv(config.non_merged_odds_fp)
non_merged_odds = non_merged_odds.drop(columns=[col for col in non_merged_odds.columns if "Unnamed" in col])


stats_history = pd.read_csv(config.stats_history_file_string) # frames BEFORE any feature engineering 
stats_history = stats_history.drop(columns=[col for col in stats_history.columns if "Unnamed" in col])

assert set(stats_history.columns) == set(non_merged_stats.columns), 'Column misalignment stats history'
stats_history = pd.concat([stats_history, non_merged_stats], axis=0, ignore_index=True)


odds_history = pd.read_csv(config.odds_history_file_string)
odds_history = odds_history.drop(columns=[col for col in odds_history.columns if "Unnamed" in col])

assert set(odds_history.columns) == set(non_merged_odds.columns), 'Column misalignment odds history'
odds_history = pd.concat([odds_history, non_merged_odds], axis=0, ignore_index=True)

# drop duplicate from history
stats_history = stats_history[~stats_history.duplicated(
    subset=['fighter_red', 'fighter_blue', 'event_date'],
    keep='first'
)]




if __name__ == "__main__":
    scraper = UFC_Webscraper()
    features = FeatureEngineering()

    # scrape all stats
    if get_all_stats is True: 
        df_entire_stats_history = scraper.scrape_until(date=None)
        df_entire_stats_history.to_csv(f'{config.stats_history_file_string}_{date_today}.csv')
        stats_history = df_entire_stats_history
    
    # scrape all odds based on stats history 
    if get_all_odds is True: 
        df_entire_odds_history = scraper.get_fighter_odds(stats_history)
        df_entire_odds_history.to_csv(f'{config.odds_history_file_string}_{date_today}.csv')
        odds_history = df_entire_odds_history




    # get upcoming fight stats
    # can run withoug getting missing stats, returns dataframes in upcoming folder 
    if get_next_fight_stats_odds is True: 

        # scrape upcoming events, MULTIPLE DATES handled in generate_model_df  
        next_fight_stats = scraper.scrape_upcoming_card()
        next_fight_date = next_fight_stats['event_date'][0] # unique date returned here, only one fightcard is scraped 
        
        # save next stats 
        commit_if_changed(next_fight_stats, 
                          f'{config.upcoming_scraped_stats_string}.csv', 
                          f'Updating upcoming scraped stats starting at: {next_fight_date}')

        # save next_odds 
        next_odds_df = scraper.get_fighter_odds(next_fight_stats)
        commit_if_changed(next_odds_df, 
                          f'{config.upcoming_scraped_odds_string}.csv' , 
                          f'Updating upcoming scraped odds starting at: {next_fight_date}')




    # merge stats/odds history 
    if get_missing_stats:

        # scrape stats from current date to prev_fight_date (last date in stats_history) 
        prev_fight_date = stats_history.sort_values(by='event_date', ascending=False, inplace=False).reset_index(drop=True)['event_date'][0] # get index 0, largest value in non ascending order  
        print(f"Previous fight date in stats history: {prev_fight_date}")

        missing_stats = scraper.scrape_until(prev_fight_date)
        missing_odds = scraper.get_fighter_odds(missing_stats) # finds odds based on fighters/date
    
        # merge with stats history, save by current date
        stats_history = pd.concat([stats_history, missing_stats], axis=0)
        stats_history['event_date'] = pd.to_datetime(stats_history['event_date'], errors='coerce')
        
        stats_history = stats_history.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        stats_history.to_csv(f'{config.stats_history_file_string}_{date_today}.csv')

        # merge odds history 
        odds_history = pd.concat([odds_history, missing_odds], axis=0)
        odds_history['event_date'] = pd.to_datetime(odds_history['event_date'], errors='coerce')

        odds_history = odds_history.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        odds_history.to_csv(f'{config.odds_history_file_string}_{date_today}.csv')




    # Build features for history and upcoming 
    if generate_model_df is True: 

        # get scraped data for ALL future events 
        next_stats_df = pd.read_csv(f'{config.upcoming_scraped_stats_string}.csv')
        next_odds_df = pd.read_csv(f'{config.upcoming_scraped_odds_string}.csv')

        # FEATURES DATAFRAMES 
        odds_stats_df, upcoming_df = features.build_all_stats(stats_history, next_stats_df, odds_history, next_odds_df)

        upcoming_groups = upcoming_df.groupby('date')
        for date, group in upcoming_groups:

            # group includes all event fights 
            group = group.reset_index(drop=True)

            date_str = date.strftime("%Y-%m-%d")   # or "%Y%m%d"
            event_file_path = config.upcoming_events_folder / f"upcoming_odds_stats_{date_str}.csv"

            # no odds available
            if group[['open_red', 'open_blue']].isna().all().all():
                pass

            # new event found 
            elif not event_file_path.exists():
                email_bets(group, date_str)
            
            # else check if different from existing
            else:
                # existing open will be nan if not available 
                existing_df = pd.read_csv(event_file_path)

                aligned = group.merge(
                    existing_df[["fighter_red", "fighter_blue", "open_red", "open_blue"]],
                    on=["fighter_red", "fighter_blue"],
                    how="left"
                )

                mask_na_update = (
                    (aligned['open_red_y'].isna() & aligned['open_red_x'].notna()) |
                    (aligned['open_blue_y'].isna() & aligned['open_blue_x'].notna())
                )

                sub_group = group[mask_na_update]
                if not sub_group.empty:
                    email_bets(sub_group, date_str)
            
            # commit event csv data every time 
            commit_if_changed(group, event_file_path, f'Updating Features for {date_str}')

            # generate betting picks with latest data 
            df_bets_all, df_parlay_all = generate_bets(group, select_odds=None)

            straight_path = config.ml_bets_folder/ f"ml_all_{date_str}.csv"
            parlay_path   = config.parlay_bets_folder/ f"parlay_all_{date_str}.csv"

            # commit csv 
            commit_if_changed(df_bets_all, straight_path, f'Update Bets for {date_str}')
            commit_if_changed(df_parlay_all, parlay_path, f'Update Parlay for {date_str}')
