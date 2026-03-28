
import os 
import sys
from pathlib import Path 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from datetime import datetime 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(Path(__file__).parent))

from DataPipeline.webscrapers.scraping_pipeline import UFC_Webscraper
from DataPipeline.FeatureEngineering.BuildFeatures.fight_time_feats import single_event_features
from DataPipeline.github_utils import commit_if_changed, delete_and_commit


ml_history_fp = BASE_DIR / 'Data' / 'betting_results' / 'moneyline_results.csv'
parlay_history_fp = BASE_DIR / 'Data' / 'betting_results' / 'parlay_results.csv'
event_features_folder = BASE_DIR / "Data" / "upcoming_events" / "event_features" / "upcoming_odds_stats"

non_merged_stats_fp = BASE_DIR / "Data" / "non_merged_features" / "non_merged_stats.csv"
non_merged_odds_fp = BASE_DIR / "Data" / "non_merged_features" / "non_merged_odds.csv"

ml_pct_returns_fp = BASE_DIR / 'Data' / 'betting_results' / 'ml_pct_returns.csv'
parlay_pct_returns_fp = BASE_DIR / 'Data' / 'betting_results' / 'parlay_pct_returns.csv'


def archive_results():

    ml_bets_folder = BASE_DIR / "Data" / "upcoming_events" / "straight_bets" 
    parlay_bets_folder = BASE_DIR / "Data" / "upcoming_events" / "parlays"

    if os.path.exists(ml_history_fp):
        df_ml_history = pd.read_csv(ml_history_fp)
    else: 
         df_ml_history = pd.DataFrame({'fighter_red':[], 'fighter_blue':[], 'winner': [],
                                        'pred_name_open':[], 'pred_name_close1':[], 'pred_name_close2':[],
                                                    'open_red':[], 'open_blue':[],  
                                                    'close1_red':[], 'close1_blue':[],
                                                    'close2_red':[], 'close2_blue':[],
                                       'net_odds_open':[], 'net_odds_close1':[], 'net_odds_close2':[],
                                         'fstar_open':[], 'fstar_close1':[], 'fstar_close2':[], 
                                         'date':[]
                                         })

    if os.path.exists(parlay_history_fp): 
        df_parlay_history = pd.read_csv(parlay_history_fp)
    else: 
        df_parlay_history = pd.DataFrame(
                    {'choice_fighter_name_open':[], 'choice_fighter_name_close1':[], 'choice_fighter_name_close2':[],
                     'open_net_fstar':[], 'close1_net_fstar':[], 'close2_net_fstar':[],
                    'open_net_odds':[], 'close1_net_odds':[], 'close2_net_odds':[],
                    'date':[]
                    }
         )
        

    earliest_date = None
    for ml_file in os.listdir(ml_bets_folder):
        date_str = ml_file.split('_')[-1].replace('.csv', '')
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        if earliest_date is None or d < earliest_date:
            earliest_date = d

    earliest_date = earliest_date.strftime("%Y-%m-%d")
    scraped_stats = get_missing_stats(earliest_date) 
    df_single_event = single_event_features(scraped_stats)

    df_single_event['date'] = pd.to_datetime(df_single_event['date'])

    for ml_file in os.listdir(ml_bets_folder):

        date_str = ml_file.split('_')[-1].replace('.csv', '')
        d_ts = pd.to_datetime(date_str)  

        if pd.Timestamp.now().normalize() > d_ts:

            fp = ml_bets_folder / ml_file
            df_ml = pd.read_csv(fp).reset_index(drop=True)

            parlay_file = parlay_bets_folder / f'parlay_all_{date_str}.csv'
            parlay_df = pd.read_csv(parlay_file).reset_index(drop=True)

            df_event_date = df_single_event[df_single_event['date'] == d_ts].reset_index(drop=True)

            df_results_ml = calc_winner_ml(df_event_date, df_ml)
            df_results_ml['date'] = d_ts
            df_ml_history = pd.concat([df_ml_history, df_results_ml], axis=0).reset_index(drop=True)

            commit_if_changed(df_ml_history, ml_history_fp, f'updating money line results for fight date: {date_str}')

            parlay_results = calc_winner_parlay(parlay_df, df_event_date)
            parlay_results['date'] = d_ts
            df_parlay_history = pd.concat([df_parlay_history, parlay_results],axis=0).reset_index(drop=True)

            commit_if_changed(df_parlay_history, parlay_history_fp, f'Updating parlay results for fight date: {date_str}')



def delete_old_files():
    ml_bets_folder = BASE_DIR / "Data" / "upcoming_events" / "straight_bets" 
    parlay_bets_folder = BASE_DIR / "Data" / "upcoming_events" / "parlays"

    earliest_date = None
    for ml_file in os.listdir(ml_bets_folder):
        date_str = ml_file.split('_')[-1].replace('.csv', '')
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        if earliest_date is None or d < earliest_date:
            earliest_date = d

    earliest_date = earliest_date.strftime("%Y-%m-%d")

    for ml_file in os.listdir(ml_bets_folder):

        date_str = ml_file.split('_')[-1].replace('.csv', '')
        d_ts = pd.to_datetime(date_str)  
        parlay_file = parlay_bets_folder / f'parlay_all_{date_str}.csv'

        if pd.Timestamp.now().normalize() > d_ts:

            delete_and_commit(parlay_file, f'Deleting parlay bets for event {date_str}')
            delete_and_commit(ml_file, f'Deleting money line bets for event {date_str}')

            features_fp = event_features_folder / f"upcoming_odds_stats_{date_str}"
            delete_and_commit(features_fp, f'Deleting event features for event {date_str}')


def calc_winner_parlay(df_parlay, df_single_event):
     
    choice_fighters_open = df_parlay['choice_fighter_name_open'].values
    choice_fighters_close1 = df_parlay['choice_fighter_name_close1'].values
    choice_fighters_close2 = df_parlay['choice_fighter_name_close2'].values

    open_odds = df_parlay['parlay_odds_open']
    close1_odds = df_parlay['parlay_odds_close1']
    close2_odds = df_parlay['parlay_odds_close2']
    
    open_stake = df_parlay['parlay_fstar_open']
    close1_stake = df_parlay['parlay_fstar_close1']
    close2_stake = df_parlay['parlay_fstar_close2']

    winner_names = df_single_event['winner'].values

    open_win = df_parlay['choice_fighter_name_open'].isin(winner_names).all()
    close1_win = df_parlay['choice_fighter_name_close1'].isin(winner_names).all()
    close2_win = df_parlay['choice_fighter_name_close2'].isin(winner_names).all()


    profit_open = open_stake * open_odds if open_win else -open_stake
    profit_close1 = close1_stake * close1_odds if close1_win else -close1_stake
    profit_close2 = close2_stake * close2_odds if close2_win else -close2_stake

    open_net_fstar = open_stake if open_win else -open_stake
    close1_net_fstar = close1_stake if close1_win else -close1_stake
    close2_net_fstar = close2_stake if close2_win else -close2_stake

    parlay_results = pd.DataFrame({'open_net_fstar':open_net_fstar, 
                      'close1_net_fstar':close1_net_fstar, 
                      'close2_net_fstar':close2_net_fstar,
                        'open_net_odds':profit_open, 
                        'close1_net_odds':profit_close1, 
                        'close2_net_odds':profit_close2,
                        'choice_fighter_name_open':choice_fighters_open, 
                        'choice_fighter_name_close1':choice_fighters_close1, 
                        'choice_fighter_name_close2':choice_fighters_close2 })

    return parlay_results


def calc_winner_ml(df_single_event, df_money_line):

    assert df_single_event.shape[0] == df_money_line.shape[0], 'scraped results df shape mismatch with bets df'

    profit_open = pd.Series(0.0, index=range(df_single_event.shape[0]))
    profit_close1 = pd.Series(0.0, index=range(df_single_event.shape[0]))
    profit_close2 = pd.Series(0.0, index=range(df_single_event.shape[0]))

    for i, row in df_money_line.iterrows():
            
            winner_name = df_single_event['winner'].loc[i].lower()
            pred_open = row['pred_name_open'].lower()
            pred_close1 = row['pred_name_close1'].lower()
            pred_close2 = row['pred_name_close2'].lower()

            pred_color_open = 'red' if pred_open == winner_name else 'blue'
            pred_color_close1 = 'red' if pred_close1 == winner_name else 'blue'
            pred_color_close2 = 'red' if pred_close2 == winner_name else 'blue'

            open_odds = row[f'open_{pred_color_open}']
            close1_odds = row[f'close1_{pred_color_close1}']
            close2_odds = row[f'close2_{pred_color_close2}']

            open_stake = row['fstar_open'] 
            close1_stake = row['fstar_close1']
            close2_stake = row['fstar_close2'] 

            profit_open[i] = moneyline_profit(open_stake, open_odds) if pred_open == winner_name else -open_stake * open_odds
            profit_close1[i] = moneyline_profit(close1_stake, close1_odds) if pred_close1 == winner_name else -close1_stake * close1_odds
            profit_close2[i] = moneyline_profit(close2_stake, close2_odds) if pred_close2 == winner_name else -close2_stake * close2_odds


    df_data = pd.concat([profit_open, profit_close1, profit_close2], axis=1)
    df_data.columns = ['net_odds_open', 'net_odds_close1', 'net_odds_close2']

    df_data = pd.concat([df_data, df_money_line[[   'fighter_red', 
                                                    'fighter_blue',
                                                    'pred_name_open', 
                                                    'pred_name_close1', 
                                                    'pred_name_close2',
                                                    'open_red', 'open_blue',  
                                                    'close1_red', 'close1_blue',
                                                    'close2_red', 'close2_blue',
                                                    'fstar_open', 
                                                    'fstar_close1', 
                                                    'fstar_close2',
                                     ]]], axis=1).reset_index(drop=True)
    
    df_data = pd.concat([df_data, df_single_event['winner']], axis=1).reset_index(drop=True)


    mask = (
        (df_money_line['fstar_open'] != 0 | df_money_line['fstar_open'].notna()) &
        (df_data['net_odds_open'] != 0 | df_data['net_odds_open'].notna())
    )
    assert mask.all(), "Money Line results profit error"

    return df_data


def moneyline_profit(stake, odds):
    if odds > 0:
        return stake * (odds / 100)
    else:
        return stake * (100 / abs(odds))


def get_missing_stats(prev_fight_date): 
    """ previous fight date from file string """

    scraper = UFC_Webscraper()
    missing_stats = scraper.scrape_until(prev_fight_date)
    missing_odds = scraper.get_fighter_odds(missing_stats) 

    if os.path.exists(non_merged_stats_fp):
        df_stats_missing = pd.read_csv(non_merged_stats_fp)
        df_stats_missing = pd.concat([df_stats_missing, missing_stats], axis=0).reset_index(drop=True)
    else:
        df_stats_missing = missing_stats

    commit_if_changed(df_stats_missing, non_merged_stats_fp, f'Updating Non Merged Stats for fight date: {prev_fight_date}')

    # merge odds history 
    if os.path.exists(non_merged_odds_fp):
        df_odds_missing = pd.read_csv(non_merged_odds_fp)
        df_odds_missing = pd.concat([df_odds_missing, missing_odds], axis=0).reset_index(drop=True)
    else:
        df_odds_missing = missing_odds

    commit_if_changed(df_odds_missing, non_merged_odds_fp, f'Updating Non Merged Odds for fight date: {prev_fight_date}')
    return df_stats_missing


def returns_by_date():
    
    df_ml = pd.read_csv(ml_history_fp)
    df_parlay = pd.read_csv(parlay_history_fp)

    types = ['open', 'close1', 'close2']
    ml_pct_returns = {type_:[] for type_ in types}
    parlay_pct_returns = {type_:[] for type_ in types}

    for date, group in df_ml.groupby('date'): 
        
        parlay_group = df_parlay[df_parlay['date']==date]

        for type_ in types: 
            ml_pct_returns[type_].append(group[f'net_odds_{type_}'].sum())

            parlay_net = parlay_group[f'{type_}_net_odds'].iloc[0]
            parlay_pct_returns[type_].append(parlay_net)

    df_ml_pct = pd.DataFrame(ml_pct_returns)
    df_parlay_pct = pd.DataFrame(parlay_pct_returns)

    commit_if_changed(df_ml_pct, ml_pct_returns_fp, f'Saving Percent Returns ML')

    commit_if_changed(df_parlay_pct, parlay_pct_returns_fp, f'Saving Percent Returns Parlay')





if __name__ == "__main__":
    archive_results()
    returns_by_date()
    delete_old_files()