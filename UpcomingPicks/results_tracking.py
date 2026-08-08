
import os 
import sys
from pathlib import Path 

import numpy as np
import pandas as pd 


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(__file__).parent))

from DataPipeline.FeatureEngineering.features_pipeline import FeatureEngineering 
from DataPipeline.webscrapers.scraping_pipeline import UFC_Webscraper

from DataPipeline.utils.github_utils import commit_if_changed, delete_and_commit
from DataPipeline.utils.bets_utils import generate_bets

from config import config
import time 


def archive_results(start_date='2026-07-25'):
    """
    Maintain moneyline and parlay history betting results. 

    Give a start date, scrape all stats of odds from now until / including that date. 
    For each event the following is calculated: 
        - Money line results
            - Create or key existing? 
            - calc_winner_ml(df_single_event, df_money_line): 
                 - returns df with predictions, net odds 
        - Parlay results
             - Create or key existing?
             - calc_winner_parlay(df_parlay, df_single_event): 
                - returns net odds and predictions 

        - Missing odds stats that are not in the original history 
    
    Create and maintain the following 3 dataframes: 
        - df_ml_history: money line results history 
        - df_parlay_history: parlay results history 
        - df_missing_stats: missing stats and odds history 

    It is possible to iterate through events that are already included, therefore it is crucial to check for duplicates 

    How to handle existing ml and parlay files: 
        - delete 

    """

    ml_bets_folder = BASE_DIR / "Data" / "upcoming_events" / "straight_bets" 
    parlay_bets_folder = BASE_DIR / "Data" / "upcoming_events" / "parlays"

    if os.path.exists(config.ml_history_fp):
        df_ml_history = pd.read_csv(config.ml_history_fp)
        df_ml_history = df_ml_history.loc[:, ~df_ml_history.columns.str.contains('^Unnamed')]
    else: 
         df_ml_history = pd.DataFrame()

    if os.path.exists(config.parlay_history_fp): 
        df_parlay_history = pd.read_csv(config.parlay_history_fp)
        df_parlay_history = df_parlay_history.loc[:, ~df_parlay_history.columns.str.contains('^Unnamed')]

    else: 
        df_parlay_history = pd.DataFrame()

    scraped_stats, scraped_odds, upcoming_df = get_missing_stats(start_date) 

    # df_single_event = single_event_features(df_stats)
    # df_rolling = apply_rolling_stats(df_single_event)
    # df_feats = non_rolling_stats(df_rolling)
    # df_total = features.standardized_merge(df_feats, df_odds)

    # df_total['date'] = pd.to_datetime(df_total['date'])
    # t2 = time.time()
    # print(f"Time to generate features: {t2-t1:.2f} seconds" )
    # print(f"Total events to process: {df_total.shape[0]}")

    # iterate through each event date and calculate the results for each event
    # df_ml_history["date"] = pd.to_datetime(
    #     df_ml_history["date"], format="mixed"
    # ).dt.date

    # df_parlay_history["date"] = pd.to_datetime(
    #     df_parlay_history["date"], format="mixed"
    # ).dt.date

    for date, event_group in upcoming_df.groupby('date'):

        event_group =  event_group.copy().reset_index(drop=True)

        date_str = date.strftime("%Y-%m-%d")
        d_ts = pd.to_datetime(date_str)  

        # check if event date is in the past 
        if pd.Timestamp.now().normalize() > d_ts:

            # ml_fp = ml_bets_folder / f'ml_all_{date_str}.csv'
            # parlay_fp = parlay_bets_folder / f'parlay_all_{date_str}.csv'
            # if ml_fp.exists():
            #     df_ml = pd.read_csv(ml_fp).reset_index(drop=True)

            #     if not parlay_fp.exists():
            #         df_ml_test, df_parlay = generate_bets(event_group, select_odds=None)
            #         assert df_ml == df_ml_test, f"Money line bets for {date_str} do not match generated bets"
            # else: 
            #     df_ml, df_parlay = generate_bets(event_group, select_odds=None)

            df_ml, df_parlay = generate_bets(event_group, select_odds=None)

            # calculate the winners and commit 
            df_results_ml = calc_winner_ml(event_group, df_ml)
            df_results_ml['date'] = d_ts

            df_ml_history = pd.concat([df_ml_history, df_results_ml], axis=0, ignore_index=True)
            df_ml_history["date"] = pd.to_datetime(
                df_ml_history["date"], format="mixed"
            ).dt.date
            df_ml_history = df_ml_history.drop_duplicates(subset=['fighter_red', 'fighter_blue', 'date'], keep='last')

            parlay_results = calc_winner_parlay(df_parlay, event_group)
            parlay_results['date'] = d_ts

            df_parlay_history = pd.concat([df_parlay_history, parlay_results],axis=0, ignore_index=True)
            df_parlay_history["date"] = pd.to_datetime(
                df_parlay_history["date"], format="mixed"
            ).dt.date
            df_parlay_history = df_parlay_history.drop_duplicates(subset=['choice_fighter_name_open','choice_fighter_name_close1','choice_fighter_name_close2', 'date'])

            # all of these deleted files are regenerated in the above functions 
            delete_old_files(date_str)

    commit_if_changed(df_parlay_history, config.parlay_history_fp, f'Updating parlay results')
    commit_if_changed(df_ml_history, config.ml_history_fp, f'updating money line results')


def delete_old_files(date_str):
    """
    Delete ml bets, parlay bets, and event features for a given date.
    """
    ml_bets_folder = BASE_DIR / "Data" / "upcoming_events" / "straight_bets" 
    parlay_bets_folder = BASE_DIR / "Data" / "upcoming_events" / "parlays"
    features_folder = config.event_features_folder 

    ml_fp = ml_bets_folder / f'ml_all_{date_str}.csv'
    delete_and_commit(ml_fp, f'Deleting Money Line bets for event {date_str}')

    parlay_fp = parlay_bets_folder / f'parlay_all_{date_str}.csv'
    delete_and_commit(parlay_fp, f'Deleting parlay bets for event {date_str}')

    features_fp = features_folder/ f"upcoming_odds_stats_{date_str}.csv"
    delete_and_commit(features_fp, f'Deleting event features for event {date_str}')


def calc_winner_parlay(df_parlay, df_single_event):

    winner_bool = df_single_event['winner']
    winner_name = df_single_event['winner_name']
    print(f'Winner Name GT in Parlay: {winner_name}')

    choice_index_open = df_parlay['fight_index_open'].astype(int)
    choice_index_close1 = df_parlay['fight_index_close1_stack'].astype(int)
    choice_index_close2 = df_parlay['fight_index_close2_stack'].astype(int)
    print(f'choice index open: {choice_index_open}')
    print(f'choice index close1: {choice_index_close1}')
    print(f'choice index close2: {choice_index_close2}')

    choice_fighters_open = df_parlay['choice_fighter_bool_open'].to_numpy(dtype=int)
    choice_fighters_close1 = df_parlay['choice_fighter_bool_close1_stack'].to_numpy(dtype=int)
    choice_fighters_close2 = df_parlay['choice_fighter_bool_close2_stack'].to_numpy(dtype=int)

    winners_bool_open = winner_bool.loc[choice_index_open].to_numpy(dtype=int)
    winner_bool_close1 = winner_bool.loc[choice_index_close1].to_numpy(dtype=int)
    winner_bool_close2 = winner_bool.loc[choice_index_close2].to_numpy(dtype=int)

    open_red = df_single_event.loc[choice_index_open]['open_red'].to_numpy()
    open_blue = df_single_event.loc[choice_index_open]['open_blue'].to_numpy()

    close1_red = df_single_event.loc[choice_index_close1]['close1_red'].to_numpy()
    close1_blue = df_single_event.loc[choice_index_close1]['close1_blue'].to_numpy()

    close2_red = df_single_event.loc[choice_index_close2]['close2_red'].to_numpy()
    close2_blue = df_single_event.loc[choice_index_close2]['close2_blue'].to_numpy()

    open_win = (choice_fighters_open == winners_bool_open).all()
    close1_win = (choice_fighters_close1 == winner_bool_close1).all()
    close2_win = (choice_fighters_close2 == winner_bool_close2).all()

    open_odds = df_parlay['parlay_odds_open'].to_numpy()
    close1_odds = df_parlay['parlay_odds_close1_stack'].to_numpy()
    close2_odds = df_parlay['parlay_odds_close2_stack'].to_numpy()
    
    open_stake = df_parlay['parlay_fstar_open'].to_numpy()
    close1_stake = df_parlay['parlay_fstar_close1_stack'].to_numpy()
    close2_stake = df_parlay['parlay_fstar_close2_stack'].to_numpy()

    profits = moneyline_profit(open_stake, open_odds)
    profit_open = np.where(open_win, open_odds, -1)
  
    profits = moneyline_profit(close1_stake, close1_odds)
    profit_close1 = np.where(close1_win, close1_odds, -1)

    profits = moneyline_profit(close2_stake, close2_odds)
    profit_close2 = np.where(close2_win, close2_odds, -1)

    open_net_fstar = np.where(open_win, open_stake, -open_stake)
    close1_net_fstar = np.where(close1_win, close1_stake, -close1_stake)
    close2_net_fstar = np.where(close2_win, close2_stake, -close2_stake)

    winner_name_open = winner_name.loc[choice_index_open].to_numpy()
    winner_name_close1 = winner_name.loc[choice_index_close1].to_numpy()
    winner_name_close2 = winner_name.loc[choice_index_close2].to_numpy()

    print(f'Winner Name Parlay Open: {winner_name_open}')    
    print(f'Winner Name Parlay close 1: {winner_name_close1}')
    print(f'Winner Name Parlay Close2: {winner_name_close2}')

    parlay_results = pd.DataFrame({
        'open_net_fstar':open_net_fstar, 'close1_net_fstar':close1_net_fstar, 'close2_net_fstar':close2_net_fstar,

        'open_red':open_red, 'open_blue':open_blue, 
        'close1_red':close1_red,'close1_blue':close1_blue, 
        'close2_red':close2_red, 'close2_blue':close2_blue, 

        'open_net_odds':profit_open, 'close1_net_odds':profit_close1, 'close2_net_odds':profit_close2,
        'open_fstar':open_stake,'close1_fstar':close1_stake, 'close2_fstar':close2_stake, 

        'choice_fighter_bool_open':choice_fighters_open, 
        'choice_fighter_bool_close1':choice_fighters_close1, 
        'choice_fighter_bool_close2':choice_fighters_close2, 

        'winner_bool_open':winners_bool_open,
        'winner_bool_close1':winner_bool_close1,
        'winner_bool_close2':winner_bool_close2,

        'choice_fighter_name_open':df_parlay['choice_fighter_name_open'], 
        'choice_fighter_name_close1':df_parlay['choice_fighter_name_close1_stack'], 
        'choice_fighter_name_close2':df_parlay['choice_fighter_name_close2_stack'],

        'fight_index_open':df_parlay['fight_index_open'],
        'fight_index_close1_stack':df_parlay['fight_index_close1_stack'],
        'fight_index_close2_stack':df_parlay['fight_index_close2_stack'],

        'winner_name':winner_name_open,
        'winner_name_close1': winner_name_close1,
        'winner_name_close2': winner_name_close2, 
    })
    test = parlay_results[['winner_name', 'winner_name_close1', 'winner_name_close2']].head()

    print(f'Parlay Results Columsn for winners: {test}')
    return parlay_results

def calc_winner_ml(df_single_event, df_money_line):

    assert df_single_event.shape[0] == df_money_line.shape[0], 'scraped results df shape mismatch with bets df'

    winner_bool = df_single_event['winner'].to_numpy(dtype=float)

    # pred_bool_open = df_money_line['pred_bool_open'].to_numpy()
    # pred_bool_close1 = df_money_line['pred_bool_close1'].to_numpy()
    # pred_bool_close2 = df_money_line['pred_bool_close2'].to_numpy()

    pred_open = df_money_line['pred_winner_open'].to_numpy(dtype=float)
    pred_close1 = df_money_line['pred_winner_close1_stack'].to_numpy(dtype=float)
    pred_close2 = df_money_line['pred_winner_close2_stack'].to_numpy(dtype=float)

    def calc_color(x):
        result = np.empty(len(x), dtype=object)

        result[x == 1] = "red"
        result[x == 0] = "blue"
        result[np.isnan(x)] = np.nan

        return result

    def odds_from_color(df, color_list, odds_type):
        odds = np.empty(len(color_list), dtype=float)

        for i, color in enumerate(color_list):
            if color == "red":
                odds[i] = df[f"{odds_type}_red"].iloc[i]
            elif color == "blue":
                odds[i] = df[f"{odds_type}_blue"].iloc[i]
            else:
                odds[i] = np.nan  # Handle NaN or unexpected values

        return odds
    
    pred_color_open = calc_color(pred_open)
    pred_color_close1 = calc_color(pred_close1)
    pred_color_close2 = calc_color(pred_close2)
    
    open_odds =  odds_from_color(df_money_line, pred_color_open, "open")
    close1_odds = odds_from_color(df_money_line, pred_color_close1, "close1")
    close2_odds = odds_from_color(df_money_line, pred_color_close2, "close2")

    open_stake = df_money_line['fstar_open'].to_numpy(dtype=float)
    close1_stake = df_money_line['fstar_close1_stack'].to_numpy(dtype=float)
    close2_stake = df_money_line['fstar_close2_stack'].to_numpy(dtype=float)

    profits = moneyline_profit(open_stake, open_odds)
    profit_open = np.where(
        pred_open == winner_bool,
        open_odds,
        -1
    )
    profits = moneyline_profit(close1_stake, close1_odds)
    profit_close1 = np.where(
        pred_close1 == winner_bool,
        close1_odds, 
        -1
    )
    profits = moneyline_profit(close2_stake, close2_odds)
    profit_close2 = np.where(
        pred_close2 == winner_bool,
        close2_odds,    
        -1
    )

    winner_name = df_single_event['winner_name'].to_numpy()
    print(f'Winner Name in ML: {winner_name}')

    df_data = pd.DataFrame({
        "net_odds_open": profit_open,
        "net_odds_close1": profit_close1,
        "net_odds_close2": profit_close2,
        'winner_bool': winner_bool,
        'winner_name':winner_name,
    })

    df_data = pd.concat([
        df_data,
        df_money_line[[
            "fighter_red", "fighter_blue",

            "pred_name_open","pred_name_close1", "pred_name_close2",
            'pred_winner_open', 'pred_winner_close1_stack','pred_winner_close2_stack',

            "open_red", "open_blue",
            "close1_red", "close1_blue",
            "close2_red", "close2_blue",

            "fstar_open", "fstar_close1", "fstar_close2",
        ]].reset_index(drop=True),
    ], axis=1)
    
    # mask = (
    #     (df_money_line['fstar_open'] != 0 | df_money_line['fstar_open'].notna()) &
    #     (df_data['net_odds_open'] != 0 | df_data['net_odds_open'].notna())
    # )
    # assert mask.all(), "Money Line results profit error"
    return df_data


def moneyline_profit(stake, odds):
    """ if scalars, numpy returns a 0 dimensional array which acts like a python float """
    stake = np.asarray(stake)
    odds = np.asarray(odds)

    return np.where(
        odds > 0,
        stake * (odds / 100),
        stake * (100 / np.abs(odds))
    )

def get_missing_stats(prev_fight_date): 
    """ previous fight date from file string """

    scraper = UFC_Webscraper()
    features = FeatureEngineering()

    stats_history = pd.read_csv(config.stats_history_file_string) # frames BEFORE any feature engineering 
    stats_history = stats_history.drop(columns=[col for col in stats_history.columns if "Unnamed" in col])

    odds_history = pd.read_csv(config.odds_history_file_string)
    odds_history = odds_history.drop(columns=[col for col in odds_history.columns if "Unnamed" in col])

    missing_stats = scraper.scrape_until(prev_fight_date)
    missing_odds = scraper.get_fighter_odds(missing_stats) 


    if os.path.exists(config.non_merged_stats_fp):
        stats_missing_prev = pd.read_csv(config.non_merged_stats_fp)
        stats_missing_prev = stats_missing_prev.drop(columns=[col for col in stats_missing_prev.columns if "Unnamed" in col])
        stats_missing_all = pd.concat([stats_missing_prev, missing_stats], axis=0, ignore_index=True)

    else:
        stats_missing_all = missing_stats.copy()

    stats_missing_all['event_date'] = pd.to_datetime(
        stats_missing_all["event_date"], format="mixed"
    ).dt.date
    stats_missing_all = stats_missing_all.drop_duplicates(subset=['fighter_red', 'fighter_blue', 'event_date', 'event_name'], keep='last').reset_index(drop=True)
    commit_if_changed(stats_missing_all, config.non_merged_stats_fp, f'Updating Non Merged Stats for fight date: {prev_fight_date}')

    all_stats = pd.concat([stats_history, stats_missing_all], axis=0)
    all_stats['event_date'] = pd.to_datetime(
        all_stats["event_date"], format="mixed"
    ).dt.date

    print(f'all stats shape dupes: {all_stats.shape}')
    all_stats = all_stats.drop_duplicates(subset=['fighter_red', 'fighter_blue', 'event_date', 'event_name'])
    print(f'all stats shape NO dupes: {all_stats.shape}')


    # merge odds history 
    if os.path.exists(config.non_merged_odds_fp):
        df_odds_missing = pd.read_csv(config.non_merged_odds_fp)
        df_odds_missing = df_odds_missing.drop(columns=[col for col in df_odds_missing.columns if "Unnamed" in col])   
        odds_missing_all = pd.concat([df_odds_missing, missing_odds], axis=0, ignore_index=True)
    else:
        odds_missing_all = missing_odds.copy()

    odds_missing_all['event_date'] = pd.to_datetime(
        odds_missing_all["event_date"], format="mixed"
    ).dt.date
    odds_missing_all = odds_missing_all.drop_duplicates(subset=['event_date', 'blue_fighter', 'red_fighter'], keep='last').reset_index(drop=True)
    commit_if_changed(odds_missing_all, config.non_merged_odds_fp, f'Updating Non Merged Odds for fight date: {prev_fight_date}')

    all_odds = pd.concat([odds_history, odds_missing_all], axis=0)
    all_odds['event_date'] = pd.to_datetime(
        all_odds["event_date"], format="mixed"
    ).dt.date
    print(f'All odds shape dupes: {all_odds.shape}')
    all_odds = all_odds.drop_duplicates(subset=['event_date', 'blue_fighter', 'red_fighter'])
    print(f'All odds shape NO dupes: {all_odds.shape}')


    t1 = time.time()
    odds_stats_df, upcoming_df = features.build_all_stats(all_stats, missing_stats.iloc[:5], all_odds, missing_odds.iloc[:5], ignore_upcoming=True)
    t2 = time.time()
    print(f"Time to merge stats and odds: {t2-t1:.2f} seconds")

    return missing_stats, missing_odds, odds_stats_df[odds_stats_df['date'] >= prev_fight_date]


def returns_by_date():
    
    df_ml = pd.read_csv(config.ml_history_fp)
    df_parlay = pd.read_csv(config.parlay_history_fp)

    types = ['open', 'close1', 'close2']
    ml_pct_returns = {type_:[] for type_ in types}
    parlay_pct_returns = {type_:[] for type_ in types}

    ml_pct_returns['date'] = []
    parlay_pct_returns['date'] = [] 

    for date, group in df_ml.groupby('date'): 
        
        parlay_group = df_parlay[df_parlay['date']==date]

        for type_ in types: 

            total_net_odds = np.sum(np.where(group[f'fstar_{type_}'] > 0, group[f'net_odds_{type_}'], 0))
            ml_pct_returns[type_].append(total_net_odds)

            valid_parlay = parlay_group[f'{type_}_fstar'].iloc[0] > 0 
            total_net_odds = parlay_group[f'{type_}_net_odds'].iloc[0] if valid_parlay else 0 
            parlay_pct_returns[type_].append(total_net_odds)
        
        ml_pct_returns['date'].append(date)
        parlay_pct_returns['date'].append(date)

    df_ml_pct = pd.DataFrame(ml_pct_returns)
    df_parlay_pct = pd.DataFrame(parlay_pct_returns)

    commit_if_changed(df_ml_pct, config.ml_pct_returns_fp, f'Saving Percent Returns ML')
    commit_if_changed(df_parlay_pct, config.parlay_pct_returns_fp, f'Saving Percent Returns Parlay')

    return df_ml_pct, df_parlay_pct



if __name__ == "__main__":
    archive_results()
    returns_by_date()
