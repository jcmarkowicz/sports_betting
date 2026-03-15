
import numpy as np 
import pandas as pd 

import statsmodels.api as sm

import os 
import sys 
import joblib
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from ufc_upcoming_analysis.betting_strategy import betting_pipeline, seperate_bets_dfs

BASE_DIR = Path(__file__).resolve().parents[1]

open_feats = [
            'proba_fair_open_diff', 'reach_diff', 
            
            'sub_att_pm_red', 'sub_att_pm_blue',
            'ratio_control_diff',

            'td_landed_pm_diff',  
            'ratio_td_diff', 
            'adjusted_td_red', 'adjusted_td_blue',

            'sig_str_absorbed_total_diff', 
            'sig_str_accuracy_pct_diff',
            'sig_str_defense_pct_diff',
            'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
            
            'win_pct_red', 'win_pct_blue',
            'win_streak_diff', 'lose_streak_diff',
            'elo_red', 'elo_blue', 'elo_pred', 'age_red', 'age_blue'
            ]

close1_feats = [
                  'proba_fair_close1_diff', 'proba_fair_open_diff', 'reach_diff', 
                  
                  'sub_att_pm_red', 'sub_att_pm_blue',
                  'ratio_control_diff',

                  'td_landed_pm_diff',  
                  'ratio_td_diff', 
                  'adjusted_td_red', 'adjusted_td_blue',

                  'sig_str_absorbed_total_diff', 
                  'sig_str_accuracy_pct_diff',
                  'sig_str_defense_pct_diff',
                  'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
                  
                  'win_pct_red', 'win_pct_blue',
                  'win_streak_diff', 'lose_streak_diff',
                  'elo_red', 'elo_blue', 'elo_pred', 'age_red', 'age_blue',
                  ]

close2_feats = [
                  'proba_fair_close2_diff', 'proba_fair_open_diff', 'reach_diff', 
                  
                  'sub_att_pm_red', 'sub_att_pm_blue',
                  'ratio_control_diff',

                  'td_landed_pm_diff',  
                  'ratio_td_diff', 
                  'adjusted_td_red', 'adjusted_td_blue',

                  'sig_str_absorbed_total_diff', 
                  'sig_str_accuracy_pct_diff',
                  'sig_str_defense_pct_diff',
                  'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
                  
                  'win_pct_red', 'win_pct_blue',
                  'win_streak_diff', 'lose_streak_diff',
                  'elo_red', 'elo_blue', 'elo_pred', 'age_red', 'age_blue',
                  ]


model_open = sm.load(BASE_DIR / "Data" / "saved_models" / "logit_model_open.pkl")
model_close1 = sm.load(BASE_DIR / "Data" / "saved_models" / "logit_model_close1.pkl")
model_close2 = sm.load( BASE_DIR / "Data" / "saved_models" / "logit_model_close2.pkl")

scaler_open = joblib.load(BASE_DIR / "Data" / "saved_models" / "scaler_open.pkl")
scaler_close1 = joblib.load(BASE_DIR / "Data" / "saved_models" / "scaler_close1.pkl")
scaler_close2 = joblib.load(BASE_DIR / "Data" / "saved_models" / "scaler_close2.pkl")

feats_list = [open_feats, close1_feats, close2_feats]
model_list = [model_open, model_close1, model_close2]
scaler_list = [scaler_open, scaler_close1, scaler_close2]

type_list = ['open', 'close1', 'close2']
fair_odds_list = [['dec_fair_open_blue', 'dec_fair_open_red'], 
                  ['dec_fair_close1_blue', 'dec_fair_close1_red'], 
                  ['dec_fair_close2_blue', 'dec_fair_close2_red']]

real_odds_list = [['dec_open_blue', 'dec_open_red'], 
                  ['dec_close1_blue', 'dec_close1_red'], 
                  ['dec_close2_blue', 'dec_close2_red'] ]


def generate_bets(df, select_odds=None, bankroll=500):

    df['math_red'] = df['math_red'].astype('category')
    df['math_blue'] = df['math_blue'].astype('category')
    df['elo_pred'] = df['elo_pred'].astype('category')

    df_bets_all, df_parlay_all = betting_pipeline(df, 
                                                feats_list=feats_list, model_list=model_list, 
                                                scaler_list=scaler_list, type_list=type_list,
                                                fair_odds_list=fair_odds_list, 
                                                real_odds_list=real_odds_list, 
                                                bankroll=bankroll, max_drawdown=0.3, N=250)
    
    df_bets_all[['open_red', 'open_blue', 'close1_red', 'close1_blue', 'close2_red', 'close2_blue']] = df[['open_red', 'open_blue', 'close1_red', 'close1_blue', 'close2_red', 'close2_blue']]
    df_bets_arr, df_parlay_arr = seperate_bets_dfs(df_bets_all, df_parlay_all, type_list)
    
    if select_odds: 
        df_bets = df_bets_arr[select_odds]
        df_parlay = df_parlay_arr[select_odds]
    
    else: 
        df_bets = df_bets_all
        df_parlay = df_parlay_all

    return df_bets, df_parlay 
