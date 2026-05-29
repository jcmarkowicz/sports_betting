from xgboost import XGBClassifier
import statsmodels.api as sm

import joblib

from config import config
from UpcomingPicks.betting_pipeline import betting_pipeline, seperate_bets_dfs


model_open = sm.load(config.model_open_path)
model_close1 = sm.load(config.model_close1_path)
model_close2 = sm.load(config.model_close2_path)

xgboost_stack = XGBClassifier()
xgboost_stack.load_model(config.xgb_stack_path)

scaler_open = joblib.load(config.scaler_open_path)
scaler_close1 = joblib.load(config.scaler_close1_path)
scaler_close2 = joblib.load(config.scaler_close2_path)

model_list = [model_open, model_close1, model_close2]
scaler_list = [scaler_open, scaler_close1, scaler_close2]
feats_list = [config.open_feats, config.close1_feats, config.close2_feats]

type_list = ['open', 'close1', 'close2']

# 0 for blue, 1 for red 
fair_odds_list = [['dec_fair_open_blue', 'dec_fair_open_red'], 
                  ['dec_fair_close1_blue', 'dec_fair_close1_red'], 
                  ['dec_fair_close2_blue', 'dec_fair_close2_red']]

real_odds_list = [['dec_open_blue', 'dec_open_red'], 
                  ['dec_close1_blue', 'dec_close1_red'], 
                  ['dec_close2_blue', 'dec_close2_red']]


def generate_bets(df, select_odds=None):

    df['math_red'] = df['math_red'].astype('category')
    df['math_blue'] = df['math_blue'].astype('category')
    df['elo_pred'] = df['elo_pred'].astype('category')

    df_bets_all, df_parlay_all = betting_pipeline(
                                                upcoming_df=df, 
                                                feats_list=feats_list, 
                                                model_list=model_list,
                                                xgb_stack=xgboost_stack, 
                                                scaler_list=scaler_list, 
                                                type_list=type_list,
                                                fair_odds_list=fair_odds_list, 
                                                real_odds_list=real_odds_list, 
                                                bankroll=config.bankroll, 
                                                mdd_ml_arr=config.mdd_ml,
                                                mdd_parlay_arr=config.mdd_parlay, 
                                                N_ml_arr=config.N_ml,
                                                N_parlay_arr=config.N_parlay,
                                                mdd_ml_stack_arr=config.mdd_ml_stack,
                                                mdd_parlay_stack_arr=config.mdd_parlay_stack,
                                                N_ml_stack_arr=config.N_ml_stack,
                                                N_parlay_stack_arr=config.N_parlay_stack
    )
    
    df_bets_arr, df_parlay_arr = seperate_bets_dfs(df_bets_all, df_parlay_all, type_list)
    
    if select_odds: 
        df_bets = df_bets_arr[select_odds]
        df_parlay = df_parlay_arr[select_odds]
    
    else: 
        df_bets = df_bets_all
        df_parlay = df_parlay_all

    return df_bets, df_parlay 
