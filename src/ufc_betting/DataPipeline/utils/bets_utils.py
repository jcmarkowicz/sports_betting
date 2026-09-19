from xgboost import XGBClassifier
import statsmodels.api as sm

import joblib

from ufc_betting.config import config
from ufc_betting.UpcomingPicks.betting_pipeline import betting_pipeline, seperate_bets_dfs


model_open = sm.load(config.model_open_path)
model_close1 = sm.load(config.model_close1_path)
model_close2 = sm.load(config.model_close2_path)

xgboost_stack = XGBClassifier()
# xgboost_stack.load_model(config.xgb_stack_path)

scaler_open = joblib.load(config.scaler_open_path)
scaler_close1 = joblib.load(config.scaler_close1_path)
scaler_close2 = joblib.load(config.scaler_close2_path)

encoder_open = joblib.load(config.encoder_open_path)
encoder_list = [encoder_open, encoder_open, encoder_open] # using same encoder for all models since they have same features

model_list = [model_open, model_open, model_open]
scaler_list = [scaler_open, scaler_open, scaler_open]
feats_list = [config.open_feats, config.open_close1_feats, config.open_close2_feats]

# defines the odds on streamlit page
type_list = ['open', 'close1', 'close2']

# 0 for blue, 1 for red 
fair_odds_list = [['dec_fair_open_blue', 'dec_fair_open_red'], 
                  ['dec_fair_close1_blue', 'dec_fair_close1_red'], 
                  ['dec_fair_close2_blue', 'dec_fair_close2_red']]

real_odds_list = [['dec_open_blue', 'dec_open_red'], 
                  ['dec_close1_blue', 'dec_close1_red'], 
                  ['dec_close2_blue', 'dec_close2_red']]

def generate_bets(df, select_odds=None):
    """ handles categorical columns before generating bets """

    cat_cols = ['math_red', 'math_blue', 'elo_pred', 'womens_fight']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    df_bets_all, df_parlay_all = betting_pipeline(
        upcoming_df=df, 
        feats_list=feats_list, 
        model_list=model_list,
        scaler_list=scaler_list, 
        encoder_list=encoder_list,
        type_list=type_list,
        fair_odds_list=fair_odds_list, 
        real_odds_list=real_odds_list, 
        bankroll=config.bankroll, 
        mdd_ml_arr=config.mdd_ml,
        mdd_parlay_arr=config.mdd_parlay, 
        N_ml_arr=config.N_ml,
        N_parlay_arr=config.N_parlay,
    )
    
    df_bets_arr, df_parlay_arr = seperate_bets_dfs(df_bets_all, df_parlay_all, type_list)
    
    if select_odds: 
        df_bets = df_bets_arr[select_odds]
        df_parlay = df_parlay_arr[select_odds]
    
    else: 
        df_bets = df_bets_all
        df_parlay = df_parlay_all

    return df_bets, df_parlay 
