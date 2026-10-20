import statsmodels.api as sm

import joblib

from ufc_betting.config import config
from ufc_betting.UpcomingPicks.betting_pipeline import betting_pipeline, seperate_bets_dfs


# Use the train + validation fit and its matching preprocessing artifacts.
model_open = sm.load(config.model_open_test_path)
model_open_se = sm.load(config.model_open_test_se_path)
scaler_open = joblib.load(config.scaler_open_test_path)

encoder_open = joblib.load(config.encoder_open_test_path)
encoder_list = [encoder_open, encoder_open, encoder_open] # using same encoder for all models since they have same features

model_list = [model_open, model_open, model_open]
scaler_list = [scaler_open, scaler_open, scaler_open]
# The saved model's fitted inputs supersede the older feature lists in config.
opening_features = list(scaler_open.feature_names_in_) + list(encoder_open.feature_names_in_)
feats_list = [
    [name.replace('open', stage) for name in opening_features]
    for stage in ('open', 'close1', 'close2')
]

# defines the odds on streamlit page
type_list = ['open', 'close1', 'close2']

# 0 for blue, 1 for red 
fair_odds_list = [['dec_fair_open_blue', 'dec_fair_open_red'], 
                  ['dec_fair_close1_blue', 'dec_fair_close1_red'], 
                  ['dec_fair_close2_blue', 'dec_fair_close2_red']]

real_odds_list = [['dec_open_blue', 'dec_open_red'], 
                  ['dec_close1_blue', 'dec_close1_red'], 
                  ['dec_close2_blue', 'dec_close2_red']]

def generate_bets(df, select_odds=None, return_event_ev=False):
    """ handles categorical columns before generating bets """

    df = df.copy()
    cat_cols = encoder_open.feature_names_in_
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    pipeline_results = betting_pipeline(
        upcoming_df=df, 
        feats_list=feats_list, 
        model_list=model_list,
        scaler_list=scaler_list, 
        encoder_list=encoder_list,
        type_list=type_list,
        fair_odds_list=fair_odds_list, 
        real_odds_list=real_odds_list, 
        bankroll=config.bankroll, 
        model_se_list=[model_open_se] * len(type_list),
        return_event_ev=return_event_ev,
    )
    if return_event_ev:
        df_bets_all, df_parlay_all, df_event_ev = pipeline_results
    else:
        df_bets_all, df_parlay_all = pipeline_results
    
    df_bets_arr, df_parlay_arr = seperate_bets_dfs(df_bets_all, df_parlay_all, type_list)
    
    if select_odds: 
        df_bets = df_bets_arr[select_odds]
        df_parlay = df_parlay_arr[select_odds]
    
    else: 
        df_bets = df_bets_all
        df_parlay = df_parlay_all

    if return_event_ev:
        return df_bets, df_parlay, df_event_ev
    return df_bets, df_parlay 
