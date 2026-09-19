
import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.kelly_worker import parlay_top_ev, run_per_bet_scaling
from ufc_betting.UpcomingPicks.set_column_names import set_ml_bets_cols, set_parlay_cols, get_ml_bet_cols, get_parlay_cols
from ufc_betting.UpcomingPicks.test_helpers import df_bets_tests, df_parlay_tests
from ufc_betting.UpcomingPicks.model_helpers import logit_predict, xgboost_predict
from ufc_betting.UpcomingPicks.data_helpers import get_X_stacked, merge_bets_types, merge_parlay_types, get_bets_input, get_bets_pkt, get_parlay_input, get_parlay_pkt
from ufc_betting.config import config

PARLAY_SIZE = config.parlay_top_ev


def manage_bets(
        df, pred_winner_bool, proba_red, proba_blue, real_odds, fair_odds, bankroll, type, mdd_ml, N_ml, df_bets_combined, valid_mask, required_df_idx
    ):
    bets_input_df = get_bets_input(
        df=df, 
        y_hat=pred_winner_bool, 
        proba_red=proba_red, 
        proba_blue=proba_blue, 
        real_odds=real_odds, 
        fair_odds=fair_odds
    )

    df_per_bet = run_per_bet_scaling(
        bets_df=bets_input_df, 
        max_drawdown=mdd_ml, 
        bankroll=bankroll, 
        N=N_ml
    )

    df_bets = get_bets_pkt(
        bets_input_df=bets_input_df, 
        df_per_bet=df_per_bet,
        type_=type,
        required_idx=required_df_idx, 
        all_na=False
    )
    
    df_bets_tests(
        df_bets=df_bets, 
        df_bets_combined=df_bets_combined, 
        valid_mask=valid_mask, 
        choice_ev=bets_input_df['choice_ev'], 
        fstar_list=df_per_bet['fstar_scaled']
    )

    df_bets_combined = merge_bets_types(
        df_bets=df_bets, 
        df_bets_combined=df_bets_combined
    )

    return df_bets_combined, bets_input_df

def manage_parlay(df_parlay_combined, bets_input_df, fighter_red, fighter_blue, dates, required_df_idx, bankroll, type, mdd_parlay, N_parlay): 
    parlay_input_df = get_parlay_input(
        bets_input_df=bets_input_df, 
        fighter_red=fighter_red, 
        fighter_blue=fighter_blue,
        dates=dates,
        required_idx=required_df_idx
    )

    # parlay_input has required_idx 
    # index is reset if NaNs, np.arange(2)
    # if not nans, index is preserved from parlay_input_df
    # parlay top ev now has 2 rows 
    df_parlay = parlay_top_ev(
        parlay_input_df, 
        bankroll, 
        type, 
        top_n=[0,1],
        parlay_mdd = mdd_parlay,
        N = N_parlay
    )
    
    df_parlay_final = get_parlay_pkt(
        df_parlay=df_parlay,
        type=type,
        all_na=False
    )

    df_parlay_tests(
        df_parlay=df_parlay_final, 
        choice_ev=bets_input_df['choice_ev']
    )

    # want to preserve df_parlay_final index in the merge
    df_parlay_combined = merge_parlay_types(
        df_parlay=df_parlay_final, 
        df_parlay_combined=df_parlay_combined,
        odds_type=type
    )
    return df_parlay_combined


def betting_pipeline(
    upcoming_df: pd.DataFrame, 
    feats_list:list[list[str]], 
    model_list:list, 
    scaler_list:list,
    encoder_list:list, 
    type_list:list[str], 
    fair_odds_list:list[list[str]], 
    real_odds_list:list[list[str]], 
    bankroll: float, 
    mdd_ml_arr:list[float],
    mdd_parlay_arr: list[float], 
    N_ml_arr:list[int],
    N_parlay_arr:list[int],
):
    """ upcoming df comes from features_pipeline.py """

    other_cols = ['date', 'fighter_red', 'fighter_blue', 
                  'open_red', 'open_blue', 'close1_red', 
                  'close2_red', 'close1_blue', 'close2_blue']


    df = upcoming_df.copy().reset_index(drop=True)
    required_df_idx = df.index

    df_parlay_combined = pd.DataFrame()
    df_proba = pd.DataFrame(
        columns=['proba_red_open', 'proba_blue_open', 'proba_red_close1', 
                 'proba_blue_close1', 'proba_red_close2','proba_blue_close2'], 
        index=required_df_idx
    )
    df_bets_combined = pd.DataFrame(
        df[other_cols].values, 
        columns=other_cols, 
        index=required_df_idx
    )
    fighter_red = df["fighter_red"]
    fighter_blue = df["fighter_blue"]
    dates = df["date"]

    data_list = []
    print(mdd_ml_arr)
    for i in range(len(feats_list)):
        dat = {
            'feats':feats_list[i],
            'model':model_list[i],
            'scaler':scaler_list[i],
            'encoder':encoder_list[i],
            'type':type_list[i],
            'fair_odds':fair_odds_list[i],
            'real_odds':real_odds_list[i],
            'mdd_ml':mdd_ml_arr[i],
            'mdd_parlay':mdd_parlay_arr[i],
            'N_ml':N_ml_arr[i],
            'N_parlay':N_parlay_arr[i]
        }
        data_list.append(dat)

    for i, dat in enumerate(data_list):
        
        model = dat['model']
        scaler = dat['scaler']
        encoder = dat['encoder']
        feats = dat['feats']
        type = dat['type']
        real_odds = dat['real_odds']
        fair_odds = dat['fair_odds']
        mdd_ml = dat['mdd_ml']
        mdd_parlay = dat['mdd_parlay']
        N_ml = dat['N_ml']
        N_parlay = dat['N_parlay']

        valid_mask = ~df[feats].isna().any(axis=1)
        y_hat = pd.Series(np.nan, index=required_df_idx)

        # split features by dtype
        num_feats = df[feats].select_dtypes(exclude='category').columns
        cat_feats = df[feats].select_dtypes(include='category').columns
        df_valid_num = df.loc[valid_mask, num_feats]

        if df_valid_num.shape[0] == 0:
            proba_red = pd.Series(np.nan, index=required_df_idx)
            proba_blue = pd.Series(np.nan, index=required_df_idx)
            df_proba[f'proba_red_{type}'] = proba_red
            df_proba[f'proba_blue_{type}'] = proba_blue

            df_bets = set_ml_bets_cols(type, {}, required_df_idx, all_na=True)
            df_bets_combined = merge_bets_types(df_bets, df_bets_combined)

            df_parlay = set_parlay_cols(type, {}, np.arange(2), all_na=True)
            df_parlay_combined = merge_parlay_types(df_parlay, df_parlay_combined, odds_type=type)
            continue

        rename_odds = type if type == 'close1' or type == 'close2' else None
        y_hat, proba_red, proba_blue = logit_predict(
            model=model, 
            df=df, 
            y_hat=y_hat, 
            feats=feats, 
            num_feats=num_feats, 
            cat_feats=cat_feats, 
            valid_mask=valid_mask, 
            scaler=scaler, 
            cat_encoder=encoder,
            required_df_idx=required_df_idx,
            rename_odds=rename_odds
        )

        pred_winner_bool = y_hat
        df_proba[f'proba_red_{type}'] = proba_red
        df_proba[f'proba_blue_{type}'] = proba_blue

        df_bets_combined, bets_input_df = manage_bets(
            df, pred_winner_bool, proba_red, proba_blue,
            real_odds, fair_odds, bankroll, type, mdd_ml, N_ml, df_bets_combined, valid_mask, required_df_idx
        )
        df_parlay_combined = manage_parlay(
            df_parlay_combined, bets_input_df, fighter_red, fighter_blue, dates,
            required_df_idx, bankroll, type, mdd_parlay, N_parlay
        )
        

    return df_bets_combined, df_parlay_combined

    # X_stacked = get_X_stacked(
    #     df=df, 
    #     df_proba=df_proba, 
    #     df_bets_combined=df_bets_combined, 
    #     required_df_idx=required_df_idx
    # )

    # bets_stacking, parlay_stacking = xgboost_stacking(
    #     xgb=xgb_stack,
    #     df=df,
    #     required_idx=required_df_idx,  
    #     feats=feats,
    #     X_stacked=X_stacked,
    #     fair_odds_arr=fair_odds_list[1:],
    #     real_odds_arr=real_odds_list[1:], 
    #     types_arr=type_list[1:], 
    #     mdd_ml_arr=mdd_ml_stack_arr,
    #     mdd_parlay_arr=mdd_parlay_stack_arr,
    #     N_ml_arr=N_ml_stack_arr,
    #     N_parlay_arr=N_parlay_stack_arr, 
    #     bankroll=bankroll
    # )
    # df_bets_combined = pd.concat([df_bets_combined, bets_stacking], axis=1)
    # df_parlay_combined = pd.concat([df_parlay_combined, parlay_stacking], axis=1)


    


def seperate_bets_dfs(df_bets, df_parlay, types):
    dfs = []
    dfs_parlay = []

    for type in types: 

        columns = get_ml_bet_cols(type).values()
        dfs.append(df_bets[columns])

        c_parlay = get_parlay_cols(type).values()
        dfs_parlay.append(df_parlay[c_parlay])

    return dfs, dfs_parlay
